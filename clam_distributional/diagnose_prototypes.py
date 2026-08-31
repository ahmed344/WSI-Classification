"""Diagnose repaired CLAM prototype histograms before retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .clam_dataset import collate_fn, create_bag_dataset
    from .config_loader import load_config
    from .prototype_init import initialize_prototypes_from_kmeans
    from .train_clam import create_model, seed_everything
except ImportError:
    from clam_dataset import collate_fn, create_bag_dataset
    from config_loader import load_config
    from prototype_init import initialize_prototypes_from_kmeans
    from train_clam import create_model, seed_everything


SOURCE_MODEL_SCHEMA = "canonical_clam_v7_detection_selective_norm"
MIN_PROTOTYPE_CV = 0.4
PROBE_FOLDS = 5


def parse_args() -> argparse.Namespace:
    """Parse prototype diagnostic command-line arguments.

    Args:
        None: Arguments are read from the command line.

    Returns:
        argparse.Namespace: Parsed configuration, checkpoint, and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Test repaired raw-feature prototypes on a trained v7 CLAM model."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def migrate_v7_model(
    checkpoint: Mapping[str, Any],
    target_config: Mapping[str, Any],
    device: torch.device,
) -> nn.Module:
    """Load compatible trained v7 weights into the repaired v8 architecture.

    Args:
        checkpoint (Mapping[str, Any]): Completed v7 checkpoint payload.
        target_config (Mapping[str, Any]): Configuration for the v8 model.
        device (torch.device): Device receiving the migrated model.

    Returns:
        nn.Module: V8 model with trained non-prototype weights and fresh prototypes.
    """
    if checkpoint.get("model_schema") != SOURCE_MODEL_SCHEMA:
        raise ValueError(
            f"Diagnostic source model_schema must be '{SOURCE_MODEL_SCHEMA}'."
        )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint must contain a model_state_dict mapping.")

    excluded_names = {
        "prototype_assignment.weight",
        "prototype_assignment.bias",
        "log_prototype_temperature",
    }
    compatible_state = {
        str(name): value
        for name, value in state_dict.items()
        if str(name) not in excluded_names
    }
    model = create_model(target_config).to(device)
    incompatible = model.load_state_dict(compatible_state, strict=False)
    if set(incompatible.missing_keys) != excluded_names:
        raise ValueError(
            "Unexpected missing keys while migrating v7 checkpoint: "
            + ", ".join(incompatible.missing_keys)
        )
    if incompatible.unexpected_keys:
        raise ValueError(
            "Unexpected v7 checkpoint keys: "
            + ", ".join(incompatible.unexpected_keys)
        )
    return model


def compute_histogram_cv(histograms: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute mean between-bag coefficient of variation across prototypes.

    Args:
        histograms (np.ndarray): Bag histograms shaped ``[N, K]``.

    Returns:
        Tuple[float, np.ndarray]: Mean CV and one CV value per prototype.
    """
    if histograms.ndim != 2 or histograms.shape[0] < 2 or histograms.shape[1] < 1:
        raise ValueError("histograms must have shape [N, K] with N >= 2 and K >= 1.")
    means = histograms.mean(axis=0)
    if np.any(means <= 0.0):
        raise ValueError("Prototype histogram means must be positive.")
    cv_per_prototype = histograms.std(axis=0, ddof=0) / means
    return float(cv_per_prototype.mean()), cv_per_prototype


def grouped_probe_scores(
    mean_std_features: np.ndarray,
    histograms: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    random_seed: int,
) -> Tuple[List[float], List[float]]:
    """Compare grouped linear probes with and without prototype histograms.

    Args:
        mean_std_features (np.ndarray): Normalized μ/σ blocks shaped ``[N, D]``.
        histograms (np.ndarray): Prototype histograms shaped ``[N, K]``.
        labels (np.ndarray): Integer bag labels shaped ``[N]``.
        groups (np.ndarray): Slide identifiers shaped ``[N]``.
        random_seed (int): Seed controlling grouped fold assignment and classifiers.

    Returns:
        Tuple[List[float], List[float]]: Per-fold balanced accuracies for μ/σ
        alone and μ/σ concatenated with ρ.
    """
    sample_count = mean_std_features.shape[0]
    if (
        mean_std_features.ndim != 2
        or histograms.ndim != 2
        or labels.shape != (sample_count,)
        or groups.shape != (sample_count,)
        or histograms.shape[0] != sample_count
    ):
        raise ValueError("Probe arrays must share a two-dimensional sample axis.")
    splitter = StratifiedGroupKFold(
        n_splits=PROBE_FOLDS,
        shuffle=True,
        random_state=random_seed,
    )
    augmented_features = np.concatenate((mean_std_features, histograms), axis=1)
    baseline_scores: List[float] = []
    augmented_scores: List[float] = []
    for train_indices, val_indices in splitter.split(
        mean_std_features, labels, groups
    ):
        baseline_probe = _linear_probe(random_seed)
        augmented_probe = _linear_probe(random_seed)
        baseline_probe.fit(mean_std_features[train_indices], labels[train_indices])
        augmented_probe.fit(augmented_features[train_indices], labels[train_indices])
        baseline_scores.append(
            float(
                balanced_accuracy_score(
                    labels[val_indices],
                    baseline_probe.predict(mean_std_features[val_indices]),
                )
            )
        )
        augmented_scores.append(
            float(
                balanced_accuracy_score(
                    labels[val_indices],
                    augmented_probe.predict(augmented_features[val_indices]),
                )
            )
        )
    return baseline_scores, augmented_scores


def _linear_probe(random_seed: int) -> Pipeline:
    """Build a fold-local standardized multinomial linear probe.

    Args:
        random_seed (int): Logistic-regression random state.

    Returns:
        Pipeline: Standardization and balanced logistic-regression pipeline.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def collect_validation_blocks(
    model: nn.Module,
    config: Mapping[str, Any],
    class_folders: Sequence[str],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect normalized statistics, histograms, labels, groups, and entropies.

    Args:
        model (nn.Module): Migrated model with repaired prototypes.
        config (Mapping[str, Any]): Checkpoint data and loader configuration.
        class_folders (Sequence[str]): Ordered class directory names.
        device (torch.device): Inference device.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        μ/σ features, histograms, labels, slide groups, and tile entropies.
    """
    dataset = create_bag_dataset(
        config,
        "val",
        class_folders=class_folders,
        bag_level="tissue",
    )
    if len(dataset) == 0:
        raise ValueError("Validation split must contain tissue bags.")
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("slide_evaluation_batch_size", 1)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    statistics_parts: List[np.ndarray] = []
    histogram_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    groups: List[str] = []
    entropy_parts: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Prototype diagnostic"):
            features = batch["features"].to(device)
            masks = batch["masks"].to(device)
            outputs = model(features, mask=masks, instance_eval=False)
            mean_block = model.distribution_mean_layernorm(
                outputs["distribution_mean"]
            )[:, 0]
            statistic_blocks = [mean_block]
            if bool(config.get("pooling_use_variance", False)):
                std_layernorm = model.distribution_std_layernorm
                if std_layernorm is None:
                    raise RuntimeError("Standard-deviation LayerNorm is missing.")
                statistic_blocks.append(
                    std_layernorm(outputs["distribution_std"])[:, 0]
                )
            statistics_parts.append(
                torch.cat(statistic_blocks, dim=-1).cpu().numpy()
            )
            histogram_parts.append(
                outputs["prototype_histogram"][:, 0].cpu().numpy()
            )
            label_parts.append(batch["labels"].cpu().numpy())
            groups.extend(
                str(provenance["slide_key"])
                for provenance in batch["provenance"]
            )
            assignments = outputs["prototype_assignments"][masks]
            safe_assignments = assignments.clamp_min(
                torch.finfo(assignments.dtype).tiny
            )
            entropy_parts.append(
                (
                    -(assignments * safe_assignments.log()).sum(dim=-1)
                    / np.log(assignments.shape[-1])
                )
                .cpu()
                .numpy()
            )
    return (
        np.concatenate(statistics_parts, axis=0),
        np.concatenate(histogram_parts, axis=0),
        np.concatenate(label_parts, axis=0),
        np.asarray(groups),
        np.concatenate(entropy_parts, axis=0),
    )


def diagnose_prototypes(
    config_path: Optional[str] = None,
    checkpoint_override: Optional[str] = None,
    output_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the §5.3 migration diagnostic and write its JSON report.

    Args:
        config_path (Optional[str]): Runtime YAML path, or ``None`` for default.
        checkpoint_override (Optional[str]): Explicit completed v7 checkpoint.
        output_override (Optional[str]): Explicit report path.

    Returns:
        Dict[str, Any]: Serializable prototype diagnostic report.
    """
    runtime_config = load_config(config_path)
    checkpoint_path = Path(
        checkpoint_override or str(runtime_config["paths"]["checkpoint"])
    ).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Prototype diagnostic checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config")
    class_folders = checkpoint.get("class_folders")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Checkpoint must contain a configuration mapping.")
    if not isinstance(class_folders, list) or not all(
        isinstance(class_name, str) for class_name in class_folders
    ):
        raise ValueError("Checkpoint must contain ordered class_folders.")

    target_config = dict(checkpoint_config)
    target_config["pooling_prototype_temperature"] = None
    target_config["prototype_assignment_entropy_target"] = float(
        runtime_config["prototype_assignment_entropy_target"]
    )
    seed_everything(int(target_config["random_seed"]))
    model = migrate_v7_model(checkpoint, target_config, device)
    train_dataset = create_bag_dataset(
        target_config,
        "train",
        class_folders=class_folders,
    )
    initialize_prototypes_from_kmeans(model, train_dataset, target_config, device)
    statistics, histograms, labels, groups, entropies = collect_validation_blocks(
        model, target_config, class_folders, device
    )
    mean_cv, cv_per_prototype = compute_histogram_cv(histograms)
    baseline_folds, augmented_folds = grouped_probe_scores(
        statistics,
        histograms,
        labels,
        groups,
        int(target_config["random_seed"]),
    )
    baseline_ba = float(np.mean(baseline_folds))
    augmented_ba = float(np.mean(augmented_folds))
    probe_delta = augmented_ba - baseline_ba
    temperature_parameter = getattr(model, "log_prototype_temperature", None)
    if not isinstance(temperature_parameter, nn.Parameter):
        raise RuntimeError("Migrated model does not expose prototype temperature.")
    report: Dict[str, Any] = {
        "source_checkpoint": str(checkpoint_path),
        "source_model_schema": str(checkpoint["model_schema"]),
        "target_contract": "canonical_clam_v8_raw_feature_prototypes",
        "prototype_temperature": float(
            temperature_parameter.exp().detach().cpu().item()
        ),
        "median_normalized_assignment_entropy": float(np.median(entropies)),
        "mean_rho_cv": mean_cv,
        "rho_cv_per_prototype": cv_per_prototype.tolist(),
        "minimum_required_rho_cv": MIN_PROTOTYPE_CV,
        "mu_sigma_balanced_accuracy": baseline_ba,
        "mu_sigma_rho_balanced_accuracy": augmented_ba,
        "probe_delta_balanced_accuracy": probe_delta,
        "mu_sigma_fold_scores": baseline_folds,
        "mu_sigma_rho_fold_scores": augmented_folds,
        "rho_cv_pass": mean_cv > MIN_PROTOTYPE_CV,
        "probe_pass": probe_delta > 0.0,
        "overall_pass": mean_cv > MIN_PROTOTYPE_CV and probe_delta > 0.0,
        "validation_bags": int(histograms.shape[0]),
        "validation_slides": int(np.unique(groups).size),
    }
    output_path = Path(
        output_override or str(checkpoint_path.parent / "prototype_diagnostic.json")
    ).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    print(
        "Prototype diagnostic: "
        f"mean_rho_cv={mean_cv:.4f}, probe_delta={probe_delta:+.4f}, "
        f"overall_pass={report['overall_pass']}; report={output_path}"
    )
    return report


def main() -> None:
    """Run the prototype diagnostic command-line entry point.

    Args:
        None: Arguments are read from the command line.

    Returns:
        None: A JSON report is written and a summary is printed.
    """
    args = parse_args()
    diagnose_prototypes(args.config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
