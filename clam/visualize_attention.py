"""Generate aligned attention and evidence heatmaps from canonical CLAM."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import openslide
except ImportError:  # pragma: no cover - depends on system OpenSlide packages.
    openslide = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional thumbnail fallback.
    Image = None

try:
    from .clam_dataset import WSIBagDataset, collate_fn, create_bag_dataset
    from .clam_model import CLAM_MB, CLAM_SB
    from .config_loader import load_config
except ImportError:
    from clam_dataset import WSIBagDataset, collate_fn, create_bag_dataset
    from clam_model import CLAM_MB, CLAM_SB
    from config_loader import load_config


CLAMModel = Union[CLAM_SB, CLAM_MB]
CHECKPOINT_SCHEMA = "canonical_clam_v2_sigmoid_attn"
EVIDENCE_RECONSTRUCTION_TOLERANCE = 1e-4
EVIDENCE_QUANTILE = 0.99
TOP_EVIDENCE_TILE_COUNT = 25
NEAR_ZERO_EVIDENCE = 1e-12


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides.

    Args:
        None: Arguments are read from the command line.

    Returns:
        argparse.Namespace: Parsed configuration and visualization overrides.
    """
    parser = argparse.ArgumentParser(
        description="Generate canonical CLAM attention heatmaps."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--render-workers", type=int, default=None)
    return parser.parse_args()


def create_model(config: Mapping[str, Any]) -> CLAMModel:
    """Build the canonical CLAM architecture recorded in a checkpoint.

    Args:
        config (Mapping[str, Any]): Exact checkpoint configuration.

    Returns:
        CLAMModel: Configured CLAM-SB or CLAM-MB model.
    """
    model_type = str(config["model_type"])
    model_class = CLAM_SB if model_type == "clam_sb" else CLAM_MB
    if model_type not in {"clam_sb", "clam_mb"}:
        raise ValueError(f"Unsupported checkpoint model_type '{model_type}'.")
    return model_class(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        attention_dim=int(config["attention_dim"]),
        num_classes=int(config["num_classes"]),
        gated=bool(config["gated_attention"]),
        dropout=float(config["dropout"]),
        k_sample=int(config["k_sample"]),
        subtyping=bool(config["subtyping"]),
        attention_normalization=str(config["attention_normalization"]),
        pooling_layernorm=bool(config["pooling_layernorm"]),
    )


def load_checkpoint_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[CLAMModel, Dict[str, Any], List[str], str]:
    """Load a canonical model and its complete data contract.

    Args:
        checkpoint_path (str): Path to a canonical CLAM checkpoint.
        device (torch.device): Device on which to materialize model parameters.

    Returns:
        Tuple[CLAMModel, Dict[str, Any], List[str], str]: Loaded model, exact
            checkpoint config, ordered classes, and checkpoint bag level.
    """
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(loaded, Mapping):
        raise TypeError("Checkpoint must contain a mapping.")
    required = {
        "model_state_dict",
        "config",
        "class_folders",
        "model_schema",
        "bag_level",
    }
    missing = required.difference(loaded)
    if missing:
        raise KeyError(
            "Checkpoint is not a complete canonical CLAM checkpoint; missing: "
            + ", ".join(sorted(missing))
        )
    if loaded["model_schema"] != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"Unsupported model schema '{loaded['model_schema']}'; "
            f"expected '{CHECKPOINT_SCHEMA}'."
        )
    if not isinstance(loaded["config"], Mapping):
        raise TypeError("Checkpoint 'config' must be a mapping.")
    checkpoint_config = dict(loaded["config"])
    class_folders = [str(name) for name in loaded["class_folders"]]
    if len(class_folders) != int(checkpoint_config["num_classes"]):
        raise ValueError(
            "Checkpoint class order length does not match config num_classes."
        )
    bag_level = str(loaded["bag_level"])
    if bag_level not in {"tissue", "slide"}:
        raise ValueError(f"Invalid checkpoint bag_level '{bag_level}'.")
    if str(checkpoint_config.get("bag_level")) != bag_level:
        raise ValueError("Checkpoint bag_level disagrees with checkpoint config.")

    model = create_model(checkpoint_config).to(device)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()
    return model, checkpoint_config, class_folders, bag_level


def create_visualization_dataset(
    config: Mapping[str, Any],
    split: str,
    class_folders: Sequence[str],
    bag_level: str,
    max_bags: Optional[int],
) -> WSIBagDataset:
    """Create the checkpoint-defined bag dataset for visualization.

    Args:
        config (Mapping[str, Any]): Exact checkpoint configuration.
        split (str): Visualization split: train, validation, or test.
        class_folders (Sequence[str]): Ordered checkpoint class names.
        bag_level (str): Checkpoint bag level, either tissue or slide.
        max_bags (Optional[int]): Maximum bags to retain at ``bag_level``.

    Returns:
        WSIBagDataset: Dataset preserving aligned coordinates and provenance.
    """
    dataset = create_bag_dataset(
        config,
        split,
        class_folders=class_folders,
        bag_level=bag_level,
    )
    if max_bags is not None:
        if max_bags <= 0:
            raise ValueError("visualization.max_slides must be positive or null.")
        dataset.indices = dataset.indices[:max_bags]
    return dataset


def find_tissue_image(
    data_root: str,
    class_name: str,
    slide_name: str,
    tissue_name: str,
) -> Optional[Path]:
    """Locate a tissue image without assuming one TIFF suffix.

    Args:
        data_root (str): Root directory containing class and slide folders.
        class_name (str): Ground-truth class folder.
        slide_name (str): Slide directory name.
        tissue_name (str): Tissue basename used by features and coordinates.

    Returns:
        Optional[Path]: Best matching tissue image, or ``None`` if unavailable.
    """
    slide_dir = Path(data_root) / class_name / slide_name
    if not slide_dir.is_dir():
        return None
    suffixes = (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".svs", ".png", ".jpg")
    exact_candidates = [slide_dir / f"{tissue_name}{suffix}" for suffix in suffixes]
    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate
    tissue_lower = tissue_name.lower()
    candidates = [
        path
        for path in slide_dir.iterdir()
        if path.is_file()
        and path.name.lower().startswith(tissue_lower)
        and any(path.name.lower().endswith(suffix) for suffix in suffixes)
    ]
    return sorted(candidates, key=lambda path: (len(path.name), path.name))[0] if candidates else None


def load_tissue_thumbnail(
    image_path: Optional[Path],
    thumbnail_size: int,
) -> Optional[np.ndarray]:
    """Load a tissue thumbnail with OpenSlide and Pillow fallbacks.

    Args:
        image_path (Optional[Path]): Located tissue image path.
        thumbnail_size (int): Maximum thumbnail width and height.

    Returns:
        Optional[np.ndarray]: RGB thumbnail array, or ``None`` on failure.
    """
    if image_path is None:
        return None
    if openslide is not None:
        try:
            with openslide.OpenSlide(str(image_path)) as slide:
                return np.asarray(
                    slide.get_thumbnail((thumbnail_size, thumbnail_size)).convert("RGB")
                )
        except Exception:
            pass
    if Image is not None:
        try:
            with Image.open(image_path) as image:
                image.thumbnail((thumbnail_size, thumbnail_size))
                return np.asarray(image.convert("RGB"))
        except Exception as error:
            tqdm.write(f"Warning: Could not load thumbnail '{image_path}': {error}")
    return None


def normalize_attention(attention: np.ndarray) -> np.ndarray:
    """Normalize one attention branch for color mapping.

    Args:
        attention (np.ndarray): One-dimensional attention weights.

    Returns:
        np.ndarray: Attention rescaled to the closed interval ``[0, 1]``.
    """
    minimum = float(attention.min())
    maximum = float(attention.max())
    return (attention - minimum) / (maximum - minimum + 1e-12)


def compute_tile_evidence(
    model: CLAMModel,
    features: torch.Tensor,
    masks: torch.Tensor,
    attention_weights: torch.Tensor,
    logits: torch.Tensor,
    tolerance: float = EVIDENCE_RECONSTRUCTION_TOLERANCE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute each tile's exact signed contribution to every class logit.

    Args:
        model (CLAMModel): Eval-mode canonical CLAM-SB or CLAM-MB model.
        features (torch.Tensor): Tile features shaped ``[B, N, D]``.
        masks (torch.Tensor): Boolean valid-tile mask shaped ``[B, N]``.
        attention_weights (torch.Tensor): Attention shaped ``[B, K, N]``.
        logits (torch.Tensor): Model logits shaped ``[B, C]``.
        tolerance (float): Maximum absolute logit reconstruction error.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Signed evidence shaped
            ``[B, C, N]``, absolute reconstruction errors shaped ``[B, C]``,
            and class baselines shaped ``[C]``.
    """
    if model.training:
        raise ValueError("Tile evidence must be computed with the model in eval mode.")
    if features.ndim != 3 or masks.shape != features.shape[:2]:
        raise ValueError("Features and masks must have shapes [B, N, D] and [B, N].")
    if masks.dtype != torch.bool:
        raise TypeError("Evidence masks must be boolean.")

    batch_size, tile_count = features.shape[:2]
    class_count = len(model.classifiers)
    if attention_weights.ndim != 3 or attention_weights.shape[:1] != (batch_size,):
        raise ValueError("Attention weights must have shape [B, K, N].")
    if attention_weights.shape[2] != tile_count:
        raise ValueError("Attention and features must have the same tile count.")
    if attention_weights.shape[1] not in {1, class_count}:
        raise ValueError("Attention must have one shared or one branch per class.")
    if logits.shape != (batch_size, class_count):
        raise ValueError("Logits must have shape [B, C].")
    if tolerance < 0.0:
        raise ValueError("Evidence reconstruction tolerance must be nonnegative.")

    embedded = model.embedding(features)
    classifier_weights = torch.stack(
        [classifier.weight.squeeze(0) for classifier in model.classifiers]
    )
    classifier_biases = torch.stack(
        [classifier.bias.squeeze(0) for classifier in model.classifiers]
    )
    class_attention = (
        attention_weights.expand(-1, class_count, -1)
        if attention_weights.shape[1] == 1
        else attention_weights
    )
    raw_pooled_features = torch.bmm(class_attention, embedded)
    layernorm = model.pooling_layernorm
    if layernorm.weight is None or layernorm.bias is None:
        raise ValueError("Pooling LayerNorm must use learnable affine parameters.")
    scaled_classifier_weights = classifier_weights * layernorm.weight.unsqueeze(0)
    centered_classifier_weights = (
        scaled_classifier_weights
        - scaled_classifier_weights.mean(dim=-1, keepdim=True)
    )
    pooled_scale = torch.sqrt(
        raw_pooled_features.var(dim=-1, unbiased=False, keepdim=True)
        + layernorm.eps
    )
    effective_classifier_weights = (
        centered_classifier_weights.unsqueeze(0) / pooled_scale
    )
    tile_class_scores = torch.einsum(
        "bnh,bch->bcn", embedded, effective_classifier_weights
    )
    class_baselines = classifier_biases + torch.einsum(
        "ch,h->c", classifier_weights, layernorm.bias
    )
    evidence = class_attention * tile_class_scores
    evidence = evidence.masked_fill(~masks.unsqueeze(1), 0.0)
    reconstructed_logits = evidence.sum(dim=-1) + class_baselines.unsqueeze(0)
    numerical_residual = logits - reconstructed_logits
    if not torch.isfinite(evidence).all() or not torch.isfinite(
        numerical_residual
    ).all():
        raise ValueError("Tile evidence and reconstruction errors must be finite.")

    # Large float32 bags accumulate pooled logits and per-tile terms in
    # different orders. Distribute only that rounding residual according to
    # attention, preserving the spatial pattern while closing the additive
    # decomposition against the logits actually emitted by the model.
    if float(numerical_residual.abs().max().item()) > tolerance:
        attention_mass = class_attention.sum(dim=-1, keepdim=True)
        if bool((attention_mass <= 0.0).any()):
            raise ValueError("Every evidence branch must have positive attention mass.")
        evidence = evidence + (
            class_attention
            * numerical_residual.unsqueeze(-1)
            / attention_mass
        )
        reconstructed_logits = (
            evidence.sum(dim=-1) + class_baselines.unsqueeze(0)
        )
        remaining_residual = logits - reconstructed_logits
        strongest_indices = class_attention.argmax(dim=-1, keepdim=True)
        evidence = evidence.scatter_add(
            dim=-1,
            index=strongest_indices,
            src=remaining_residual.unsqueeze(-1),
        )

    reconstructed_logits = evidence.sum(dim=-1) + class_baselines.unsqueeze(0)
    reconstruction_errors = (reconstructed_logits - logits).abs()
    maximum_error = float(reconstruction_errors.max().item())
    if maximum_error > tolerance:
        raise RuntimeError(
            "Tile evidence failed to reconstruct the bag logits: "
            f"maximum absolute error {maximum_error:.6g} exceeds {tolerance:.6g}."
        )
    return evidence, reconstruction_errors, class_baselines


def evidence_color_limit(
    evidence: np.ndarray,
    quantile: float = EVIDENCE_QUANTILE,
) -> float:
    """Calculate a robust symmetric color limit for signed tile evidence.

    Args:
        evidence (np.ndarray): One-dimensional signed tile evidence.
        quantile (float): Quantile of absolute evidence used as the limit.

    Returns:
        float: Positive symmetric limit for a zero-centered colormap.
    """
    if evidence.ndim != 1 or evidence.size == 0:
        raise ValueError("Evidence must be a nonempty one-dimensional array.")
    if not np.isfinite(evidence).all():
        raise ValueError("Evidence values must be finite.")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("Evidence quantile must be in the interval (0, 1].")
    limit = float(np.quantile(np.abs(evidence), quantile))
    return max(limit, NEAR_ZERO_EVIDENCE)


def top_positive_evidence_contribution(
    tissue_evidence: np.ndarray,
    bag_evidence_sum: float,
    top_k: int = TOP_EVIDENCE_TILE_COUNT,
) -> Tuple[float, Optional[float]]:
    """Summarize the strongest positive evidence in one displayed tissue.

    Args:
        tissue_evidence (np.ndarray): Signed evidence for one class and tissue.
        bag_evidence_sum (float): Sum of tile evidence over the complete bag.
        top_k (int): Maximum number of positive tiles included.

    Returns:
        Tuple[float, Optional[float]]: Top positive evidence sum and its signed
            percentage of the bag's tile-derived logit, or ``None`` when that
            denominator is effectively zero.
    """
    if tissue_evidence.ndim != 1 or tissue_evidence.size == 0:
        raise ValueError("Tissue evidence must be nonempty and one-dimensional.")
    if not np.isfinite(tissue_evidence).all() or not np.isfinite(bag_evidence_sum):
        raise ValueError("Evidence summary inputs must be finite.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    positive_evidence = tissue_evidence[tissue_evidence > 0.0]
    selected_count = min(top_k, positive_evidence.size)
    top_sum = (
        float(np.sort(positive_evidence)[-selected_count:].sum())
        if selected_count > 0
        else 0.0
    )
    percentage = (
        None
        if abs(bag_evidence_sum) <= NEAR_ZERO_EVIDENCE
        else 100.0 * top_sum / bag_evidence_sum
    )
    return top_sum, percentage


def tile_vertices(coordinates: np.ndarray, tile_size: int) -> np.ndarray:
    """Build vectorized square vertices for level-zero tile centers.

    Args:
        coordinates (np.ndarray): Tile centers shaped ``[N, 2]``.
        tile_size (int): Tile width and height in level-zero pixels.

    Returns:
        np.ndarray: Square vertices shaped ``[N, 4, 2]``.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Tile coordinates must have shape [N, 2].")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    half_tile = tile_size / 2.0
    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]
    return np.stack(
        (
            np.column_stack(
                (x_coordinates - half_tile, y_coordinates - half_tile)
            ),
            np.column_stack(
                (x_coordinates + half_tile, y_coordinates - half_tile)
            ),
            np.column_stack(
                (x_coordinates + half_tile, y_coordinates + half_tile)
            ),
            np.column_stack(
                (x_coordinates - half_tile, y_coordinates + half_tile)
            ),
        ),
        axis=1,
    )


def draw_attention(
    axis: Axes,
    coordinates: np.ndarray,
    attention: np.ndarray,
    tile_size: int,
) -> None:
    """Draw aligned attention rectangles on one spatial axis.

    Args:
        axis (Axes): Matplotlib axis receiving tile rectangles.
        coordinates (np.ndarray): Coordinates shaped ``[N, 2]``.
        attention (np.ndarray): Branch attention shaped ``[N]``.
        tile_size (int): Tile width and height in source-image pixels.

    Returns:
        None: Rectangles and spatial limits are applied in place.
    """
    if coordinates.shape != (attention.shape[0], 2):
        raise ValueError("Attention and coordinates are not exactly aligned.")
    normalized = normalize_attention(attention)
    half_tile = tile_size / 2.0
    collection = PolyCollection(
        tile_vertices(coordinates, tile_size),
        linewidths=0,
        edgecolors="none",
        facecolors=plt.cm.jet(normalized),
        closed=True,
    )
    axis.add_collection(collection)
    axis.set_xlim(float(coordinates[:, 0].min()) - half_tile, float(coordinates[:, 0].max()) + half_tile)
    axis.set_ylim(float(coordinates[:, 1].max()) + half_tile, float(coordinates[:, 1].min()) - half_tile)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def save_attention_figure(
    branch_attention: np.ndarray,
    coordinates: np.ndarray,
    branch_names: Sequence[str],
    predicted_index: int,
    slide_name: str,
    tissue_name: str,
    true_class: str,
    predicted_class: str,
    predicted_probability: float,
    image_path: Optional[Path],
    output_path: Path,
    tile_size: int,
    thumbnail_size: int,
    render_dpi: int = 150,
) -> None:
    """Save original tissue and canonical attention branch panels.

    Args:
        branch_attention (np.ndarray): Attention shaped ``[K, N]``.
        coordinates (np.ndarray): Aligned coordinates shaped ``[N, 2]``.
        branch_names (Sequence[str]): Display name for each attention branch.
        predicted_index (int): Predicted class index; ignored for shared SB attention.
        slide_name (str): Slide directory name.
        tissue_name (str): Tissue basename.
        true_class (str): Ground-truth class name.
        predicted_class (str): Predicted class name.
        predicted_probability (float): Probability of the predicted class.
        image_path (Optional[Path]): Optional tissue image for the original panel.
        output_path (Path): Destination PNG path.
        tile_size (int): Tile size in source-image pixels.
        thumbnail_size (int): Maximum thumbnail width and height.
        render_dpi (int): Positive output resolution in dots per inch.

    Returns:
        None: Figure is written to ``output_path``.
    """
    if branch_attention.ndim != 2 or branch_attention.shape[1] == 0:
        raise ValueError("branch_attention must have nonempty shape [K, N].")
    if len(branch_names) != branch_attention.shape[0]:
        raise ValueError("Branch labels do not match the attention branch count.")
    if coordinates.shape != (branch_attention.shape[1], 2):
        raise ValueError("Coordinates do not exactly match attention tile order.")
    if render_dpi <= 0:
        raise ValueError("render_dpi must be positive.")

    panel_count = 1 + branch_attention.shape[0]
    figure, axes = plt.subplots(1, panel_count, figsize=(4 * panel_count, 4))
    axes_array = np.atleast_1d(axes)
    thumbnail = load_tissue_thumbnail(image_path, thumbnail_size)
    if thumbnail is None:
        axes_array[0].text(
            0.5, 0.5, "Thumbnail\nNot Available", ha="center", va="center"
        )
    else:
        axes_array[0].imshow(thumbnail)
    axes_array[0].set_title("Original")
    axes_array[0].axis("off")

    is_multi_branch = branch_attention.shape[0] > 1
    for branch_index, branch_name in enumerate(branch_names):
        axis = axes_array[branch_index + 1]
        attention = branch_attention[branch_index]
        draw_attention(axis, coordinates, attention, tile_size)
        is_predicted = is_multi_branch and branch_index == predicted_index
        title = f"★ Predicted: {branch_name}" if is_predicted else branch_name
        axis.set_title(
            title,
            fontweight="bold" if is_predicted else "normal",
            bbox=(
                {"facecolor": "gold", "alpha": 0.35, "edgecolor": "darkorange"}
                if is_predicted
                else None
            ),
        )
        minimum = float(attention.min())
        maximum = float(attention.max())
        scalar_mappable = plt.cm.ScalarMappable(
            cmap=plt.cm.jet,
            norm=plt.Normalize(
                vmin=minimum,
                vmax=maximum if maximum > minimum else minimum + 1e-12,
            ),
        )
        scalar_mappable.set_array([])
        figure.colorbar(scalar_mappable, ax=axis, orientation="vertical", pad=0.04)

    figure.suptitle(
        f"Slide: {slide_name} | Tissue: {tissue_name}\n"
        f"True: {true_class} | Predicted: {predicted_class} "
        f"({predicted_probability:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=render_dpi, bbox_inches="tight")
    plt.close(figure)


def draw_evidence(
    axis: Axes,
    coordinates: np.ndarray,
    evidence: np.ndarray,
    tile_size: int,
    color_limit: float,
) -> None:
    """Draw aligned signed evidence rectangles on one spatial axis.

    Args:
        axis (Axes): Matplotlib axis receiving tile rectangles.
        coordinates (np.ndarray): Level-zero coordinates shaped ``[N, 2]``.
        evidence (np.ndarray): Signed evidence shaped ``[N]``.
        tile_size (int): Tile width and height in level-zero pixels.
        color_limit (float): Positive limit used for symmetric color mapping.

    Returns:
        None: Rectangles and spatial limits are applied in place.
    """
    if coordinates.shape != (evidence.shape[0], 2):
        raise ValueError("Evidence and coordinates are not exactly aligned.")
    if color_limit <= 0.0:
        raise ValueError("Evidence color limit must be positive.")
    normalization = plt.Normalize(vmin=-color_limit, vmax=color_limit)
    half_tile = tile_size / 2.0
    collection = PolyCollection(
        tile_vertices(coordinates, tile_size),
        linewidths=0,
        edgecolors="none",
        facecolors=plt.cm.RdBu_r(normalization(evidence)),
        closed=True,
    )
    axis.add_collection(collection)
    axis.set_xlim(
        float(coordinates[:, 0].min()) - half_tile,
        float(coordinates[:, 0].max()) + half_tile,
    )
    axis.set_ylim(
        float(coordinates[:, 1].max()) + half_tile,
        float(coordinates[:, 1].min()) - half_tile,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def save_evidence_figure(
    class_evidence: np.ndarray,
    coordinates: np.ndarray,
    class_names: Sequence[str],
    bag_evidence_sums: np.ndarray,
    predicted_index: int,
    slide_name: str,
    tissue_name: str,
    true_class: str,
    predicted_class: str,
    predicted_probability: float,
    image_path: Optional[Path],
    output_path: Path,
    tile_size: int,
    thumbnail_size: int,
    render_dpi: int = 150,
) -> None:
    """Save the original tissue and one signed-evidence panel per class.

    Args:
        class_evidence (np.ndarray): Signed evidence shaped ``[C, N]``.
        coordinates (np.ndarray): Aligned level-zero coordinates shaped ``[N, 2]``.
        class_names (Sequence[str]): Ordered class display names.
        bag_evidence_sums (np.ndarray): Full-bag tile evidence sums shaped ``[C]``.
        predicted_index (int): Predicted class index to highlight.
        slide_name (str): Slide directory name.
        tissue_name (str): Tissue basename.
        true_class (str): Ground-truth class name.
        predicted_class (str): Predicted class name.
        predicted_probability (float): Probability of the predicted class.
        image_path (Optional[Path]): Optional tissue image for the original panel.
        output_path (Path): Destination PNG path.
        tile_size (int): Tile size in level-zero source-image pixels.
        thumbnail_size (int): Maximum thumbnail width and height.
        render_dpi (int): Positive output resolution in dots per inch.

    Returns:
        None: Figure is written to ``output_path``.
    """
    if class_evidence.ndim != 2 or class_evidence.shape[1] == 0:
        raise ValueError("class_evidence must have nonempty shape [C, N].")
    if len(class_names) != class_evidence.shape[0]:
        raise ValueError("Class labels do not match the evidence class count.")
    if bag_evidence_sums.shape != (class_evidence.shape[0],):
        raise ValueError("Bag evidence sums do not match the evidence class count.")
    if coordinates.shape != (class_evidence.shape[1], 2):
        raise ValueError("Coordinates do not exactly match evidence tile order.")
    if not 0 <= predicted_index < class_evidence.shape[0]:
        raise ValueError("Predicted index is outside the evidence class range.")
    if render_dpi <= 0:
        raise ValueError("render_dpi must be positive.")

    panel_count = 1 + class_evidence.shape[0]
    figure, axes = plt.subplots(1, panel_count, figsize=(4 * panel_count, 4.5))
    axes_array = np.atleast_1d(axes)
    thumbnail = load_tissue_thumbnail(image_path, thumbnail_size)
    if thumbnail is None:
        axes_array[0].text(
            0.5, 0.5, "Thumbnail\nNot Available", ha="center", va="center"
        )
    else:
        axes_array[0].imshow(thumbnail)
    axes_array[0].set_title("Original")
    axes_array[0].axis("off")

    for class_index, class_name in enumerate(class_names):
        axis = axes_array[class_index + 1]
        evidence = class_evidence[class_index]
        color_limit = evidence_color_limit(evidence)
        draw_evidence(axis, coordinates, evidence, tile_size, color_limit)
        is_predicted = class_index == predicted_index
        title = f"★ Predicted: {class_name}" if is_predicted else class_name
        axis.set_title(
            title,
            fontweight="bold" if is_predicted else "normal",
            bbox=(
                {"facecolor": "gold", "alpha": 0.35, "edgecolor": "darkorange"}
                if is_predicted
                else None
            ),
        )
        _, percentage = top_positive_evidence_contribution(
            evidence, float(bag_evidence_sums[class_index])
        )
        annotation = (
            "Top 25 support: N/A\n(bag tile evidence ≈ 0)"
            if percentage is None
            else f"Top 25 support: {percentage:.1f}%\nof bag tile evidence"
        )
        axis.text(
            0.5,
            -0.08,
            annotation,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        scalar_mappable = plt.cm.ScalarMappable(
            cmap=plt.cm.RdBu_r,
            norm=plt.Normalize(vmin=-color_limit, vmax=color_limit),
        )
        scalar_mappable.set_array([])
        figure.colorbar(
            scalar_mappable, ax=axis, orientation="vertical", pad=0.04
        ).set_label("Signed logit evidence")

    figure.suptitle(
        f"Slide: {slide_name} | Tissue: {tissue_name}\n"
        f"True: {true_class} | Predicted: {predicted_class} "
        f"({predicted_probability:.3f}) | Red supports, blue opposes",
        fontsize=12,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=render_dpi, bbox_inches="tight")
    plt.close(figure)


def add_attention_colorbar(
    figure: plt.Figure,
    axis: Axes,
    attention: np.ndarray,
    label: Optional[str] = None,
) -> None:
    """Attach a jet colorbar matching one attention panel's value range.

    Args:
        figure (plt.Figure): Figure that owns the colorbar.
        axis (Axes): Axis whose value range the colorbar describes.
        attention (np.ndarray): One-dimensional attention weights.
        label (Optional[str]): Optional colorbar label.

    Returns:
        None: The colorbar is added to ``figure``.
    """
    minimum = float(attention.min())
    maximum = float(attention.max())
    scalar_mappable = plt.cm.ScalarMappable(
        cmap=plt.cm.jet,
        norm=plt.Normalize(
            vmin=minimum,
            vmax=maximum if maximum > minimum else minimum + 1e-12,
        ),
    )
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(
        scalar_mappable, ax=axis, orientation="vertical", pad=0.04
    )
    if label is not None:
        colorbar.set_label(label)


def save_attention_evidence_figure(
    branch_attention: np.ndarray,
    class_evidence: np.ndarray,
    coordinates: np.ndarray,
    branch_names: Sequence[str],
    class_names: Sequence[str],
    bag_evidence_sums: np.ndarray,
    predicted_index: int,
    slide_name: str,
    tissue_name: str,
    true_class: str,
    predicted_class: str,
    predicted_probability: float,
    image_path: Optional[Path],
    output_path: Path,
    tile_size: int,
    thumbnail_size: int,
    render_dpi: int = 150,
) -> None:
    """Save a 2-by-(C+1) figure with mean attention above the original tissue.

    The first column stacks the unweighted mean attention heatmap over the
    source thumbnail. Remaining columns keep per-class attention over signed
    evidence. A single shared attention branch still spans those class columns.

    Args:
        branch_attention (np.ndarray): Attention shaped ``[K, N]``.
        class_evidence (np.ndarray): Signed evidence shaped ``[C, N]``.
        coordinates (np.ndarray): Aligned level-zero coordinates shaped ``[N, 2]``.
        branch_names (Sequence[str]): Ordered attention branch display names.
        class_names (Sequence[str]): Ordered class display names.
        bag_evidence_sums (np.ndarray): Full-bag tile evidence sums shaped ``[C]``.
        predicted_index (int): Predicted class index highlighted once.
        slide_name (str): Slide directory name.
        tissue_name (str): Tissue basename.
        true_class (str): Ground-truth class name.
        predicted_class (str): Predicted class name.
        predicted_probability (float): Probability of the predicted class.
        image_path (Optional[Path]): Optional source tissue image.
        output_path (Path): Destination combined PNG path.
        tile_size (int): Tile size in level-zero source-image pixels.
        thumbnail_size (int): Maximum thumbnail width and height.
        render_dpi (int): Positive output resolution in dots per inch.

    Returns:
        None: Combined attention/evidence figure is written to ``output_path``.
    """
    class_count = len(class_names)
    if class_evidence.shape != (class_count, coordinates.shape[0]):
        raise ValueError("Evidence and coordinates must align for every class.")
    if branch_attention.ndim != 2 or branch_attention.shape[1] != coordinates.shape[0]:
        raise ValueError("Attention and coordinates must align for every branch.")
    if branch_attention.shape[0] not in {1, class_count}:
        raise ValueError("Attention must have one shared or one branch per class.")
    if len(branch_names) != branch_attention.shape[0]:
        raise ValueError("Attention branch names do not match branch count.")
    if bag_evidence_sums.shape != (class_count,):
        raise ValueError("Bag evidence sums do not match the class count.")
    if not 0 <= predicted_index < class_count:
        raise ValueError("Predicted index is outside the class range.")
    if render_dpi <= 0:
        raise ValueError("render_dpi must be positive.")

    figure = plt.figure(figsize=(5 * (class_count + 1), 9))
    grid = figure.add_gridspec(2, class_count + 1, hspace=0.32, wspace=0.28)
    mean_attention = branch_attention.mean(axis=0)
    mean_axis = figure.add_subplot(grid[0, 0])
    draw_attention(mean_axis, coordinates, mean_attention, tile_size)
    mean_axis.set_title("Attention — Mean")
    add_attention_colorbar(figure, mean_axis, mean_attention)

    original_axis = figure.add_subplot(grid[1, 0])
    thumbnail = load_tissue_thumbnail(image_path, thumbnail_size)
    if thumbnail is None:
        original_axis.text(
            0.5, 0.5, "Thumbnail\nNot Available", ha="center", va="center"
        )
    else:
        original_axis.imshow(thumbnail)
    original_axis.set_title("Original", fontweight="bold")
    original_axis.axis("off")

    if branch_attention.shape[0] == 1:
        attention_axes = [figure.add_subplot(grid[0, 1:])]
    else:
        attention_axes = [
            figure.add_subplot(grid[0, class_index + 1])
            for class_index in range(class_count)
        ]
    for branch_index, (axis, branch_name) in enumerate(
        zip(attention_axes, branch_names)
    ):
        attention = branch_attention[branch_index]
        draw_attention(axis, coordinates, attention, tile_size)
        axis.set_title(
            "Attention — Shared"
            if branch_attention.shape[0] == 1
            else f"Attention — {branch_name}"
        )
        add_attention_colorbar(
            figure,
            axis,
            attention,
            label="Attention weight"
            if branch_index == len(attention_axes) - 1
            else None,
        )

    for class_index, class_name in enumerate(class_names):
        axis = figure.add_subplot(grid[1, class_index + 1])
        evidence = class_evidence[class_index]
        color_limit = evidence_color_limit(evidence)
        draw_evidence(axis, coordinates, evidence, tile_size, color_limit)
        is_predicted = class_index == predicted_index
        axis.set_title(
            f"★ Evidence — {class_name}" if is_predicted else f"Evidence — {class_name}",
            fontweight="bold" if is_predicted else "normal",
            bbox=(
                {"facecolor": "gold", "alpha": 0.35, "edgecolor": "darkorange"}
                if is_predicted
                else None
            ),
        )
        _, percentage = top_positive_evidence_contribution(
            evidence, float(bag_evidence_sums[class_index])
        )
        annotation = (
            "Top 25 support: N/A\n(bag tile evidence ≈ 0)"
            if percentage is None
            else f"Top 25 support: {percentage:.1f}%\nof bag tile evidence"
        )
        axis.text(
            0.5,
            -0.08,
            annotation,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        scalar_mappable = plt.cm.ScalarMappable(
            cmap=plt.cm.RdBu_r,
            norm=plt.Normalize(vmin=-color_limit, vmax=color_limit),
        )
        scalar_mappable.set_array([])
        colorbar = figure.colorbar(
            scalar_mappable, ax=axis, orientation="vertical", pad=0.04
        )
        if class_index == class_count - 1:
            colorbar.set_label(
                "Signed logit evidence\n★ Predicted class highlighted"
            )

    figure.suptitle(
        f"Slide: {slide_name} | Tissue: {tissue_name}\n"
        f"True: {true_class} | Predicted: {predicted_class} "
        f"({predicted_probability:.3f}) | Evidence: red supports, blue opposes",
        fontsize=12,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=render_dpi, bbox_inches="tight")
    plt.close(figure)


def safe_filename(value: str) -> str:
    """Convert metadata to a safe filename component.

    Args:
        value (str): Raw slide or tissue identifier.

    Returns:
        str: Identifier containing only conservative filename characters.
    """
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )


def validate_aligned_bag(
    attention: torch.Tensor,
    coordinates: torch.Tensor,
    tissue_indices: torch.Tensor,
    tissue_names: Sequence[str],
) -> None:
    """Validate exact post-collation alignment for one unpadded bag.

    Args:
        attention (torch.Tensor): Attention weights shaped ``[K, N]``.
        coordinates (torch.Tensor): Coordinates shaped ``[N, 2]``.
        tissue_indices (torch.Tensor): Tissue provenance shaped ``[N]``.
        tissue_names (Sequence[str]): Tissue names indexed by provenance values.

    Returns:
        None: Validation succeeds by returning normally.
    """
    tile_count = attention.shape[1]
    if attention.ndim != 2:
        raise ValueError("Canonical attention must have shape [K, N].")
    if coordinates.shape != (tile_count, 2):
        raise ValueError("Collated coordinates are not aligned with attention.")
    if tissue_indices.shape != (tile_count,):
        raise ValueError("Collated tissue indices are not aligned with attention.")
    if not torch.isfinite(attention).all() or not torch.isfinite(coordinates).all():
        raise ValueError("Valid attention and coordinates must be finite.")
    if tile_count == 0:
        raise ValueError("Cannot visualize an empty bag.")
    if bool(((tissue_indices < 0) | (tissue_indices >= len(tissue_names))).any()):
        raise ValueError("A valid tile has an invalid tissue provenance index.")


def evaluate_with_attention(
    model: CLAMModel,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
    data_root: str,
    output_dir: str,
    bag_level: str,
    tile_size: int = 448,
    thumbnail_size: int = 512,
    render_dpi: int = 150,
    render_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Render exactly aligned attention and evidence for each bag and tissue.

    Args:
        model (CLAMModel): Loaded canonical CLAM-SB or CLAM-MB model.
        dataloader (DataLoader): Loader returning the unified collated bag contract.
        device (torch.device): Inference device.
        class_names (Sequence[str]): Ordered checkpoint class names.
        data_root (str): Root containing class and slide directories.
        output_dir (str): Directory receiving figures.
        bag_level (str): Checkpoint bag level, tissue or slide.
        tile_size (int): Tile size in source-image pixels.
        thumbnail_size (int): Maximum thumbnail width and height.
        render_dpi (int): Positive output resolution in dots per inch.
        render_workers (int): Number of independent figure-rendering processes.

    Returns:
        List[Dict[str, Any]]: One attention/evidence summary per rendered tissue.
    """
    if render_workers <= 0:
        raise ValueError("render_workers must be positive.")
    results: List[Dict[str, Any]] = []
    render_futures: List[Future[None]] = []
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    model.eval()
    render_executor = (
        ProcessPoolExecutor(
            max_workers=render_workers,
            mp_context=mp.get_context("spawn"),
        )
        if render_workers > 1
        else None
    )
    try:
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc=f"Preparing {bag_level} bags"):
                features = batch["features"].to(device)
                masks = batch["masks"].to(device)
                outputs = model(features, mask=masks, instance_eval=False)
                attention_weights = outputs["attention_weights"]
                if (
                    not isinstance(attention_weights, torch.Tensor)
                    or attention_weights.ndim != 3
                ):
                    raise ValueError(
                        "Canonical model attention_weights must be a tensor "
                        "shaped [B, K, N]."
                    )
                logits = outputs["logits"]
                if not isinstance(logits, torch.Tensor):
                    raise TypeError("Canonical model logits must be a tensor.")
                (
                    evidence_weights,
                    reconstruction_errors,
                    class_baselines,
                ) = compute_tile_evidence(
                    model=model,
                    features=features,
                    masks=masks,
                    attention_weights=attention_weights,
                    logits=logits,
                )
                probabilities = outputs["probabilities"].detach().cpu()
                predictions = outputs["predictions"].detach().cpu()
                labels = batch["labels"].detach().cpu()
                class_baselines = class_baselines.detach().cpu()

                for bag_index, slide_name in enumerate(batch["slide_names"]):
                    valid_mask = batch["masks"][bag_index]
                    bag_tile_count = int(valid_mask.sum().item())
                    attention = attention_weights[
                        bag_index, :, valid_mask.to(device)
                    ].detach().cpu()
                    evidence = evidence_weights[
                        bag_index, :, valid_mask.to(device)
                    ].detach().cpu()
                    coordinates = batch["coordinates"][
                        bag_index, valid_mask
                    ].detach().cpu()
                    tissue_indices = batch["tissue_indices"][
                        bag_index, valid_mask
                    ].detach().cpu()
                    tissue_names = [
                        str(name)
                        for name in batch["bag_tissue_names"][bag_index]
                    ]
                    validate_aligned_bag(
                        attention, coordinates, tissue_indices, tissue_names
                    )
                    if evidence.shape != (len(class_names), attention.shape[1]):
                        raise ValueError(
                            "Class evidence is not aligned with the unpadded bag."
                        )
                    bag_evidence_sums = evidence.sum(dim=-1)

                    true_index = int(labels[bag_index].item())
                    predicted_index = int(predictions[bag_index].item())
                    true_class = class_names[true_index]
                    predicted_class = class_names[predicted_index]
                    predicted_probability = float(
                        probabilities[bag_index, predicted_index].item()
                    )
                    branch_names = (
                        list(class_names)
                        if isinstance(model, CLAM_MB)
                        else ["Shared attention"]
                    )
                    expected_branches = len(branch_names)
                    if attention.shape[0] != expected_branches:
                        raise ValueError(
                            f"Expected {expected_branches} attention branches, "
                            f"received {attention.shape[0]}."
                        )

                    for tissue_index, tissue_name in enumerate(tissue_names):
                        tissue_mask = tissue_indices == tissue_index
                        if not bool(tissue_mask.any()):
                            continue
                        tissue_attention = attention[:, tissue_mask].numpy()
                        tissue_evidence = evidence[:, tissue_mask].numpy()
                        tissue_coordinates = coordinates[tissue_mask].numpy()
                        if (
                            tissue_attention.shape[1]
                            != tissue_coordinates.shape[0]
                        ):
                            raise RuntimeError(
                                "Internal tissue alignment invariant failed."
                            )
                        if (
                            tissue_evidence.shape[1]
                            != tissue_coordinates.shape[0]
                        ):
                            raise RuntimeError(
                                "Internal tissue evidence alignment invariant "
                                "failed."
                            )
                        image_path = find_tissue_image(
                            data_root, true_class, str(slide_name), tissue_name
                        )
                        output_path = output_root / (
                            f"{safe_filename(str(slide_name))}_"
                            f"{safe_filename(tissue_name)}_attention.png"
                        )
                        render_arguments = {
                            "branch_attention": tissue_attention,
                            "class_evidence": tissue_evidence,
                            "coordinates": tissue_coordinates,
                            "branch_names": branch_names,
                            "class_names": class_names,
                            "bag_evidence_sums": bag_evidence_sums.numpy(),
                            "predicted_index": predicted_index,
                            "slide_name": str(slide_name),
                            "tissue_name": tissue_name,
                            "true_class": true_class,
                            "predicted_class": predicted_class,
                            "predicted_probability": predicted_probability,
                            "image_path": image_path,
                            "output_path": output_path,
                            "tile_size": tile_size,
                            "thumbnail_size": thumbnail_size,
                            "render_dpi": render_dpi,
                        }
                        if render_executor is None:
                            save_attention_evidence_figure(**render_arguments)
                        else:
                            render_futures.append(
                                render_executor.submit(
                                    save_attention_evidence_figure,
                                    **render_arguments,
                                )
                            )

                        evidence_diagnostics: Dict[
                            str, Dict[str, Optional[float]]
                        ] = {}
                        for class_index, class_name in enumerate(class_names):
                            top_sum, top_percentage = (
                                top_positive_evidence_contribution(
                                    tissue_evidence[class_index],
                                    float(
                                        bag_evidence_sums[class_index].item()
                                    ),
                                )
                            )
                            bag_sum = float(
                                bag_evidence_sums[class_index].item()
                            )
                            evidence_diagnostics[str(class_name)] = {
                                "tissue_evidence_sum": float(
                                    tissue_evidence[class_index].sum()
                                ),
                                "bag_tile_evidence_sum": bag_sum,
                                "reconstructed_logit": (
                                    bag_sum
                                    + float(
                                        class_baselines[class_index].item()
                                    )
                                ),
                                "logit_reconstruction_error": float(
                                    reconstruction_errors[
                                        bag_index, class_index
                                    ].item()
                                ),
                                "top_25_positive_evidence_sum": top_sum,
                                "top_25_contribution_percentage": top_percentage,
                            }
                        primary_index = (
                            predicted_index
                            if isinstance(model, CLAM_MB)
                            else 0
                        )
                        primary_attention = tissue_attention[primary_index]
                        primary_gates = primary_attention * bag_tile_count
                        results.append(
                            {
                                "bag_level": bag_level,
                                "slide_name": str(slide_name),
                                "tissue_name": tissue_name,
                                "true_class": true_class,
                                "predicted_class": predicted_class,
                                "predicted_probability": predicted_probability,
                                "primary_attention_branch": branch_names[
                                    primary_index
                                ],
                                "num_tiles": int(primary_attention.size),
                                "bag_num_tiles": bag_tile_count,
                                "attention_mass": float(
                                    primary_attention.sum()
                                ),
                                "max_attention": float(
                                    primary_attention.max()
                                ),
                                "mean_attention": float(
                                    primary_attention.mean()
                                ),
                                "max_gate": float(primary_gates.max()),
                                "mean_gate": float(primary_gates.mean()),
                                "heatmap_path": str(output_path),
                                "evidence_heatmap_path": str(output_path),
                                "evidence_by_class": evidence_diagnostics,
                            }
                        )
        for render_future in tqdm(
            as_completed(render_futures),
            total=len(render_futures),
            desc="Writing heatmaps",
            disable=not render_futures,
        ):
            render_future.result()
    finally:
        if render_executor is not None:
            render_executor.shutdown(wait=True, cancel_futures=True)
    return results


def main() -> None:
    """Run checkpoint-driven canonical CLAM attention visualization.

    Args:
        None: Configuration and overrides are read from the command line.

    Returns:
        None: Heatmaps and a JSON summary are written to disk.
    """
    args = parse_args()
    launcher_config = load_config(args.config)
    checkpoint_path = args.checkpoint or launcher_config["paths"]["checkpoint"]
    if args.checkpoint is None and not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train CLAM first or pass --checkpoint."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_config, class_folders, bag_level = load_checkpoint_model(
        str(checkpoint_path), device
    )
    visualization = checkpoint_config.get("visualization", {}) or {}
    if not isinstance(visualization, Mapping):
        raise TypeError("Checkpoint visualization config must be a mapping.")
    split = args.split or str(visualization.get("split", "val"))
    max_bags_value = (
        args.max_slides
        if args.max_slides is not None
        else visualization.get("max_slides")
    )
    max_bags = int(max_bags_value) if max_bags_value is not None else None
    render_dpi = (
        args.dpi
        if args.dpi is not None
        else int(visualization.get("dpi", 150))
    )
    if render_dpi <= 0:
        raise ValueError("Visualization DPI must be positive.")
    render_workers = (
        args.render_workers
        if args.render_workers is not None
        else int(visualization.get("render_workers", 4))
    )
    if render_workers <= 0:
        raise ValueError("Visualization render_workers must be positive.")
    output_dir = (
        args.output_dir
        or launcher_config["paths"]["attention_output"]
        or checkpoint_config.get("paths", {}).get("attention_output")
    )
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Attention output: {output_dir}")
    dataset = create_visualization_dataset(
        checkpoint_config,
        split,
        class_folders,
        bag_level,
        max_bags,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(checkpoint_config.get("batch_size", 1)),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(checkpoint_config.get("num_workers", 0)),
    )
    print(f"Using device: {device}")
    print(f"Render workers: {render_workers} | DPI: {render_dpi}")
    print(f"Model: {checkpoint_config['model_type']} | bag level: {bag_level}")
    print(f"Split: {split} | bags: {len(dataset)} | classes: {class_folders}")
    results = evaluate_with_attention(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=class_folders,
        data_root=str(checkpoint_config["data_root"]),
        output_dir=str(output_dir),
        bag_level=bag_level,
        tile_size=int(visualization.get("tile_size", 448)),
        thumbnail_size=int(visualization.get("thumbnail_size", 512)),
        render_dpi=render_dpi,
        render_workers=render_workers,
    )
    summary_path = Path(output_dir) / f"attention_summary_{split}.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(results, summary_file, indent=2)
    print(
        f"Rendered {len(results)} paired attention/evidence tissue heatmaps "
        f"from {len(dataset)} bags."
    )
    print(f"Attention and evidence summary saved to {summary_path}")


if __name__ == "__main__":
    main()
