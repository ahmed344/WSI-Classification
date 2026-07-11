"""Generate aligned tissue heatmaps from canonical CLAM attention branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
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
CHECKPOINT_SCHEMA = "canonical_clam_v1"


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
    for (x_coord, y_coord), weight in zip(coordinates, normalized):
        axis.add_patch(
            patches.Rectangle(
                (float(x_coord) - half_tile, float(y_coord) - half_tile),
                tile_size,
                tile_size,
                linewidth=0,
                facecolor=plt.cm.jet(float(weight)),
            )
        )
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

    Returns:
        None: Figure is written to ``output_path``.
    """
    if branch_attention.ndim != 2 or branch_attention.shape[1] == 0:
        raise ValueError("branch_attention must have nonempty shape [K, N].")
    if len(branch_names) != branch_attention.shape[0]:
        raise ValueError("Branch labels do not match the attention branch count.")
    if coordinates.shape != (branch_attention.shape[1], 2):
        raise ValueError("Coordinates do not exactly match attention tile order.")

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
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    tqdm.write(f"Saved attention heatmap to {output_path}")


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
) -> List[Dict[str, Any]]:
    """Render exactly aligned canonical attention for each bag and tissue.

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

    Returns:
        List[Dict[str, Any]]: One summary record per rendered tissue.
    """
    results: List[Dict[str, Any]] = []
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Visualizing {bag_level} bags"):
            features = batch["features"].to(device)
            masks = batch["masks"].to(device)
            outputs = model(features, mask=masks, instance_eval=False)
            attention_weights = outputs["attention_weights"]
            if not isinstance(attention_weights, torch.Tensor) or attention_weights.ndim != 3:
                raise ValueError(
                    "Canonical model attention_weights must be a tensor shaped [B, K, N]."
                )
            probabilities = outputs["probabilities"].detach().cpu()
            predictions = outputs["predictions"].detach().cpu()
            labels = batch["labels"].detach().cpu()

            for bag_index, slide_name in enumerate(batch["slide_names"]):
                valid_mask = batch["masks"][bag_index]
                attention = attention_weights[bag_index, :, valid_mask.to(device)].detach().cpu()
                coordinates = batch["coordinates"][bag_index, valid_mask].detach().cpu()
                tissue_indices = batch["tissue_indices"][bag_index, valid_mask].detach().cpu()
                tissue_names = [str(name) for name in batch["bag_tissue_names"][bag_index]]
                validate_aligned_bag(
                    attention, coordinates, tissue_indices, tissue_names
                )

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
                    tissue_coordinates = coordinates[tissue_mask].numpy()
                    if tissue_attention.shape[1] != tissue_coordinates.shape[0]:
                        raise RuntimeError("Internal tissue alignment invariant failed.")
                    image_path = find_tissue_image(
                        data_root, true_class, str(slide_name), tissue_name
                    )
                    output_path = output_root / (
                        f"{safe_filename(str(slide_name))}_"
                        f"{safe_filename(tissue_name)}_attention.png"
                    )
                    save_attention_figure(
                        branch_attention=tissue_attention,
                        coordinates=tissue_coordinates,
                        branch_names=branch_names,
                        predicted_index=predicted_index,
                        slide_name=str(slide_name),
                        tissue_name=tissue_name,
                        true_class=true_class,
                        predicted_class=predicted_class,
                        predicted_probability=predicted_probability,
                        image_path=image_path,
                        output_path=output_path,
                        tile_size=tile_size,
                        thumbnail_size=thumbnail_size,
                    )
                    primary_index = predicted_index if isinstance(model, CLAM_MB) else 0
                    primary_attention = tissue_attention[primary_index]
                    results.append(
                        {
                            "bag_level": bag_level,
                            "slide_name": str(slide_name),
                            "tissue_name": tissue_name,
                            "true_class": true_class,
                            "predicted_class": predicted_class,
                            "predicted_probability": predicted_probability,
                            "primary_attention_branch": branch_names[primary_index],
                            "num_tiles": int(primary_attention.size),
                            "attention_mass": float(primary_attention.sum()),
                            "max_attention": float(primary_attention.max()),
                            "mean_attention": float(primary_attention.mean()),
                            "heatmap_path": str(output_path),
                        }
                    )
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
    output_dir = (
        args.output_dir
        or checkpoint_config.get("paths", {}).get("attention_output")
        or launcher_config["paths"]["attention_output"]
    )
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
    )
    summary_path = Path(output_dir) / f"attention_summary_{split}.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(results, summary_file, indent=2)
    print(f"Rendered {len(results)} tissue heatmaps from {len(dataset)} bags.")
    print(f"Attention summary saved to {summary_path}")


if __name__ == "__main__":
    main()
