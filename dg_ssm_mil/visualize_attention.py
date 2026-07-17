"""
Attention heatmap generation for tissue-level DG-SSM-MIL models.
"""
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import openslide
except ImportError:  # pragma: no cover - environment dependent.
    openslide = None

try:
    from .config_loader import (
        get_coordinate_columns,
        load_config,
        resolve_device,
        resolve_feature_file_suffix,
    )
    from .dataset import DGSSMMILTissueDataset, collate_fn
    from .evaluate import load_trained_model
except ImportError:
    from config_loader import (  # type: ignore
        get_coordinate_columns,
        load_config,
        resolve_device,
        resolve_feature_file_suffix,
    )
    from dataset import DGSSMMILTissueDataset, collate_fn  # type: ignore
    from evaluate import load_trained_model  # type: ignore


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for DG attention visualization.

    Args:
        None: Arguments are read from the command line.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate tissue-level DG-SSM-MIL attention heatmaps."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yml. Defaults to dg_ssm_mil/config.yml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to paths.checkpoint from config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional attention output directory.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Dataset split to visualize. Defaults to visualization.split.",
    )
    parser.add_argument(
        "--max-tissues",
        type=int,
        default=None,
        help="Maximum number of tissues to visualize.",
    )
    return parser.parse_args()


def create_dataset_for_split(
    config: Dict[str, Any],
    split: str,
    class_folders: Optional[List[str]] = None,
) -> DGSSMMILTissueDataset:
    """
    Create a tissue-level DG dataset for one split.

    Args:
        config (Dict[str, Any]): Parsed DG-SSM-MIL configuration.
        split (str): Dataset split, either `train` or `val`.
        class_folders (Optional[List[str]]): Optional class folder order from
            the checkpoint.

    Returns:
        DGSSMMILTissueDataset: Dataset for the requested split.
    """
    return DGSSMMILTissueDataset(
        data_root=str(config["data_root"]),
        class_folders=class_folders,
        split=split,
        train_ratio=float(config["train_ratio"]),
        val_ratio=float(config.get("val_ratio", 0.1)),
        test_ratio=float(config.get("test_ratio", 0.1)),
        random_seed=int(config["random_seed"]),
        feature_file_suffix=resolve_feature_file_suffix(config),
        coordinate_columns=get_coordinate_columns(config),
        coordinate_mismatch=str(config.get("coordinate_mismatch", "error")),
        sort_tiles_spatially=bool(config.get("sort_tiles_spatially", False)),
        normalize_coordinates=bool(config.get("normalize_coordinates", False)),
        max_tiles_per_tissue=config.get(f"max_tiles_per_tissue_{split}"),
        expected_feature_dim=int(config["input_dim"]),
        bag_level=str(config.get("bag_level", "tissue")),
        tile_sampling=str(config.get("tile_sampling", "random")),
        feature_normalization=str(config.get("feature_normalization", "none")),
    )


def load_raw_coordinates(
    tiles_path: str,
    coordinate_columns: Tuple[str, str],
    sort_tiles_spatially: bool,
) -> np.ndarray:
    """
    Load raw tile coordinates for plotting.

    Args:
        tiles_path (str): Path to a tile coordinate CSV file.
        coordinate_columns (Tuple[str, str]): Names of x and y coordinate columns.
        sort_tiles_spatially (bool): Whether to apply the same row-major sorting
            used by the dataset.

    Returns:
        np.ndarray: Raw coordinate array with shape `[n_tiles, 2]`.
    """
    tiles_df = pd.read_csv(tiles_path)
    coords = tiles_df.loc[:, list(coordinate_columns)].to_numpy(dtype=np.float32)
    if sort_tiles_spatially and len(coords) > 0:
        order = np.lexsort((coords[:, 0], coords[:, 1]))
        coords = coords[order]
    return coords


def load_tissue_thumbnail(
    feature_path: str,
    tissue_name: str,
    thumbnail_size: int,
) -> Optional[np.ndarray]:
    """
    Load a tissue thumbnail from the slide directory when available.

    Args:
        feature_path (str): Path to the feature file in the tissue slide directory.
        tissue_name (str): Tissue name used to infer the `.ome.tiff` filename.
        thumbnail_size (int): Width and height of the thumbnail request.

    Returns:
        Optional[np.ndarray]: RGB thumbnail array, or None if unavailable.
    """
    if openslide is None:
        return None
    slide_dir = os.path.dirname(feature_path)
    tissue_slide_path = os.path.join(slide_dir, f"{tissue_name}.ome.tiff")
    if not os.path.exists(tissue_slide_path):
        return None
    try:
        slide = openslide.OpenSlide(tissue_slide_path)
        thumbnail = slide.get_thumbnail(size=(thumbnail_size, thumbnail_size))
        slide.close()
        return np.asarray(thumbnail)
    except Exception as exc:
        tqdm.write(f"Warning: Could not load thumbnail {tissue_slide_path}: {exc}")
        return None


def normalize_attention(attention: np.ndarray) -> np.ndarray:
    """
    Normalize attention weights to the `[0, 1]` range for color mapping.

    Args:
        attention (np.ndarray): Attention weights with shape `[n_tiles]`.

    Returns:
        np.ndarray: Normalized attention weights with shape `[n_tiles]`.
    """
    if len(attention) <= 1:
        return np.zeros_like(attention, dtype=np.float32)
    order = np.argsort(attention, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(attention), dtype=np.float32)
    return ranks / float(len(attention) - 1)


def select_top_attention_indices(
    attention: np.ndarray,
    top_k_tiles: Optional[int],
    top_percentile: Optional[float],
) -> np.ndarray:
    """
    Select high-attention tile indices for highlighting and export.

    Args:
        attention (np.ndarray): Attention weights with shape `[n_tiles]`.
        top_k_tiles (Optional[int]): Number of highest-attention tiles to select.
        top_percentile (Optional[float]): Percentile threshold for selection.

    Returns:
        np.ndarray: Sorted tile indices selected by attention strength.
    """
    if len(attention) == 0:
        return np.asarray([], dtype=np.int64)
    selected_masks: List[np.ndarray] = []
    if top_percentile is not None:
        threshold = np.percentile(attention, float(top_percentile))
        selected_masks.append(np.where(attention >= threshold)[0])
    if top_k_tiles is not None and int(top_k_tiles) > 0:
        top_k = min(int(top_k_tiles), len(attention))
        selected_masks.append(np.argsort(attention)[-top_k:])
    if not selected_masks:
        return np.asarray([], dtype=np.int64)
    selected = np.unique(np.concatenate(selected_masks))
    return selected[np.argsort(attention[selected])[::-1]]


def draw_attention_rectangles(
    axis: plt.Axes,
    tile_coords: np.ndarray,
    attention: np.ndarray,
    tile_size: int,
    selected_indices: Optional[np.ndarray] = None,
) -> None:
    """
    Draw tile rectangles colored by attention weight.

    Args:
        axis (plt.Axes): Matplotlib axis where rectangles are drawn.
        tile_coords (np.ndarray): Raw coordinates with shape `[n_tiles, 2]`.
        attention (np.ndarray): Attention weights with shape `[n_tiles]`.
        tile_size (int): Pixel size of each square tile.
        selected_indices (Optional[np.ndarray]): Optional subset of indices to draw.

    Returns:
        None: Rectangles are added to `axis`.
    """
    attention_norm = normalize_attention(attention)
    indices = selected_indices if selected_indices is not None else np.arange(len(attention))
    for tile_idx in indices:
        x_coord, y_coord = tile_coords[tile_idx]
        rect = patches.Rectangle(
            (x_coord - tile_size // 2, y_coord - tile_size // 2),
            tile_size,
            tile_size,
            linewidth=0,
            edgecolor="none",
            facecolor=plt.cm.jet(attention_norm[tile_idx]),
        )
        axis.add_patch(rect)


def configure_spatial_axis(
    axis: plt.Axes,
    tile_coords: np.ndarray,
    tile_size: int,
) -> None:
    """
    Apply coordinate limits and styling to a heatmap axis.

    Args:
        axis (plt.Axes): Matplotlib axis to configure.
        tile_coords (np.ndarray): Raw coordinates with shape `[n_tiles, 2]`.
        tile_size (int): Pixel size of each square tile.

    Returns:
        None: Axis styling is applied in-place.
    """
    x_min = float(tile_coords[:, 0].min()) - tile_size // 2
    x_max = float(tile_coords[:, 0].max()) + tile_size // 2
    y_min = float(tile_coords[:, 1].min()) - tile_size // 2
    y_max = float(tile_coords[:, 1].max()) + tile_size // 2
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_max, y_min)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def visualize_attention_heatmap(
    attention: np.ndarray,
    tile_coords: np.ndarray,
    top_indices: np.ndarray,
    output_path: str,
    slide_name: str,
    tissue_name: str,
    true_class: str,
    predicted_class: str,
    predicted_probability: float,
    feature_path: str,
    tile_size: int,
    thumbnail_size: int,
) -> None:
    """
    Save a CLAM-style DG attention heatmap figure.

    Args:
        attention (np.ndarray): Attention weights with shape `[n_tiles]`.
        tile_coords (np.ndarray): Raw coordinates with shape `[n_tiles, 2]`.
        top_indices (np.ndarray): High-attention tile indices used only for
            summary export; the figure renders a single full attention heatmap.
        output_path (str): Destination PNG path.
        slide_name (str): Slide directory name.
        tissue_name (str): Tissue name.
        true_class (str): Ground-truth class name.
        predicted_class (str): Predicted class name.
        predicted_probability (float): Probability assigned to `predicted_class`.
        feature_path (str): Feature file path used to locate thumbnail image.
        tile_size (int): Pixel size of each tile rectangle.
        thumbnail_size (int): Thumbnail size for the original image panel.

    Returns:
        None: The figure is saved to `output_path`.
    """
    if len(tile_coords) == 0 or len(attention) == 0:
        raise ValueError("Cannot visualize attention without coordinates and weights.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    thumbnail = load_tissue_thumbnail(feature_path, tissue_name, thumbnail_size)
    if thumbnail is not None:
        axes[0].imshow(thumbnail)
        axes[0].axis("off")
    else:
        axes[0].text(
            0.5,
            0.5,
            "Thumbnail\nNot Available",
            ha="center",
            va="center",
            fontsize=10,
        )
        axes[0].axis("off")

    draw_attention_rectangles(axes[1], tile_coords, attention, tile_size)
    configure_spatial_axis(axes[1], tile_coords, tile_size)
    scalar_mappable = plt.cm.ScalarMappable(
        cmap=plt.cm.jet,
        norm=plt.Normalize(vmin=0.0, vmax=100.0),
    )
    scalar_mappable.set_array([])
    plt.colorbar(scalar_mappable, ax=axes[1], orientation="vertical", pad=0.05)
    axes[1].set_title("Attention percentile")

    fig.suptitle(
        f"Slide: {slide_name} | Tissue: {tissue_name}\n"
        f"True: {true_class} | Predicted: {predicted_class} "
        f"({predicted_probability:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    tqdm.write(f"Saved attention heatmap to {output_path}")


def safe_filename(value: str) -> str:
    """
    Convert an identifier into a filesystem-safe filename component.

    Args:
        value (str): Raw identifier.

    Returns:
        str: Filename-safe identifier.
    """
    return value.replace("/", "_").replace(" ", "_")


def build_top_tile_rows(
    attention: np.ndarray,
    tile_coords: np.ndarray,
    top_indices: np.ndarray,
    slide_name: str,
    tissue_name: str,
) -> List[Dict[str, Any]]:
    """
    Build summary rows for high-attention tiles.

    Args:
        attention (np.ndarray): Attention weights with shape `[n_tiles]`.
        tile_coords (np.ndarray): Raw coordinates with shape `[n_tiles, 2]`.
        top_indices (np.ndarray): Selected high-attention tile indices.
        slide_name (str): Slide directory name.
        tissue_name (str): Tissue name.

    Returns:
        List[Dict[str, Any]]: Per-tile attention summary rows.
    """
    rows: List[Dict[str, Any]] = []
    for rank_idx, tile_idx in enumerate(top_indices, start=1):
        rows.append(
            {
                "slide_name": slide_name,
                "tissue_name": tissue_name,
                "rank": rank_idx,
                "tile_index": int(tile_idx),
                "x": float(tile_coords[tile_idx, 0]),
                "y": float(tile_coords[tile_idx, 1]),
                "attention": float(attention[tile_idx]),
            }
        )
    return rows


def align_attention_and_coordinates(
    attention: np.ndarray,
    tile_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align attention and coordinate arrays by common tile count.

    Args:
        attention (np.ndarray): Attention weights with shape `[n_attention]`.
        tile_coords (np.ndarray): Coordinates with shape `[n_coords, 2]`.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Length-aligned attention and coordinates.
    """
    if len(attention) == len(tile_coords):
        return attention, tile_coords
    n_common = min(len(attention), len(tile_coords))
    return attention[:n_common], tile_coords[:n_common]


def generate_attention_heatmaps(
    config: Dict[str, Any],
    checkpoint_path: str,
    output_dir: str,
    split: str,
    max_tissues: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Generate DG attention heatmaps for one tissue-level split.

    Args:
        config (Dict[str, Any]): Parsed DG-SSM-MIL configuration.
        checkpoint_path (str): Path to a trained checkpoint.
        output_dir (str): Directory where attention artifacts are written.
        split (str): Dataset split to visualize.
        max_tissues (Optional[int]): Optional maximum number of tissues to process.

    Returns:
        List[Dict[str, Any]]: Per-tissue attention summary rows.
    """
    device = torch.device(resolve_device(config))
    model, checkpoint = load_trained_model(checkpoint_path, device)
    class_folders = checkpoint.get("class_folders")
    if not class_folders:
        raise KeyError("Checkpoint is missing frozen 'class_folders'.")
    checkpoint_config = checkpoint["config"]
    dataset = create_dataset_for_split(
        checkpoint_config, split, class_folders=class_folders
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(config.get("num_workers", 0)),
    )
    class_names = dataset.class_folders
    coordinate_columns = get_coordinate_columns(checkpoint_config)
    vis_config = config.get("visualization", {}) or {}
    tile_size = int(vis_config.get("tile_size", 448))
    thumbnail_size = int(vis_config.get("thumbnail_size", 512))
    top_k_tiles = vis_config.get("top_k_tiles", 25)
    top_percentile = vis_config.get("top_percentile")
    save_top_tiles_csv = bool(vis_config.get("save_top_tiles_csv", True))

    os.makedirs(output_dir, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    top_tile_rows: List[Dict[str, Any]] = []
    processed_count = 0

    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Visualizing {split} attention")
        for batch in progress_bar:
            if max_tissues is not None and processed_count >= max_tissues:
                break
            features = batch["features"].to(device)
            coords = batch["coords"].to(device)
            masks = batch["masks"].to(device)
            labels = batch["labels"].to(device)
            tissue_indices = batch.get("tissue_indices")
            if tissue_indices is not None:
                tissue_indices = tissue_indices.to(device)
            outputs = model(
                features,
                coords,
                masks,
                tissue_indices=tissue_indices,
            )
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=1)
            predicted_label = int(torch.argmax(logits, dim=1).cpu().item())
            true_label = int(labels.cpu().item())
            attention = outputs["attention_weights"][0].detach().cpu().numpy()
            valid_mask = masks[0].detach().cpu().numpy()
            attention = attention[valid_mask]
            selected_tile_indices = (
                batch["tile_indices"][0][batch["masks"][0]].detach().cpu().numpy()
            )

            slide_name = batch["slide_names"][0]
            tissue_name = batch["tissue_names"][0]
            feature_path = batch["feature_paths"][0]
            tiles_path = batch["tiles_paths"][0]
            tile_coords = load_raw_coordinates(
                tiles_path,
                coordinate_columns,
                False,
            )
            if np.any(selected_tile_indices < 0) or np.any(
                selected_tile_indices >= len(tile_coords)
            ):
                raise ValueError(
                    f"Invalid retained tile indices for {tiles_path}: "
                    f"CSV has {len(tile_coords)} rows."
                )
            tile_coords = tile_coords[selected_tile_indices]
            if len(attention) != len(tile_coords):
                raise ValueError("Attention and retained coordinate counts differ.")
            top_indices = select_top_attention_indices(
                attention,
                int(top_k_tiles) if top_k_tiles is not None else None,
                float(top_percentile) if top_percentile is not None else None,
            )

            slide_safe = safe_filename(slide_name)
            tissue_safe = safe_filename(tissue_name)
            output_path = os.path.join(
                output_dir,
                f"{slide_safe}_{tissue_safe}_dg_attention.png",
            )
            predicted_probability = float(probabilities[0, predicted_label].cpu().item())
            visualize_attention_heatmap(
                attention=attention,
                tile_coords=tile_coords,
                top_indices=top_indices,
                output_path=output_path,
                slide_name=slide_name,
                tissue_name=tissue_name,
                true_class=class_names[true_label],
                predicted_class=class_names[predicted_label],
                predicted_probability=predicted_probability,
                feature_path=feature_path,
                tile_size=tile_size,
                thumbnail_size=thumbnail_size,
            )
            tile_rows = build_top_tile_rows(
                attention,
                tile_coords,
                top_indices,
                slide_name,
                tissue_name,
            )
            top_tile_rows.extend(tile_rows)
            attention_entropy = float(-np.sum(attention * np.log(attention + 1e-8)))
            summary_rows.append(
                {
                    "slide_name": slide_name,
                    "tissue_name": tissue_name,
                    "true_class": class_names[true_label],
                    "predicted_class": class_names[predicted_label],
                    "predicted_probability": predicted_probability,
                    "max_attention": float(np.max(attention)),
                    "mean_attention": float(np.mean(attention)),
                    "attention_entropy": attention_entropy,
                    "num_tiles": int(len(attention)),
                    "heatmap_path": output_path,
                    "top_tiles": tile_rows,
                }
            )
            processed_count += 1

    summary_path = os.path.join(output_dir, f"attention_summary_{split}.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary_rows, summary_file, indent=2)
    if save_top_tiles_csv and top_tile_rows:
        top_tiles_path = os.path.join(output_dir, f"top_attention_tiles_{split}.csv")
        pd.DataFrame(top_tile_rows).to_csv(top_tiles_path, index=False)
        print(f"Top attention tile CSV saved to {top_tiles_path}")
    print(f"Attention summary saved to {summary_path}")
    return summary_rows


def main() -> None:
    """
    Run DG-SSM-MIL tissue-level attention heatmap generation.

    Args:
        None: Configuration and overrides are read from CLI arguments.

    Returns:
        None: Attention heatmaps and summaries are written to disk.
    """
    args = parse_args()
    config = load_config(args.config)
    vis_config = config.get("visualization", {}) or {}
    split = args.split or str(vis_config.get("split", "val"))
    max_tissues = (
        args.max_tissues
        if args.max_tissues is not None
        else vis_config.get("max_tissues")
    )
    if max_tissues is not None:
        max_tissues = int(max_tissues)
    checkpoint_path = args.checkpoint or config["paths"]["checkpoint"]
    if args.checkpoint is None and not os.path.exists(checkpoint_path):
        parent, filename = os.path.split(checkpoint_path)
        checkpoint_path = os.path.join(parent, "repeat_00", filename)
    output_dir = args.output_dir or config["paths"]["attention_output"]

    print(f"Using feature model '{config['feature_model']}'")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Writing attention heatmaps to: {output_dir}")
    generate_attention_heatmaps(
        config=config,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        split=split,
        max_tissues=max_tissues,
    )


if __name__ == "__main__":
    main()
