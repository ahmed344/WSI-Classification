"""Reproduce PANTHER prototypical-assignment maps and mixture plots.

This is an independent port of the official PANTHER visualization notebook at
Mahmood Lab PANTHER commit e7ffae402e146363fe1ca3813ffe68d248ec570f.
Tiles are colored by argmax of the slide GMM posterior. Mixture probabilities
use a purple- and pink-free categorical palette rather than the notebook's
original colors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

try:
    import openslide
except ImportError:  # pragma: no cover - optional fallback for non-TIFF WSIs.
    openslide = None

try:
    import tifffile
except ImportError:  # pragma: no cover - OpenSlide remains available as fallback.
    tifffile = None

from .config_loader import load_config, resolve_evaluation_run
from .panther_dataset import (
    PantherSlideDataset,
    build_datasets,
    load_split_manifest,
    select_tile_indices,
    stable_seed,
)
from .panther_model import LinearClassifier
from .prototype import load_prototypes
from .train_panther import (
    MODEL_SCHEMA,
    create_panther,
    plot_training_history,
    resolve_device,
    seed_everything,
)


OFFICIAL_PANTHER_COMMIT = "e7ffae402e146363fe1ca3813ffe68d248ec570f"
OFFICIAL_NOTEBOOK = (
    "https://github.com/mahmoodlab/PANTHER/blob/"
    + OFFICIAL_PANTHER_COMMIT
    + "/src/visualization/prototypical_assignment_map_visualization.ipynb"
)
DEFAULT_HEX_COLORS = (
    "#4A4A4A", "#2E7D32", "#8D6E63", "#00897B",
    "#1565C0", "#EF6C00", "#6D4C41", "#43A047",
    "#0277BD", "#C62828", "#F9A825", "#558B2F",
    "#00838F", "#F57C00", "#1B5E20", "#FDD835",
    "#01579B", "#A1887F", "#7CB342", "#00ACC1",
    "#E53935", "#FFB300", "#33691E", "#26A69A",
    "#1E88E5", "#5D4037", "#D84315", "#9E9D24",
    "#81C784", "#FF7043", "#90A4AE", "#BCAAA4",
)
IMAGE_SUFFIXES = (
    ".ome.tiff", ".ome.tif", ".tiff", ".tif", ".svs", ".png", ".jpg", ".jpeg"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument(
        "--slide-key", action="append", default=None,
        help="Exact class/slide key to render; may be repeated.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def get_default_color_map(num_prototypes: int) -> Dict[int, tuple[int, int, int]]:
    """Return a 32-color categorical palette without purple or pink hues.

    Args:
        num_prototypes (int): Number of GMM components to color, in ``[1, 32]``.

    Returns:
        Dict[int, tuple[int, int, int]]: Prototype index to RGB triple.
    """
    if not 0 < num_prototypes <= len(DEFAULT_HEX_COLORS):
        raise ValueError(
            f"num_prototypes must be in [1, {len(DEFAULT_HEX_COLORS)}]."
        )
    return {
        index: tuple(
            int(DEFAULT_HEX_COLORS[index][offset : offset + 2], 16)
            for offset in (1, 3, 5)
        )
        for index in range(num_prototypes)
    }


def load_coordinates(path: Path) -> np.ndarray:
    """Load level-zero tile-center coordinates without changing row order."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"x", "y"}.issubset(reader.fieldnames):
            raise ValueError(f"Coordinate CSV must contain x and y columns: {path}")
        coordinates = [(float(row["x"]), float(row["y"])) for row in reader]
    result = np.asarray(coordinates, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != (2,) or result.shape[0] == 0:
        raise ValueError(f"Coordinate CSV must contain at least one row: {path}")
    if not np.isfinite(result).all():
        raise ValueError(f"Coordinate CSV contains non-finite values: {path}")
    return result


def find_tissue_image(
    data_root: Path, class_name: str, slide_name: str, tissue_name: str
) -> Path:
    """Locate the source image paired with one tissue feature/coordinate pair."""
    slide_dir = data_root / class_name / slide_name
    for suffix in IMAGE_SUFFIXES:
        candidate = slide_dir / f"{tissue_name}{suffix}"
        if candidate.is_file():
            return candidate
    tissue_lower = tissue_name.lower()
    matches = sorted(
        path for path in slide_dir.iterdir()
        if path.is_file()
        and path.name.lower().startswith(tissue_lower)
        and any(path.name.lower().endswith(suffix) for suffix in IMAGE_SUFFIXES)
    )
    if not matches:
        raise FileNotFoundError(
            f"No source image found for {class_name}/{slide_name}/{tissue_name}."
        )
    return min(matches, key=lambda path: (len(path.name), path.name))


def _to_rgb_uint8(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    while result.ndim > 3 and result.shape[0] == 1:
        result = result[0]
    if result.ndim == 2:
        result = np.repeat(result[..., None], 3, axis=2)
    elif result.ndim == 3 and result.shape[-1] not in (3, 4):
        if result.shape[0] in (3, 4):
            result = np.moveaxis(result, 0, -1)
        else:
            raise ValueError(f"Unsupported image array shape {result.shape}.")
    if result.ndim != 3 or result.shape[-1] not in (3, 4):
        raise ValueError(f"Unsupported image array shape {result.shape}.")
    result = result[..., :3]
    if result.dtype == np.uint8:
        return np.ascontiguousarray(result)
    if np.issubdtype(result.dtype, np.integer):
        maximum = float(np.iinfo(result.dtype).max)
        result = np.rint(result.astype(np.float32) * (255.0 / maximum))
    else:
        result = result.astype(np.float32)
        if float(np.nanmax(result)) <= 1.0:
            result *= 255.0
    return np.ascontiguousarray(np.clip(result, 0, 255).astype(np.uint8))


def _load_tiff_preview(
    path: Path, downsample: int
) -> tuple[np.ndarray, tuple[int, int]]:
    if tifffile is None:
        raise RuntimeError("tifffile is not installed.")
    with tifffile.TiffFile(path) as handle:
        levels = handle.series[0].levels
        shapes = [level.shape for level in levels]
        first = shapes[0]
        if first[-1] in (3, 4):
            height, width = int(first[-3]), int(first[-2])
        else:
            height, width = int(first[-2]), int(first[-1])
        target_width = max(1, int(width / downsample))
        target_height = max(1, int(height / downsample))

        candidates = []
        for index, shape in enumerate(shapes):
            if shape[-1] in (3, 4):
                level_height, level_width = int(shape[-3]), int(shape[-2])
            else:
                level_height, level_width = int(shape[-2]), int(shape[-1])
            if level_width >= target_width and level_height >= target_height:
                candidates.append((level_width * level_height, index))
        level_index = min(candidates)[1] if candidates else len(levels) - 1
        array = levels[level_index].asarray()

    preview = Image.fromarray(_to_rgb_uint8(array))
    if preview.size != (target_width, target_height):
        preview = preview.resize(
            (target_width, target_height), resample=Image.Resampling.BICUBIC
        )
    return np.asarray(preview), (width, height)


def _load_openslide_preview(
    path: Path, downsample: int
) -> tuple[np.ndarray, tuple[int, int]]:
    if openslide is None:
        raise RuntimeError("OpenSlide is not installed.")
    with openslide.OpenSlide(str(path)) as slide:
        width, height = slide.dimensions
        target = (max(1, int(width / downsample)), max(1, int(height / downsample)))
        preview = slide.get_thumbnail(target).convert("RGB")
        if preview.size != target:
            preview = preview.resize(target, resample=Image.Resampling.BICUBIC)
    return np.asarray(preview), (width, height)


def load_image_preview(
    path: Path, downsample: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load a low-memory RGB preview and its original level-zero dimensions."""
    lower_name = path.name.lower()
    if tifffile is not None and lower_name.endswith((".tif", ".tiff")):
        return _load_tiff_preview(path, downsample)
    if openslide is not None:
        try:
            return _load_openslide_preview(path, downsample)
        except Exception:
            pass
    with Image.open(path) as image:
        dimensions = image.size
        target = (
            max(1, int(dimensions[0] / downsample)),
            max(1, int(dimensions[1] / downsample)),
        )
        image.thumbnail(target, resample=Image.Resampling.BICUBIC)
        preview = image.convert("RGB").resize(
            target, resample=Image.Resampling.BICUBIC
        )
    return np.asarray(preview), dimensions


def blend_categorical_overlay(
    preview: np.ndarray,
    coordinates: np.ndarray,
    labels: np.ndarray,
    original_dimensions: tuple[int, int],
    tile_size: int,
    alpha: float,
    color_map: Mapping[int, tuple[int, int, int]],
) -> Image.Image:
    """Blend hard prototype assignments over a tissue preview.

    The official PANTHER utility receives top-left patch coordinates. HE-MYO CSV
    files store patch centers, so half a tile is subtracted before applying the
    same colored-block blend without a per-tile border.

    Args:
        preview (np.ndarray): RGB preview with shape ``[height, width, 3]``.
        coordinates (np.ndarray): Level-zero tile-center coordinates ``[N, 2]``.
        labels (np.ndarray): Prototype index for each coordinate, length ``N``.
        original_dimensions (tuple[int, int]): Level-zero ``(width, height)``.
        tile_size (int): Tile side length in level-zero pixels.
        alpha (float): Overlay opacity in ``[0, 1]``.
        color_map (Mapping[int, tuple[int, int, int]]): Prototype RGB colors.

    Returns:
        Image.Image: RGB overlay with the same spatial size as ``preview``.
    """
    if coordinates.shape != (labels.shape[0], 2):
        raise ValueError("Coordinates and labels must be exactly aligned.")
    if preview.ndim != 3 or preview.shape[2] != 3:
        raise ValueError("preview must have shape [height, width, 3].")
    if tile_size <= 0 or not 0.0 <= alpha <= 1.0:
        raise ValueError("tile_size must be positive and alpha must be in [0, 1].")
    unknown = set(int(value) for value in np.unique(labels)) - set(color_map)
    if unknown:
        raise ValueError(f"Assignments reference missing prototype colors: {unknown}")

    canvas = np.asarray(preview, dtype=np.uint8).copy()
    preview_height, preview_width = canvas.shape[:2]
    original_width, original_height = original_dimensions
    scale_x = preview_width / original_width
    scale_y = preview_height / original_height
    half_tile = tile_size / 2.0

    for coordinate, raw_label in zip(coordinates, labels):
        left = max(0, int(math.floor((coordinate[0] - half_tile) * scale_x)))
        top = max(0, int(math.floor((coordinate[1] - half_tile) * scale_y)))
        right = min(
            preview_width, int(math.ceil((coordinate[0] + half_tile) * scale_x))
        )
        bottom = min(
            preview_height, int(math.ceil((coordinate[1] + half_tile) * scale_y))
        )
        if right <= left or bottom <= top:
            continue
        image_block = canvas[top:bottom, left:right].copy()
        color_block = np.empty_like(image_block)
        color_block[...] = color_map[int(raw_label)]
        canvas[top:bottom, left:right] = cv2.addWeighted(
            color_block, alpha, image_block, 1.0 - alpha, 0
        )
    return Image.fromarray(canvas)


def _mixture_axis(
    axis: plt.Axes,
    mixtures: np.ndarray,
    color_map: Mapping[int, tuple[int, int, int]],
) -> None:
    """Draw the official-style GMM mixture-proportion bar chart on one axis.

    Args:
        axis (plt.Axes): Destination matplotlib axis.
        mixtures (np.ndarray): Nonnegative mixture weights of length ``C``.
        color_map (Mapping[int, tuple[int, int, int]]): Prototype RGB colors.

    Returns:
        None: The axis is mutated in place.
    """
    colors = [np.asarray(color_map[index]) / 255.0 for index in range(len(mixtures))]
    indices = np.arange(len(mixtures))
    axis.bar(indices, mixtures, color=colors, width=0.8)
    axis.set_xlabel("Cluster")
    axis.set_ylabel("Proportion / Mixture")
    axis.set_xticks(indices)
    axis.set_xticklabels([f"c{index}" for index in indices], rotation=90)
    upper = max(0.55, math.ceil(float(mixtures.max()) * 10.0) / 10.0 + 0.05)
    axis.set_ylim(0.0, min(1.05, upper))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_mixture_plot(
    mixtures: np.ndarray,
    color_map: Mapping[int, tuple[int, int, int]],
    output_path: Path,
    dpi: int,
) -> None:
    """Save the official-style GMM mixture-proportion bar chart.

    Args:
        mixtures (np.ndarray): Nonnegative mixture weights of length ``C``.
        color_map (Mapping[int, tuple[int, int, int]]): Prototype RGB colors.
        output_path (Path): Destination PNG path.
        dpi (int): Positive output resolution in dots per inch.

    Returns:
        None: The figure is written to disk.
    """
    figure, axis = plt.subplots(figsize=(6, 3))
    _mixture_axis(axis, mixtures, color_map)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_slide_overview(
    tissues: Sequence[tuple[str, Image.Image, Image.Image]],
    slide_name: str,
    true_class: str,
    predicted_class: str,
    class_probabilities: Mapping[str, float],
    split: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Save original tissue previews above their prototypical assignment maps.

    Args:
        tissues (Sequence[tuple[str, Image.Image, Image.Image]]): Tissue name,
            original preview, and assignment overlay for each tissue.
        slide_name (str): Slide directory name.
        true_class (str): Ground-truth class name.
        predicted_class (str): Predicted class name.
        class_probabilities (Mapping[str, float]): Softmax probability per class.
        split (str): Dataset split name.
        output_path (Path): Destination PNG path.
        dpi (int): Positive output resolution in dots per inch.

    Returns:
        None: The figure is written to disk.
    """
    if not tissues:
        raise ValueError("Cannot save an assignment overview with no tissues.")
    if predicted_class not in class_probabilities:
        raise ValueError(
            f"Predicted class '{predicted_class}' is missing from class_probabilities."
        )
    columns = len(tissues)
    figure, axes = plt.subplots(
        2, columns, figsize=(4.5 * columns, 7.2), squeeze=False
    )
    for column, (tissue_name, original, overlay) in enumerate(tissues):
        axes[0, column].imshow(original)
        axes[0, column].set_title(tissue_name, fontsize=8)
        axes[0, column].axis("off")
        axes[1, column].imshow(overlay)
        axes[1, column].set_title("")
        axes[1, column].axis("off")
    probability_text = " | ".join(
        f"{name} {class_probabilities[name]:.3f}" for name in class_probabilities
    )
    predicted_probability = float(class_probabilities[predicted_class])
    figure.suptitle(
        f"True: {true_class} | Predicted: {predicted_class} "
        f"(p={predicted_probability:.3f})\n"
        f"{probability_text}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def assignment_output_paths(
    output_root: Path,
    split: str,
    class_name: str,
    slide_name: str,
    tissue_names: Sequence[str],
    save_mixture: bool,
    save_individual_tissues: bool,
) -> Dict[str, Any]:
    """Build flat PNG paths under one split directory.

    Args:
        output_root (Path): Visualization output root.
        split (str): Dataset split name, for example ``test``.
        class_name (str): Ground-truth class name.
        slide_name (str): Slide directory name.
        tissue_names (Sequence[str]): Ordered tissue names on the slide.
        save_mixture (bool): Whether an independent mixture PNG is requested.
        save_individual_tissues (bool): Whether per-tissue assignment PNGs are
            requested.

    Returns:
        Dict[str, Any]: Split directory, overview path, optional mixture path,
        and a tissue-name to assignment-PNG mapping.
    """
    split_dir = output_root / split
    stem = f"{_safe_slug(class_name)}_{_safe_slug(slide_name)}"
    return {
        "split_dir": split_dir,
        "assignment_overview": split_dir / f"{stem}_assignment_overview.png",
        "mixture_plot": (
            split_dir / f"{stem}_mixture_proportions.png" if save_mixture else None
        ),
        "tissues": {
            name: split_dir / f"{stem}_{_safe_slug(name)}_assignment.png"
            for name in tissue_names
        }
        if save_individual_tissues
        else {},
    }


def _safe_slug(value: str) -> str:
    """Convert a class, slide, or tissue name into a unique filename stem.

    Args:
        value (str): Raw identifier.

    Returns:
        str: Conservative slug with an 8-character content hash suffix.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:90] or 'slide'}_{digest}"


def _split_assignments_by_tissue(
    record: Mapping[str, Any],
    hard_assignments: np.ndarray,
    config: Mapping[str, Any],
) -> list[tuple[Mapping[str, str], np.ndarray, np.ndarray]]:
    coordinate_arrays = [
        load_coordinates(Path(tissue["tiles_path"])) for tissue in record["tissues"]
    ]
    raw_total = sum(len(coordinates) for coordinates in coordinate_arrays)
    selected = select_tile_indices(
        raw_total,
        config.get("max_tiles_per_slide"),
        str(config["tile_sampling"]),
        stable_seed(int(config["random_seed"]), str(record["slide_key"])),
    ).numpy()
    if selected.shape != hard_assignments.shape:
        raise ValueError(
            f"Selected coordinate count {len(selected)} does not match assignment "
            f"count {len(hard_assignments)} for {record['slide_key']}."
        )

    result = []
    offset = 0
    for tissue, coordinates in zip(record["tissues"], coordinate_arrays):
        local_mask = (selected >= offset) & (selected < offset + len(coordinates))
        positions = np.flatnonzero(local_mask)
        local_indices = selected[positions] - offset
        result.append((tissue, coordinates[local_indices], hard_assignments[positions]))
        offset += len(coordinates)
    return result


@torch.inference_mode()
def _visualize_slide(
    dataset: PantherSlideDataset,
    index: int,
    panther: torch.nn.Module,
    classifier: LinearClassifier,
    class_names: Sequence[str],
    device: torch.device,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    output_root: Path,
    split: str,
    color_map: Mapping[int, tuple[int, int, int]],
) -> Dict[str, Any]:
    """Render one slide's original/assignment overview and optional extras.

    Args:
        dataset (PantherSlideDataset): Split dataset containing the slide.
        index (int): Dataset index of the slide to render.
        panther (torch.nn.Module): Frozen MAP-EM encoder.
        classifier (LinearClassifier): Trained downstream linear head.
        class_names (Sequence[str]): Ordered checkpoint class names.
        device (torch.device): Inference device.
        config (Mapping[str, Any]): Checkpoint training configuration.
        settings (Mapping[str, Any]): Visualization settings.
        output_root (Path): Visualization output root.
        split (str): Dataset split name.
        color_map (Mapping[int, tuple[int, int, int]]): Prototype RGB colors.

    Returns:
        Dict[str, Any]: Per-slide manifest payload including output paths.
    """
    item = dataset[index]
    record = dataset.records[index]
    encoded = panther(item["features"].to(device), return_assignments=True)
    logits = classifier(encoded["representation"])
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    predicted_index = int(probabilities.argmax())
    true_class = str(item["class_name"])
    predicted_class = str(class_names[predicted_index])
    predicted_probability = float(probabilities[predicted_index])
    class_probabilities = {
        str(name): float(probabilities[class_index])
        for class_index, name in enumerate(class_names)
    }
    responsibilities = encoded["assignments"][0].cpu().numpy()
    mixtures = encoded["mixture_weights"][0].cpu().numpy()
    hard_assignments = responsibilities.argmax(axis=1).astype(np.int64)
    tissue_assignments = _split_assignments_by_tissue(record, hard_assignments, config)
    tissue_names = [str(tissue["tissue_name"]) for tissue, _, _ in tissue_assignments]
    paths = assignment_output_paths(
        output_root,
        split,
        true_class,
        str(item["slide_name"]),
        tissue_names,
        bool(settings["save_mixture_proportions"]),
        bool(settings["save_individual_tissues"]),
    )
    paths["split_dir"].mkdir(parents=True, exist_ok=True)

    tissues: list[tuple[str, Image.Image, Image.Image]] = []
    tissue_payload = []
    individual_paths = paths["tissues"]
    for tissue, coordinates, labels in tissue_assignments:
        tissue_name = str(tissue["tissue_name"])
        image_path = find_tissue_image(
            Path(str(config["data_root"])), str(item["class_name"]),
            str(item["slide_name"]), tissue_name,
        )
        preview, dimensions = load_image_preview(
            image_path, int(settings["downsample"])
        )
        original = Image.fromarray(preview)
        overlay = blend_categorical_overlay(
            preview, coordinates, labels, dimensions, int(settings["tile_size"]),
            float(settings["alpha"]), color_map,
        )
        tissues.append((tissue_name, original, overlay))
        tissue_output: Optional[Path] = individual_paths.get(tissue_name)
        if tissue_output is not None:
            overlay.save(tissue_output)
        tissue_payload.append(
            {
                "tissue_name": tissue_name,
                "source_image": str(image_path),
                "coordinates": str(tissue["tiles_path"]),
                "rendered_tiles": int(len(labels)),
                "assignment_map": str(tissue_output) if tissue_output else None,
            }
        )

    mixture_path = paths["mixture_plot"]
    overview_path = paths["assignment_overview"]
    if mixture_path is not None:
        save_mixture_plot(mixtures, color_map, mixture_path, int(settings["dpi"]))
    save_slide_overview(
        tissues,
        str(item["slide_name"]),
        true_class,
        predicted_class,
        class_probabilities,
        split,
        overview_path,
        int(settings["dpi"]),
    )
    hard_counts = np.bincount(
        hard_assignments, minlength=len(color_map)
    ).astype(np.int64)
    entropy = -np.sum(
        responsibilities
        * np.log(np.clip(responsibilities, np.finfo(np.float32).tiny, 1.0)),
        axis=1,
    )
    return {
        "slide_key": str(item["slide_key"]),
        "slide_name": str(item["slide_name"]),
        "class_name": true_class,
        "true_class": true_class,
        "label": int(item["label"]),
        "predicted_class": predicted_class,
        "predicted_label": predicted_index,
        "predicted_probability": predicted_probability,
        "class_probabilities": class_probabilities,
        "split": split,
        "num_tiles": int(item["num_tiles"]),
        "num_tissues": int(item["num_tissues"]),
        "mixture_weights": [float(value) for value in mixtures],
        "dominant_mixture": int(mixtures.argmax()),
        "dominant_mixture_weight": float(mixtures.max()),
        "hard_assignment_counts": hard_counts.tolist(),
        "dominant_hard_assignment": int(hard_counts.argmax()),
        "dominant_hard_assignment_fraction": float(
            hard_counts.max() / hard_counts.sum()
        ),
        "mean_normalized_posterior_entropy": float(
            entropy.mean() / math.log(len(color_map))
        ),
        "mixture_plot": str(mixture_path) if mixture_path is not None else None,
        "assignment_overview": str(overview_path),
        "tissues": tissue_payload,
    }


def _select_indices(
    dataset: PantherSlideDataset,
    slide_keys: Optional[Sequence[str]],
    maximum: Optional[int],
) -> list[int]:
    indices = list(range(len(dataset)))
    if slide_keys:
        requested = list(dict.fromkeys(slide_keys))
        lookup = {
            str(record["slide_key"]): index
            for index, record in enumerate(dataset.records)
        }
        missing = [key for key in requested if key not in lookup]
        if missing:
            raise ValueError(
                "Requested slide keys are absent from the selected split: "
                + ", ".join(missing)
            )
        indices = [lookup[key] for key in requested]
    if maximum is not None:
        if maximum <= 0:
            raise ValueError("max_slides must be positive.")
        indices = indices[:maximum]
    if not indices:
        raise ValueError("No slides were selected for visualization.")
    return indices


def _plot_run_training_history(run_dir: Path, checkpoint: Mapping[str, Any]) -> Optional[Path]:
    """Write ``training_history.png`` from a run's saved epoch JSON if present.

    Args:
        run_dir (Path): Dated training-run directory.
        checkpoint (Mapping[str, Any]): Loaded checkpoint payload.

    Returns:
        Optional[Path]: Written PNG path, or ``None`` when history is absent.
    """
    history_path = run_dir / "training_history.json"
    if not history_path.is_file():
        return None
    with history_path.open(encoding="utf-8") as handle:
        history = json.load(handle)
    if not isinstance(history, list) or not history:
        return None
    details = checkpoint.get("training_details", {})
    best_epoch = int(details.get("best_validation_epoch", -1)) if isinstance(details, Mapping) else -1
    plot_path = run_dir / "training_history.png"
    plot_training_history(history, plot_path, best_epoch)
    return plot_path


def run_visualization(
    config_path: Optional[str] = None,
    split_override: Optional[str] = None,
    max_slides_override: Optional[int] = None,
    slide_keys: Optional[Sequence[str]] = None,
    output_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Render original-style PANTHER visualizations for a checkpoint split.

    Args:
        config_path (Optional[str]): YAML path, or ``None`` for the module default.
        split_override (Optional[str]): Optional split replacing ``visualization.split``.
        max_slides_override (Optional[int]): Optional slide cap.
        slide_keys (Optional[Sequence[str]]): Optional exact slide keys to render.
        output_override (Optional[str]): Optional visualization output directory.

    Returns:
        Dict[str, Any]: Visualization manifest payload written to disk.
    """
    runtime_config = load_config(config_path)
    run_dir = resolve_evaluation_run(runtime_config)
    checkpoint_path = Path(runtime_config["paths"]["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_schema") != MODEL_SCHEMA:
        raise ValueError(
            f"Checkpoint schema must be '{MODEL_SCHEMA}', received "
            f"'{checkpoint.get('model_schema')}'."
        )
    checkpoint_config = checkpoint.get("config")
    class_names = checkpoint.get("class_folders")
    if not isinstance(checkpoint_config, Mapping) or not isinstance(class_names, list):
        raise ValueError("Checkpoint is missing its exact config or ordered classes.")

    saved_classes, assignments = load_split_manifest(run_dir / "split_manifest.json")
    if saved_classes != class_names:
        raise ValueError("Checkpoint classes disagree with the split manifest.")
    datasets, discovered_classes, _ = build_datasets(
        checkpoint_config, class_folders=class_names, split_assignments=assignments,
    )
    if discovered_classes != class_names:
        raise ValueError("Current dataset class order disagrees with the checkpoint.")

    settings = dict(runtime_config["visualization"])
    split = split_override or str(settings["split"])
    maximum = (
        max_slides_override if max_slides_override is not None
        else settings.get("max_slides")
    )
    settings["max_slides"] = maximum
    output_root = (
        Path(output_override).expanduser().resolve()
        if output_override is not None
        else Path(runtime_config["paths"]["visualization_output"])
    )
    indices = _select_indices(datasets[split], slide_keys, maximum)

    seed_everything(int(checkpoint_config["random_seed"]))
    device = resolve_device(runtime_config)
    prototypes, prototype_metadata = load_prototypes(run_dir / "prototypes.pkl")
    panther = create_panther(prototypes, checkpoint_config).eval().to(device)
    input_dim = int(checkpoint["training_details"]["input_dim"])
    classifier = LinearClassifier(
        input_dim,
        len(class_names),
        bias=bool(checkpoint_config["training"]["classifier_bias"]),
    ).eval().to(device)
    classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
    color_map = get_default_color_map(panther.num_prototypes)
    history_plot = _plot_run_training_history(run_dir, checkpoint)

    output_root.mkdir(parents=True, exist_ok=True)
    slide_payload = []
    global_counts = np.zeros(panther.num_prototypes, dtype=np.int64)
    print(
        f"Visualizing {len(indices)} PANTHER {split} slides on {device} "
        f"from run {run_dir.name}."
    )
    for position, index in enumerate(indices, start=1):
        payload = _visualize_slide(
            datasets[split], index, panther, classifier, class_names, device,
            checkpoint_config, settings, output_root, split, color_map,
        )
        slide_payload.append(payload)
        global_counts += np.asarray(payload["hard_assignment_counts"], dtype=np.int64)
        print(
            f"PANTHER visualization {position}/{len(indices)}: "
            f"{payload['slide_key']} ({payload['num_tiles']:,} tiles)"
        )

    hard_dominance = np.asarray(
        [slide["dominant_hard_assignment_fraction"] for slide in slide_payload]
    )
    mixture_dominance = np.asarray(
        [slide["dominant_mixture_weight"] for slide in slide_payload]
    )
    manifest: Dict[str, Any] = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "prototypes": str(run_dir / "prototypes.pkl"),
        "prototype_metadata": prototype_metadata,
        "official_reference_commit": OFFICIAL_PANTHER_COMMIT,
        "official_visualization_notebook": OFFICIAL_NOTEBOOK,
        "method": {
            "assignment": "argmax of PANTHER GMM posterior qq",
            "mixture_plot": "fitted slide GMM mixture weights pi",
            "palette": list(DEFAULT_HEX_COLORS[: panther.num_prototypes]),
            "coordinates": "level-zero tile centers",
            "adaptation": (
                "one slide-level GMM over concatenated tissues; originals and "
                "assignments rendered side by side under the split directory"
            ),
            "settings": settings,
        },
        "split": split,
        "training_history_plot": str(history_plot) if history_plot is not None else None,
        "num_slides": len(slide_payload),
        "num_tissues": int(sum(slide["num_tissues"] for slide in slide_payload)),
        "num_tiles": int(global_counts.sum()),
        "global_hard_assignment_counts": global_counts.tolist(),
        "globally_unused_prototypes": np.flatnonzero(global_counts == 0).tolist(),
        "slides_with_single_hard_prototype": int(np.sum(hard_dominance == 1.0)),
        "slides_with_at_least_90_percent_one_hard_prototype": int(
            np.sum(hard_dominance >= 0.9)
        ),
        "mean_dominant_hard_assignment_fraction": float(hard_dominance.mean()),
        "mean_dominant_mixture_weight": float(mixture_dominance.mean()),
        "slides": slide_payload,
    }
    manifest_path = output_root / "visualization_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote visualization manifest: {manifest_path}")
    return manifest


def main() -> None:
    """Run assignment visualization from the command line.

    Args:
        None: Configuration is read from command-line arguments.

    Returns:
        None: Figures and a JSON manifest are written to disk.
    """
    arguments = parse_args()
    run_visualization(
        arguments.config,
        split_override=arguments.split,
        max_slides_override=arguments.max_slides,
        slide_keys=arguments.slide_key,
        output_override=arguments.output_dir,
    )


if __name__ == "__main__":
    main()
