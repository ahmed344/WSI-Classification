"""Reproduce PANTHER prototypical-assignment maps and mixture plots.

This is an independent port of the official PANTHER visualization notebook at
Mahmood Lab PANTHER commit e7ffae402e146363fe1ca3813ffe68d248ec570f.
The original notebook colors each tile by argmax of the slide GMM posterior and
plots the fitted mixture probabilities with a fixed categorical palette.
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
from PIL import Image, ImageOps

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
from .prototype import load_prototypes
from .train_panther import MODEL_SCHEMA, create_panther, resolve_device, seed_everything


OFFICIAL_PANTHER_COMMIT = "e7ffae402e146363fe1ca3813ffe68d248ec570f"
OFFICIAL_NOTEBOOK = (
    "https://github.com/mahmoodlab/PANTHER/blob/"
    + OFFICIAL_PANTHER_COMMIT
    + "/src/visualization/prototypical_assignment_map_visualization.ipynb"
)
DEFAULT_HEX_COLORS = (
    "#696969", "#556b2f", "#a0522d", "#483d8b",
    "#008000", "#008b8b", "#000080", "#7f007f",
    "#8fbc8f", "#b03060", "#ff0000", "#ffa500",
    "#00ff00", "#8a2be2", "#00ff7f", "#FFFF54",
    "#00ffff", "#00bfff", "#f4a460", "#adff2f",
    "#da70d6", "#b0c4de", "#ff00ff", "#1e90ff",
    "#f0e68c", "#0000ff", "#dc143c", "#90ee90",
    "#ff1493", "#7b68ee", "#ffefd5", "#ffb6c1",
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
    """Return the exact 32-color categorical palette from the official notebook."""
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
    same colored-block blend and one-pixel dark border.
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
        blended = cv2.addWeighted(color_block, alpha, image_block, 1.0 - alpha, 0)
        bordered = ImageOps.expand(
            Image.fromarray(blended), border=1, fill=(50, 50, 50)
        ).resize((right - left, bottom - top))
        canvas[top:bottom, left:right] = np.asarray(bordered)
    return Image.fromarray(canvas)


def _mixture_axis(
    axis: plt.Axes,
    mixtures: np.ndarray,
    color_map: Mapping[int, tuple[int, int, int]],
) -> None:
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
    """Save the official-style GMM mixture-proportion bar chart."""
    figure, axis = plt.subplots(figsize=(6, 3))
    _mixture_axis(axis, mixtures, color_map)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_slide_overview(
    overlays: Sequence[tuple[str, Image.Image]],
    mixtures: np.ndarray,
    color_map: Mapping[int, tuple[int, int, int]],
    slide_name: str,
    class_name: str,
    split: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Save all tissue assignment maps and the slide mixture chart as one figure."""
    panel_count = len(overlays) + 1
    columns = min(3, panel_count)
    rows = math.ceil(panel_count / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.5 * columns, 3.9 * rows), squeeze=False
    )
    flat_axes = axes.ravel()
    for axis, (tissue_name, overlay) in zip(flat_axes, overlays):
        axis.imshow(overlay)
        axis.set_title(tissue_name, fontsize=8)
        axis.axis("off")
    mixture_axis = flat_axes[len(overlays)]
    _mixture_axis(mixture_axis, mixtures, color_map)
    mixture_axis.set_title("Slide GMM mixture proportions", fontsize=9)
    for axis in flat_axes[panel_count:]:
        axis.axis("off")
    figure.suptitle(
        f"PANTHER prototypical assignments | {class_name} | {slide_name} | {split}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _safe_slug(value: str) -> str:
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
    device: torch.device,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    output_root: Path,
    split: str,
    color_map: Mapping[int, tuple[int, int, int]],
) -> Dict[str, Any]:
    item = dataset[index]
    record = dataset.records[index]
    encoded = panther(item["features"].to(device), return_assignments=True)
    responsibilities = encoded["assignments"][0].cpu().numpy()
    mixtures = encoded["mixture_weights"][0].cpu().numpy()
    hard_assignments = responsibilities.argmax(axis=1).astype(np.int64)
    tissue_assignments = _split_assignments_by_tissue(record, hard_assignments, config)

    slide_dir = (
        output_root / split / _safe_slug(str(item["class_name"]))
        / _safe_slug(str(item["slide_name"]))
    )
    slide_dir.mkdir(parents=True, exist_ok=True)
    tissue_dir = slide_dir / "tissue_assignment_maps"
    overlays: list[tuple[str, Image.Image]] = []
    tissue_payload = []
    for tissue, coordinates, labels in tissue_assignments:
        tissue_name = str(tissue["tissue_name"])
        image_path = find_tissue_image(
            Path(str(config["data_root"])), str(item["class_name"]),
            str(item["slide_name"]), tissue_name,
        )
        preview, dimensions = load_image_preview(
            image_path, int(settings["downsample"])
        )
        overlay = blend_categorical_overlay(
            preview, coordinates, labels, dimensions, int(settings["tile_size"]),
            float(settings["alpha"]), color_map,
        )
        overlays.append((tissue_name, overlay))
        tissue_output: Optional[Path] = None
        if bool(settings["save_individual_tissues"]):
            tissue_output = tissue_dir / f"{_safe_slug(tissue_name)}.png"
            tissue_output.parent.mkdir(parents=True, exist_ok=True)
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

    mixture_path = slide_dir / "mixture_proportions.png"
    overview_path = slide_dir / "assignment_overview.png"
    save_mixture_plot(mixtures, color_map, mixture_path, int(settings["dpi"]))
    save_slide_overview(
        overlays, mixtures, color_map, str(item["slide_name"]),
        str(item["class_name"]), split, overview_path, int(settings["dpi"]),
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
        "class_name": str(item["class_name"]),
        "label": int(item["label"]),
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
        "mixture_plot": str(mixture_path),
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


def run_visualization(
    config_path: Optional[str] = None,
    split_override: Optional[str] = None,
    max_slides_override: Optional[int] = None,
    slide_keys: Optional[Sequence[str]] = None,
    output_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Render original-style PANTHER visualizations for a checkpoint split."""
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
    color_map = get_default_color_map(panther.num_prototypes)

    output_root.mkdir(parents=True, exist_ok=True)
    slide_payload = []
    global_counts = np.zeros(panther.num_prototypes, dtype=np.int64)
    print(
        f"Visualizing {len(indices)} PANTHER {split} slides on {device} "
        f"from run {run_dir.name}."
    )
    for position, index in enumerate(indices, start=1):
        payload = _visualize_slide(
            datasets[split], index, panther, device, checkpoint_config, settings,
            output_root, split, color_map,
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
                "one slide-level GMM over concatenated tissues; assignments "
                "rendered separately on each aligned tissue image"
            ),
            "settings": settings,
        },
        "split": split,
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
