"""Generate L1 attribution and signed mean/std evidence heatmaps."""

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
import torch.nn.functional as F
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
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
    from .config_loader import (
        load_config,
        normalize_visualization_samples,
        resolve_inference_run_paths,
    )
    from .dataset import WSIBagDataset, create_bag_dataset
    from .evaluate import evaluation_data_config, load_checkpoint_bundle
    from .model import (
        RawFeatureStatisticsPooler,
        TorchLogisticRegression,
        normalize_pooling_statistics,
        pool_raw_features,
    )
    from .train import (
        create_dataloader,
        full_class_probabilities,
        json_safe,
        seed_everything,
    )
except ImportError:
    from config_loader import (
        load_config,
        normalize_visualization_samples,
        resolve_inference_run_paths,
    )
    from dataset import WSIBagDataset, create_bag_dataset
    from evaluate import evaluation_data_config, load_checkpoint_bundle
    from model import (
        RawFeatureStatisticsPooler,
        TorchLogisticRegression,
        normalize_pooling_statistics,
        pool_raw_features,
    )
    from train import (
        create_dataloader,
        full_class_probabilities,
        json_safe,
        seed_everything,
    )


REQUIRED_POOLING_STATISTICS: Tuple[str, ...] = ("mean", "standard_deviation")
EVIDENCE_RECONSTRUCTION_TOLERANCE = 1e-4
EVIDENCE_QUANTILE = 0.99
NEAR_ZERO_EVIDENCE = 1e-12
_MAGNITUDE_CMAP = plt.cm.magma
_EVIDENCE_CMAP = plt.cm.RdBu_r
ALL_CLASS_COLORBAR_GAP_FRACTION = 0.40


def require_mean_std_pooling(statistics: Sequence[str]) -> Tuple[str, ...]:
    """Reject pooling contracts that cannot fill the 3-by-(1+C) figure.

    Args:
        statistics (Sequence[str]): Checkpoint pooling-statistic names.

    Returns:
        Tuple[str, ...]: The required mean-then-standard-deviation contract.
    """
    names = normalize_pooling_statistics(statistics)
    if names != REQUIRED_POOLING_STATISTICS:
        raise ValueError(
            "Attribution heatmaps require pooling_statistics "
            f"{list(REQUIRED_POOLING_STATISTICS)}; received {list(names)}."
        )
    return names


def attribution_grid_maps(
    mean_evidence: np.ndarray,
    std_evidence: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build the 3-by-(1+C) panel values from signed mean and std evidence.

    Args:
        mean_evidence (np.ndarray): Signed mean evidence shaped ``[C, N]``.
        std_evidence (np.ndarray): Signed std evidence shaped ``[C, N]``.

    Returns:
        Dict[str, np.ndarray]: Class L1 maps and all-class row aggregates.
    """
    if mean_evidence.shape != std_evidence.shape or mean_evidence.ndim != 2:
        raise ValueError("Mean and std evidence must share nonempty shape [C, N].")
    if mean_evidence.shape[0] == 0 or mean_evidence.shape[1] == 0:
        raise ValueError("Mean and std evidence must share nonempty shape [C, N].")
    class_l1 = np.abs(mean_evidence) + np.abs(std_evidence)
    return {
        "class_l1": class_l1,
        "all_l1": class_l1.sum(axis=0),
        "all_mean_abs": np.abs(mean_evidence).sum(axis=0),
        "all_std_abs": np.abs(std_evidence).sum(axis=0),
    }


def expand_classifier_parameters(
    model: TorchLogisticRegression,
    num_classes: int,
    feature_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter represented-class weights into the frozen class space.

    Args:
        model (TorchLogisticRegression): Fitted estimator.
        num_classes (int): Fixed checkpoint class count.
        feature_width (int): Pooled bag-vector width.
        device (torch.device): Device receiving parameter tensors.
        dtype (torch.dtype): Floating dtype matching bag features.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Full
            weights, intercepts, scaler mean, and scaler scale.
    """
    fitted = model._require_fitted()
    classes = np.asarray(fitted["classes"], dtype=np.int64)
    if (
        classes.ndim != 1
        or classes.size < 2
        or np.any(classes < 0)
        or np.any(classes >= num_classes)
        or len(np.unique(classes)) != classes.size
    ):
        raise ValueError("Fitted classifier classes are invalid for the fixed class space.")
    coef = np.asarray(fitted["coef"])
    intercept = np.asarray(fitted["intercept"])
    mean = np.asarray(fitted["mean"])
    scale = np.asarray(fitted["scale"])
    if coef.shape != (classes.size, feature_width):
        raise ValueError("Fitted coefficient width does not match pooled features.")
    if intercept.shape != (classes.size,):
        raise ValueError("Fitted intercepts do not match represented classes.")
    if mean.shape != (feature_width,) or scale.shape != (feature_width,):
        raise ValueError("Fitted scaler arrays do not match pooled features.")
    weights = torch.zeros((num_classes, feature_width), device=device, dtype=dtype)
    biases = torch.zeros((num_classes,), device=device, dtype=dtype)
    class_index = torch.as_tensor(classes, dtype=torch.long, device=device)
    weights[class_index] = torch.as_tensor(coef, device=device, dtype=dtype)
    biases[class_index] = torch.as_tensor(intercept, device=device, dtype=dtype)
    mean_tensor = torch.as_tensor(mean, device=device, dtype=dtype)
    scale_tensor = torch.as_tensor(scale, device=device, dtype=dtype)
    if bool((scale_tensor <= 0.0).any()):
        raise ValueError("Fitted scaler scale must be positive.")
    return weights, biases, mean_tensor, scale_tensor


def compute_tile_attribution(
    features: torch.Tensor,
    masks: torch.Tensor,
    model: TorchLogisticRegression,
    pooling_statistics: Sequence[str],
    epsilon: float,
    num_classes: int,
    tolerance: float = EVIDENCE_RECONSTRUCTION_TOLERANCE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose each class logit into mask-aware mean and std tile evidence.

    Args:
        features (torch.Tensor): Tile features shaped ``[B, T, D]``.
        masks (torch.Tensor): Boolean valid-tile mask shaped ``[B, T]``.
        model (TorchLogisticRegression): Fitted standardized linear classifier.
        pooling_statistics (Sequence[str]): Checkpoint pooling-statistic names.
        epsilon (float): Nonnegative population-std variance floor.
        num_classes (int): Fixed checkpoint class count.
        tolerance (float): Maximum absolute logit reconstruction error.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Mean evidence ``[B, C, T]``, std evidence ``[B, C, T]``, absolute
            reconstruction errors ``[B, C]``, bag baselines ``[B, C]``, and
            full-space logits ``[B, C]``.
    """
    require_mean_std_pooling(pooling_statistics)
    if features.ndim != 3 or masks.shape != features.shape[:2]:
        raise ValueError("Features and masks must have shapes [B, T, D] and [B, T].")
    if masks.dtype != torch.bool:
        raise TypeError("Attribution masks must be boolean.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two.")
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative.")
    if tolerance < 0.0:
        raise ValueError("Evidence reconstruction tolerance must be nonnegative.")
    if not torch.isfinite(features).all():
        raise ValueError("features contain non-finite values.")
    execution_device = torch.device(model.device)
    feature_tensor = features.to(device=execution_device, dtype=torch.float32)
    valid = masks.to(device=execution_device, dtype=torch.bool)
    if torch.any(valid.sum(dim=1) == 0):
        raise ValueError("Every bag must contain at least one valid tile.")
    pooled = pool_raw_features(
        feature_tensor,
        masks=valid,
        epsilon=epsilon,
        statistics=REQUIRED_POOLING_STATISTICS,
    )
    input_dim = int(feature_tensor.shape[2])
    if pooled.shape[1] != 2 * input_dim:
        raise ValueError("Pooled bag vectors are not concatenated mean and std blocks.")
    weights, intercept, scaler_mean, scaler_scale = expand_classifier_parameters(
        model,
        num_classes=num_classes,
        feature_width=int(pooled.shape[1]),
        device=execution_device,
        dtype=feature_tensor.dtype,
    )
    standardized = (pooled - scaler_mean) / scaler_scale
    logits = F.linear(standardized, weights, intercept)
    alpha = weights / scaler_scale.unsqueeze(0)
    alpha_mean = alpha[:, :input_dim]
    alpha_std = alpha[:, input_dim:]
    bag_mean = pooled[:, :input_dim]
    bag_std = pooled[:, input_dim:]
    tile_weights = valid.unsqueeze(-1).to(dtype=feature_tensor.dtype)
    tile_counts = valid.sum(dim=1).to(dtype=feature_tensor.dtype)
    mean_evidence = torch.einsum(
        "cd,btd->bct",
        alpha_mean,
        feature_tensor * tile_weights,
    ) / tile_counts.view(-1, 1, 1)
    centered = (feature_tensor - bag_mean.unsqueeze(1)) * tile_weights
    squared = centered.square()
    variance_mass = squared.sum(dim=1)
    positive_variance = variance_mass > 0.0
    share = torch.where(
        positive_variance.unsqueeze(1),
        squared / variance_mass.unsqueeze(1).clamp_min(NEAR_ZERO_EVIDENCE),
        torch.zeros_like(squared),
    )
    std_evidence = torch.einsum("cd,bd,btd->bct", alpha_std, bag_std, share)
    mean_evidence = mean_evidence.masked_fill(~valid.unsqueeze(1), 0.0)
    std_evidence = std_evidence.masked_fill(~valid.unsqueeze(1), 0.0)
    scaler_baseline = intercept - (alpha * scaler_mean.unsqueeze(0)).sum(dim=1)
    zero_variance = (~positive_variance).to(dtype=feature_tensor.dtype)
    epsilon_mass = torch.einsum("cd,bd,bd->bc", alpha_std, bag_std, zero_variance)
    bag_baselines = scaler_baseline.unsqueeze(0) + epsilon_mass
    reconstructed = mean_evidence.sum(dim=-1) + std_evidence.sum(dim=-1) + bag_baselines
    residual = logits - reconstructed
    mean_evidence = mean_evidence + (
        residual.unsqueeze(-1)
        * valid.unsqueeze(1).to(dtype=feature_tensor.dtype)
        / tile_counts.view(-1, 1, 1)
    )
    reconstructed = mean_evidence.sum(dim=-1) + std_evidence.sum(dim=-1) + bag_baselines
    remaining = logits - reconstructed
    if float(remaining.abs().max().item()) > tolerance:
        first_valid = valid.to(dtype=torch.long).argmax(dim=1)
        scatter_index = first_valid.view(-1, 1, 1).expand(-1, num_classes, 1)
        mean_evidence = mean_evidence.scatter_add(
            dim=-1,
            index=scatter_index,
            src=remaining.unsqueeze(-1),
        )
    reconstructed = mean_evidence.sum(dim=-1) + std_evidence.sum(dim=-1) + bag_baselines
    reconstruction_errors = (reconstructed - logits).abs()
    if not torch.isfinite(mean_evidence).all() or not torch.isfinite(std_evidence).all():
        raise ValueError("Tile evidence must be finite.")
    maximum_error = float(reconstruction_errors.max().item())
    if maximum_error > tolerance:
        raise RuntimeError(
            "Tile evidence failed to reconstruct the bag logits: "
            f"maximum absolute error {maximum_error:.6g} exceeds {tolerance:.6g}."
        )
    return mean_evidence, std_evidence, reconstruction_errors, bag_baselines, logits


def evidence_color_limit(
    evidence: np.ndarray,
    quantile: float = EVIDENCE_QUANTILE,
) -> float:
    """Calculate a robust symmetric color limit for signed tile evidence.

    Args:
        evidence (np.ndarray): Flattened or one-dimensional signed evidence.
        quantile (float): Quantile of absolute evidence used as the limit.

    Returns:
        float: Positive symmetric limit for a zero-centered colormap.
    """
    values = np.asarray(evidence, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Evidence must be a nonempty array.")
    if not np.isfinite(values).all():
        raise ValueError("Evidence values must be finite.")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("Evidence quantile must be in the interval (0, 1].")
    limit = float(np.quantile(np.abs(values), quantile))
    return max(limit, NEAR_ZERO_EVIDENCE)


def magnitude_color_limit(
    magnitude: np.ndarray,
    quantile: float = EVIDENCE_QUANTILE,
) -> float:
    """Calculate a robust nonnegative color limit for L1 magnitude maps.

    Args:
        magnitude (np.ndarray): Flattened or one-dimensional nonnegative values.
        quantile (float): Quantile used as the upper limit.

    Returns:
        float: Positive sequential colormap limit.
    """
    values = np.asarray(magnitude, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Magnitude must be a nonempty array.")
    if not np.isfinite(values).all():
        raise ValueError("Magnitude values must be finite.")
    if np.any(values < -NEAR_ZERO_EVIDENCE):
        raise ValueError("Magnitude values must be nonnegative.")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("Magnitude quantile must be in the interval (0, 1].")
    limit = float(np.quantile(np.clip(values, 0.0, None), quantile))
    return max(limit, NEAR_ZERO_EVIDENCE)


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
            np.column_stack((x_coordinates - half_tile, y_coordinates - half_tile)),
            np.column_stack((x_coordinates + half_tile, y_coordinates - half_tile)),
            np.column_stack((x_coordinates + half_tile, y_coordinates + half_tile)),
            np.column_stack((x_coordinates - half_tile, y_coordinates + half_tile)),
        ),
        axis=1,
    )


def draw_tile_map(
    axis: Axes,
    coordinates: np.ndarray,
    values: np.ndarray,
    tile_size: int,
    colormap: Any,
    vmin: float,
    vmax: float,
) -> None:
    """Draw aligned tile rectangles colored by one attribution map.

    Args:
        axis (Axes): Matplotlib axis receiving tile rectangles.
        coordinates (np.ndarray): Level-zero coordinates shaped ``[N, 2]``.
        values (np.ndarray): Per-tile values shaped ``[N]``.
        tile_size (int): Tile width and height in level-zero pixels.
        colormap (Any): Matplotlib colormap used for face colors.
        vmin (float): Colormap lower bound.
        vmax (float): Colormap upper bound.

    Returns:
        None: Rectangles and spatial limits are applied in place.
    """
    if coordinates.shape != (values.shape[0], 2):
        raise ValueError("Map values and coordinates are not exactly aligned.")
    if vmax <= vmin:
        raise ValueError("Colormap upper bound must exceed the lower bound.")
    normalization = plt.Normalize(vmin=vmin, vmax=vmax)
    half_tile = tile_size / 2.0
    collection = PolyCollection(
        tile_vertices(coordinates, tile_size),
        linewidths=0,
        edgecolors="none",
        facecolors=colormap(normalization(values)),
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


def add_map_colorbar(
    figure: plt.Figure,
    axis: Union[Axes, Sequence[Axes]],
    colormap: Any,
    vmin: float,
    vmax: float,
    label: Optional[str] = None,
    cax: Optional[Axes] = None,
) -> None:
    """Attach a colorbar matching one panel's colormap bounds.

    Args:
        figure (plt.Figure): Figure that owns the colorbar.
        axis (Union[Axes, Sequence[Axes]]): Heatmap axis or axes described by
            the colorbar. Passing every class axis in a row keeps one shared
            scale for that entire class row.
        colormap (Any): Matplotlib colormap.
        vmin (float): Colormap lower bound.
        vmax (float): Colormap upper bound.
        label (Optional[str]): Optional colorbar label.
        cax (Optional[Axes]): Dedicated colorbar axis. When provided, heatmap
            axes are not resized to make room for the colorbar.

    Returns:
        None: The colorbar is added to ``figure``.
    """
    scalar_mappable = plt.cm.ScalarMappable(
        cmap=colormap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    scalar_mappable.set_array([])
    colorbar_kwargs: Dict[str, Any] = {
        "ax": axis,
        "orientation": "vertical",
    }
    if cax is None:
        colorbar_kwargs["pad"] = 0.04
    else:
        colorbar_kwargs["cax"] = cax
    colorbar = figure.colorbar(scalar_mappable, **colorbar_kwargs)
    if label is not None:
        colorbar.set_label(label)


def shift_colorbar_toward_heatmap(
    colorbar_axis: Axes,
    heatmap_axis: Axes,
    gap_fraction: float,
) -> None:
    """Slide a colorbar left toward its heatmap without changing its size.

    Args:
        colorbar_axis (Axes): Colorbar axis to move.
        heatmap_axis (Axes): Heatmap the colorbar describes.
        gap_fraction (float): Fraction of the current gap to close, in ``(0, 1)``.

    Returns:
        None: The colorbar axis position is updated in place.
    """
    if not 0.0 < gap_fraction < 1.0:
        raise ValueError("gap_fraction must be in the interval (0, 1).")
    heatmap_position = heatmap_axis.get_position()
    colorbar_position = colorbar_axis.get_position()
    gap = colorbar_position.x0 - heatmap_position.x1
    if gap <= 0.0:
        return
    colorbar_axis.set_position(
        (
            colorbar_position.x0 - gap * gap_fraction,
            colorbar_position.y0,
            colorbar_position.width,
            colorbar_position.height,
        )
    )


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
    if thumbnail_size <= 0:
        raise ValueError("thumbnail_size must be positive.")
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


def _predicted_title(label: str, is_predicted: bool) -> Tuple[str, Dict[str, Any]]:
    """Return a class-panel title and optional predicted-class highlight.

    Args:
        label (str): Panel title without predicted-class decoration.
        is_predicted (bool): Whether this column is the predicted class.

    Returns:
        Tuple[str, Dict[str, Any]]: Display title and ``set_title`` kwargs.
    """
    if not is_predicted:
        return label, {"fontweight": "normal", "bbox": None}
    return (
        f"★ {label}",
        {
            "fontweight": "bold",
            "bbox": {"facecolor": "gold", "alpha": 0.35, "edgecolor": "darkorange"},
        },
    )


def save_attribution_figure(
    mean_evidence: np.ndarray,
    std_evidence: np.ndarray,
    coordinates: np.ndarray,
    class_names: Sequence[str],
    predicted_index: int,
    slide_name: str,
    tissue_name: str,
    true_class: str,
    predicted_class: str,
    predicted_probability: float,
    class_probabilities: Sequence[float],
    image_path: Optional[Path],
    output_path: Path,
    tile_size: int,
    thumbnail_size: int,
    render_dpi: int = 150,
) -> None:
    """Save a 3-by-(1+C) attribution grid with the original tissue on the left.

    Args:
        mean_evidence (np.ndarray): Signed mean evidence shaped ``[C, N]``.
        std_evidence (np.ndarray): Signed std evidence shaped ``[C, N]``.
        coordinates (np.ndarray): Aligned level-zero coordinates shaped ``[N, 2]``.
        class_names (Sequence[str]): Ordered class display names.
        predicted_index (int): Predicted class index highlighted on the first row.
        slide_name (str): Slide directory name.
        tissue_name (str): Tissue basename.
        true_class (str): Ground-truth class name.
        predicted_class (str): Predicted class name.
        predicted_probability (float): Probability of the predicted class.
        class_probabilities (Sequence[float]): Full-class probabilities aligned
            with ``class_names``.
        image_path (Optional[Path]): Optional source tissue image.
        output_path (Path): Destination PNG path.
        tile_size (int): Tile size in level-zero source-image pixels.
        thumbnail_size (int): Maximum original-tissue thumbnail size.
        render_dpi (int): Positive output resolution in dots per inch.

    Returns:
        None: Combined attribution figure is written to ``output_path``.
    """
    class_count = len(class_names)
    if mean_evidence.shape != (class_count, coordinates.shape[0]):
        raise ValueError("Mean evidence and coordinates must align for every class.")
    if std_evidence.shape != mean_evidence.shape:
        raise ValueError("Std evidence must match mean evidence shape.")
    if not 0 <= predicted_index < class_count:
        raise ValueError("Predicted index is outside the class range.")
    if render_dpi <= 0:
        raise ValueError("render_dpi must be positive.")
    probability_values = np.asarray(class_probabilities, dtype=np.float64)
    if probability_values.shape != (class_count,):
        raise ValueError("class_probabilities must contain one probability per class.")
    if not np.isfinite(probability_values).all():
        raise ValueError("class_probabilities must be finite.")
    if np.any(probability_values < 0.0):
        raise ValueError("class_probabilities must be nonnegative.")
    if abs(float(probability_values[predicted_index]) - float(predicted_probability)) > 1e-6:
        raise ValueError("predicted_probability does not match class_probabilities.")
    maps = attribution_grid_maps(mean_evidence, std_evidence)
    l1_limit = magnitude_color_limit(maps["class_l1"])
    all_l1_limit = magnitude_color_limit(maps["all_l1"])
    mean_limit = evidence_color_limit(mean_evidence)
    std_limit = evidence_color_limit(std_evidence)
    all_mean_limit = magnitude_color_limit(maps["all_mean_abs"])
    all_std_limit = magnitude_color_limit(maps["all_std_abs"])

    figure = plt.figure(figsize=(3.4 * (class_count + 1) + 8.0, 11.0))
    all_map_column = 1
    all_colorbar_column = 2
    class_map_start = 3
    class_colorbar_column = class_count + 3
    grid = figure.add_gridspec(
        3,
        class_count + 4,
        width_ratios=[2.4, 1.0, 0.07] + [1.0] * class_count + [0.07],
        hspace=0.38,
        wspace=0.30,
    )
    row_specs = (
        (
            maps["all_l1"],
            maps["class_l1"],
            _MAGNITUDE_CMAP,
            0.0,
            all_l1_limit,
            0.0,
            l1_limit,
            "L1 magnitude",
            "L1 attribution magnitude",
        ),
        (
            maps["all_mean_abs"],
            mean_evidence,
            _MAGNITUDE_CMAP,
            0.0,
            all_mean_limit,
            -mean_limit,
            mean_limit,
            "Mean evidence",
            "Signed mean logit evidence",
        ),
        (
            maps["all_std_abs"],
            std_evidence,
            _MAGNITUDE_CMAP,
            0.0,
            all_std_limit,
            -std_limit,
            std_limit,
            "Std evidence",
            "Signed std logit evidence",
        ),
    )
    for row_index, spec in enumerate(row_specs):
        (
            all_values,
            class_values,
            all_cmap,
            all_vmin,
            all_vmax,
            class_vmin,
            class_vmax,
            row_name,
            class_colorbar_label,
        ) = spec
        all_vmax = all_vmax if all_vmax > all_vmin else all_vmin + NEAR_ZERO_EVIDENCE
        class_vmax = class_vmax if class_vmax > class_vmin else class_vmin + NEAR_ZERO_EVIDENCE
        all_axis = figure.add_subplot(grid[row_index, all_map_column])
        draw_tile_map(
            all_axis,
            coordinates,
            all_values,
            tile_size,
            all_cmap,
            all_vmin,
            all_vmax,
        )
        if row_index == 0:
            all_axis.set_title("All classes")
        all_cax = figure.add_subplot(grid[row_index, all_colorbar_column])
        add_map_colorbar(
            figure,
            all_axis,
            all_cmap,
            all_vmin,
            all_vmax,
            label=row_name,
            cax=all_cax,
        )
        shift_colorbar_toward_heatmap(
            all_cax,
            all_axis,
            ALL_CLASS_COLORBAR_GAP_FRACTION,
        )
        class_cmap = _MAGNITUDE_CMAP if row_index == 0 else _EVIDENCE_CMAP
        class_axes: List[Axes] = []
        for class_index, class_name in enumerate(class_names):
            axis = figure.add_subplot(grid[row_index, class_map_start + class_index])
            draw_tile_map(
                axis,
                coordinates,
                class_values[class_index],
                tile_size,
                class_cmap,
                class_vmin,
                class_vmax,
            )
            if row_index == 0:
                title, title_kwargs = _predicted_title(
                    f"{class_name} ({probability_values[class_index]:.3f})",
                    class_index == predicted_index,
                )
                axis.set_title(title, **title_kwargs)
            class_axes.append(axis)
        add_map_colorbar(
            figure,
            class_axes,
            class_cmap,
            class_vmin,
            class_vmax,
            label=class_colorbar_label,
            cax=figure.add_subplot(grid[row_index, class_colorbar_column]),
        )

    original_axis = figure.add_subplot(grid[:, 0])
    thumbnail = load_tissue_thumbnail(image_path, thumbnail_size)
    if thumbnail is None:
        original_axis.text(
            0.5, 0.5, "Thumbnail\nNot Available", ha="center", va="center"
        )
    else:
        original_axis.imshow(thumbnail)
    original_axis.set_title("Original", fontweight="bold")
    original_axis.axis("off")
    figure.suptitle(
        f"Slide: {slide_name} | Tissue: {tissue_name}\n"
        f"True: {true_class} | Predicted: {predicted_class} "
        f"({predicted_probability:.3f}) | Red supports, blue opposes",
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
    coordinates: torch.Tensor,
    tissue_indices: torch.Tensor,
    tissue_names: Sequence[str],
) -> None:
    """Validate exact post-collation alignment for one unpadded bag.

    Args:
        coordinates (torch.Tensor): Coordinates shaped ``[N, 2]``.
        tissue_indices (torch.Tensor): Tissue provenance shaped ``[N]``.
        tissue_names (Sequence[str]): Tissue names indexed by provenance values.

    Returns:
        None: Validation succeeds by returning normally.
    """
    tile_count = coordinates.shape[0]
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Coordinates must have shape [N, 2].")
    if tissue_indices.shape != (tile_count,):
        raise ValueError("Collated tissue indices are not aligned with coordinates.")
    if not torch.isfinite(coordinates).all():
        raise ValueError("Valid coordinates must be finite.")
    if tile_count == 0:
        raise ValueError("Cannot visualize an empty bag.")
    if bool(((tissue_indices < 0) | (tissue_indices >= len(tissue_names))).any()):
        raise ValueError("A valid tile has an invalid tissue provenance index.")


def resolve_visualization_settings(
    runtime_config: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
    samples_override: Optional[Sequence[str]],
    dpi_override: Optional[int],
    render_workers_override: Optional[int],
) -> Dict[str, Any]:
    """Resolve runtime visualization settings with checkpoint fallbacks.

    Args:
        runtime_config (Mapping[str, Any]): Launcher YAML configuration.
        checkpoint_config (Mapping[str, Any]): Embedded training configuration.
        samples_override (Optional[Sequence[str]]): Optional CLI split list.
        dpi_override (Optional[int]): Optional CLI DPI override.
        render_workers_override (Optional[int]): Optional CLI worker override.

    Returns:
        Dict[str, Any]: Resolved samples, tile size, DPI, workers, and thumbnail.
    """
    runtime_visualization = runtime_config.get("visualization") or {}
    checkpoint_visualization = checkpoint_config.get("visualization") or {}
    if runtime_visualization and not isinstance(runtime_visualization, Mapping):
        raise ValueError("Runtime visualization config must be a mapping.")
    if checkpoint_visualization and not isinstance(checkpoint_visualization, Mapping):
        raise ValueError("Checkpoint visualization config must be a mapping.")
    visualization = dict(checkpoint_visualization)
    visualization.update(dict(runtime_visualization))
    samples = (
        normalize_visualization_samples(samples_override)
        if samples_override is not None
        else normalize_visualization_samples(visualization.get("samples"))
    )
    dpi = int(dpi_override if dpi_override is not None else visualization.get("dpi", 150))
    render_workers = int(
        render_workers_override
        if render_workers_override is not None
        else visualization.get("render_workers", 8)
    )
    tile_size = int(visualization.get("tile_size", 448))
    thumbnail_size = int(visualization.get("thumbnail_size", 1024))
    if dpi <= 0:
        raise ValueError("Visualization DPI must be positive.")
    if render_workers <= 0:
        raise ValueError("visualization.render_workers must be positive.")
    if tile_size <= 0:
        raise ValueError("visualization.tile_size must be positive.")
    if thumbnail_size <= 0:
        raise ValueError("visualization.thumbnail_size must be positive.")
    return {
        "samples": samples,
        "dpi": dpi,
        "render_workers": render_workers,
        "tile_size": tile_size,
        "thumbnail_size": thumbnail_size,
    }


def render_split_attributions(
    model: TorchLogisticRegression,
    dataset: WSIBagDataset,
    config: Mapping[str, Any],
    class_folders: Sequence[str],
    pooler: RawFeatureStatisticsPooler,
    pooling_statistics: Sequence[str],
    epsilon: float,
    data_root: str,
    output_dir: Path,
    bag_level: str,
    split: str,
    tile_size: int,
    thumbnail_size: int,
    render_dpi: int,
    render_workers: int,
) -> List[Dict[str, Any]]:
    """Render per-tissue attribution figures for one nonempty split.

    Args:
        model (TorchLogisticRegression): Loaded fitted classifier.
        dataset (WSIBagDataset): Split dataset at the checkpoint bag level.
        config (Mapping[str, Any]): Evaluation loading configuration.
        class_folders (Sequence[str]): Frozen checkpoint class order.
        pooler (RawFeatureStatisticsPooler): Checkpoint-compatible pooler.
        pooling_statistics (Sequence[str]): Checkpoint pooling-statistic names.
        epsilon (float): Population-std variance floor.
        data_root (str): Root containing class and slide directories.
        output_dir (Path): Directory receiving figures and the split summary.
        bag_level (str): Checkpoint bag level, tissue or slide.
        split (str): Rendered split name.
        tile_size (int): Tile size in source-image pixels.
        thumbnail_size (int): Maximum original-tissue thumbnail size.
        render_dpi (int): Positive output DPI.
        render_workers (int): Number of independent figure-rendering processes.

    Returns:
        List[Dict[str, Any]]: One attribution summary per rendered tissue.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot visualize an empty dataset.")
    if render_workers <= 0:
        raise ValueError("render_workers must be positive.")
    dataset.set_epoch(0)
    dataloader = create_dataloader(dataset, config)
    results: List[Dict[str, Any]] = []
    render_futures: List[Future[None]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_folders)
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
            for batch in tqdm(dataloader, desc=f"Preparing {bag_level}/{split} bags"):
                features = batch["features"]
                masks = batch["masks"]
                (
                    mean_evidence,
                    std_evidence,
                    reconstruction_errors,
                    bag_baselines,
                    logits,
                ) = compute_tile_attribution(
                    features=features,
                    masks=masks,
                    model=model,
                    pooling_statistics=pooling_statistics,
                    epsilon=epsilon,
                    num_classes=num_classes,
                )
                pooled = pooler(features, masks)
                probabilities = full_class_probabilities(
                    model,
                    pooled.detach().cpu().numpy(),
                    num_classes,
                )
                predictions = np.asarray(model.predict(pooled), dtype=np.int64)
                labels = batch["labels"].detach().cpu().numpy()
                mean_evidence = mean_evidence.detach().cpu()
                std_evidence = std_evidence.detach().cpu()
                reconstruction_errors = reconstruction_errors.detach().cpu()
                bag_baselines = bag_baselines.detach().cpu()
                logits = logits.detach().cpu()
                del pooled

                for bag_index, slide_name in enumerate(batch["slide_names"]):
                    valid_mask = batch["masks"][bag_index]
                    bag_mean = mean_evidence[bag_index, :, valid_mask]
                    bag_std = std_evidence[bag_index, :, valid_mask]
                    coordinates = batch["coordinates"][bag_index, valid_mask].detach().cpu()
                    tissue_indices = batch["tissue_indices"][
                        bag_index, valid_mask
                    ].detach().cpu()
                    tissue_names = [
                        str(name) for name in batch["bag_tissue_names"][bag_index]
                    ]
                    validate_aligned_bag(coordinates, tissue_indices, tissue_names)
                    true_index = int(labels[bag_index])
                    predicted_index = int(predictions[bag_index])
                    true_class = class_folders[true_index]
                    predicted_class = class_folders[predicted_index]
                    predicted_probability = float(
                        probabilities[bag_index, predicted_index]
                    )
                    class_probabilities = tuple(
                        float(value) for value in probabilities[bag_index].tolist()
                    )
                    bag_mean_sums = bag_mean.sum(dim=-1)
                    bag_std_sums = bag_std.sum(dim=-1)
                    for tissue_index, tissue_name in enumerate(tissue_names):
                        tissue_mask = tissue_indices == tissue_index
                        if not bool(tissue_mask.any()):
                            continue
                        tissue_mean = bag_mean[:, tissue_mask].numpy()
                        tissue_std = bag_std[:, tissue_mask].numpy()
                        tissue_coordinates = coordinates[tissue_mask].numpy()
                        image_path = find_tissue_image(
                            data_root, true_class, str(slide_name), tissue_name
                        )
                        output_path = output_dir / (
                            f"{safe_filename(str(slide_name))}_"
                            f"{safe_filename(tissue_name)}_attribution.png"
                        )
                        render_arguments = {
                            "mean_evidence": tissue_mean,
                            "std_evidence": tissue_std,
                            "coordinates": tissue_coordinates,
                            "class_names": class_folders,
                            "predicted_index": predicted_index,
                            "slide_name": str(slide_name),
                            "tissue_name": tissue_name,
                            "true_class": true_class,
                            "predicted_class": predicted_class,
                            "predicted_probability": predicted_probability,
                            "class_probabilities": class_probabilities,
                            "image_path": image_path,
                            "output_path": output_path,
                            "tile_size": tile_size,
                            "thumbnail_size": thumbnail_size,
                            "render_dpi": render_dpi,
                        }
                        if render_executor is None:
                            save_attribution_figure(**render_arguments)
                        else:
                            render_futures.append(
                                render_executor.submit(
                                    save_attribution_figure,
                                    **render_arguments,
                                )
                            )
                        maps = attribution_grid_maps(tissue_mean, tissue_std)
                        results.append(
                            {
                                "bag_level": bag_level,
                                "split": split,
                                "slide_name": str(slide_name),
                                "tissue_name": tissue_name,
                                "true_class": true_class,
                                "predicted_class": predicted_class,
                                "predicted_probability": predicted_probability,
                                "num_tiles": int(tissue_mean.shape[1]),
                                "bag_num_tiles": int(valid_mask.sum().item()),
                                "heatmap_path": str(output_path),
                                "logit_reconstruction_error": [
                                    float(value)
                                    for value in reconstruction_errors[bag_index].tolist()
                                ],
                                "bag_baselines": [
                                    float(value)
                                    for value in bag_baselines[bag_index].tolist()
                                ],
                                "bag_logits": [
                                    float(value) for value in logits[bag_index].tolist()
                                ],
                                "bag_mean_evidence_sum": [
                                    float(value) for value in bag_mean_sums.tolist()
                                ],
                                "bag_std_evidence_sum": [
                                    float(value) for value in bag_std_sums.tolist()
                                ],
                                "tissue_l1_mass": float(maps["all_l1"].sum()),
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


def visualize_attribution(
    config_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    output_dir: Optional[str] = None,
    samples: Optional[Sequence[str]] = None,
    dpi: Optional[int] = None,
    render_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Render attribution heatmaps for the configured visualization splits.

    Args:
        config_path (Optional[str]): Runtime YAML path or package default.
        checkpoint (Optional[str]): Explicit checkpoint path override.
        output_dir (Optional[str]): Explicit heatmap directory override.
        samples (Optional[Sequence[str]]): Optional split-name override.
        dpi (Optional[int]): Optional output DPI override.
        render_workers (Optional[int]): Optional figure-process override.

    Returns:
        Dict[str, Any]: JSON-safe manifest including per-split summaries.
    """
    runtime_config = load_config(config_path)
    resolve_inference_run_paths(runtime_config)
    checkpoint_path = Path(
        checkpoint
        if checkpoint is not None
        else str(runtime_config["paths"]["checkpoint"])
    ).expanduser().resolve()
    destination = Path(
        output_dir
        if output_dir is not None
        else str(runtime_config["paths"]["attribution_output"])
    ).expanduser().resolve()
    bundle = load_checkpoint_bundle(checkpoint_path)
    checkpoint_config = bundle["config"]
    class_folders = list(bundle["class_folders"])
    pooling_statistics = require_mean_std_pooling(bundle["pooling_statistics"])
    data_config = evaluation_data_config(checkpoint_config, runtime_config)
    settings = resolve_visualization_settings(
        runtime_config,
        checkpoint_config,
        samples,
        dpi,
        render_workers,
    )
    seed_everything(int(data_config["random_seed"]))
    pooler = RawFeatureStatisticsPooler(
        epsilon=float(bundle["pooling_population_std_epsilon"]),
        statistics=list(pooling_statistics),
    )
    destination.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}
    for split in settings["samples"]:
        dataset = create_bag_dataset(
            data_config,
            split,
            class_folders=class_folders,
            bag_level=str(bundle["bag_level"]),
        )
        if dataset.class_folders != class_folders:
            raise ValueError(
                f"{split} dataset class order violates checkpoint contract."
            )
        if len(dataset) == 0:
            results[split] = {
                "skipped": True,
                "reason": "empty split",
                "tissues": [],
                "summary": None,
            }
            print(f"{split} skipped: empty split")
            continue
        tissues = render_split_attributions(
            model=bundle["model"],
            dataset=dataset,
            config=data_config,
            class_folders=class_folders,
            pooler=pooler,
            pooling_statistics=pooling_statistics,
            epsilon=float(bundle["pooling_population_std_epsilon"]),
            data_root=str(data_config["data_root"]),
            output_dir=destination,
            bag_level=str(bundle["bag_level"]),
            split=split,
            tile_size=int(settings["tile_size"]),
            thumbnail_size=int(settings["thumbnail_size"]),
            render_dpi=int(settings["dpi"]),
            render_workers=int(settings["render_workers"]),
        )
        summary_path = destination / f"attribution_summary_{split}.json"
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(json_safe(tissues), summary_file, indent=2)
        results[split] = {
            "skipped": False,
            "num_tissues": len(tissues),
            "num_bags": len(dataset),
            "summary": str(summary_path.resolve()),
            "tissues": tissues,
        }
        print(
            f"{split} bags={len(dataset)} tissues={len(tissues)} "
            f"summary={summary_path}"
        )
    manifest_path = destination / "attribution_manifest.json"
    manifest: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "class_folders": class_folders,
        "bag_level": str(bundle["bag_level"]),
        "samples": list(settings["samples"]),
        "output_dir": str(destination),
        "results": results,
        "manifest": str(manifest_path),
    }
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(json_safe(manifest), manifest_file, indent=2)
    print(f"Saved attribution manifest: {manifest_path}")
    return json_safe(manifest)


def parse_args() -> argparse.Namespace:
    """Parse attribution-visualization command-line arguments.

    Args:
        None: This function reads process command-line arguments.

    Returns:
        argparse.Namespace: Parsed configuration, paths, and overrides.
    """
    parser = argparse.ArgumentParser(
        description="Visualize logistic-regression L1 attribution heatmaps."
    )
    parser.add_argument("--config", type=str, default=None, help="Runtime YAML.")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint joblib override."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Heatmap output override."
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Split names to render; defaults to visualization.samples.",
    )
    parser.add_argument("--dpi", type=int, default=None, help="Output DPI override.")
    parser.add_argument(
        "--render-workers",
        type=int,
        default=None,
        help="Figure-rendering process count.",
    )
    return parser.parse_args()


def main() -> None:
    """Run checkpoint-driven logistic-regression attribution visualization.

    Args:
        None: Configuration and overrides are read from the command line.

    Returns:
        None: Heatmaps and JSON summaries are written to disk.
    """
    arguments = parse_args()
    visualize_attribution(
        config_path=arguments.config,
        checkpoint=arguments.checkpoint,
        output_dir=arguments.output_dir,
        samples=arguments.samples,
        dpi=arguments.dpi,
        render_workers=arguments.render_workers,
    )


if __name__ == "__main__":
    main()
