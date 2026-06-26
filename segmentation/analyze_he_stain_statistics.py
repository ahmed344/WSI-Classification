from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib
import numpy as np
import openslide
from scipy import stats
from skimage.color import rgb2hed

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_EXAMPLE_SLIDE = Path(
    "/workspaces/WSI-Classification/data/HE-MYO/Processed/Dystrophic/"
    "P841738 - 2026-05-12 17.15.06/1920_37632.ome.tiff"
)
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parent / "he_stain_statistics.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "he_stain_statistics_visualizations"
POSITIVE_DISTRIBUTIONS = {
    "burr12": stats.burr12,
    "fisk": stats.fisk,
    "lognorm": stats.lognorm,
    "gamma": stats.gamma,
    "weibull_min": stats.weibull_min,
    "invgauss": stats.invgauss,
    "betaprime": stats.betaprime,
}
DEFAULT_DISTRIBUTION_CANDIDATES = tuple(POSITIVE_DISTRIBUTIONS)


@dataclass(frozen=True)
class StainDistributionFit:
    """Store fitted positive-distribution parameters for one stain signal.

    Args:
        distribution (str): Name of the fitted distribution.
        positive_fraction (float): Fraction of finite tissue pixels with strictly positive stain signal.
        shape (float): Primary fitted distribution shape parameter.
        second_shape (float): Secondary fitted distribution shape parameter.
        scale (float): Fitted distribution scale parameter.
        mean (float): Fitted distribution mean.
        std (float): Fitted distribution standard deviation.
        mode (float): Fitted distribution mode estimated from its PDF.
        q25 (float): Fitted distribution 25th percentile.
        median (float): Fitted distribution median.
        q75 (float): Fitted distribution 75th percentile.
        q95 (float): Fitted distribution 95th percentile.
        q99 (float): Fitted distribution 99th percentile.
        fit_percentile (float): Upper positive-signal percentile used for distribution fitting.
        aic (float): Akaike information criterion for the fit on sampled positive values.

    Returns:
        StainDistributionFit: Immutable container for fitted positive-distribution parameters.
    """

    distribution: str
    positive_fraction: float
    shape: float
    second_shape: float
    scale: float
    mean: float
    std: float
    mode: float
    q25: float
    median: float
    q75: float
    q95: float
    q99: float
    fit_percentile: float
    aic: float


@dataclass(frozen=True)
class SlideStainStatistics:
    """Store H&E fitted positive-distribution statistics for one tissue slide.

    Args:
        slide_path (Path): Path to the analyzed tissue slide.
        slide_width (int): Full-resolution slide width in pixels.
        slide_height (int): Full-resolution slide height in pixels.
        analysis_width (int): Width of the image used for analysis.
        analysis_height (int): Height of the image used for analysis.
        n_tissue_pixels_available (int): Number of masked tissue pixels available in the analysis image.
        n_tissue_pixels_used (int): Number of tissue pixels sampled for distribution fitting.
        hematoxylin (StainDistributionFit): Fitted distribution parameters for the Hematoxylin signal.
        eosin (StainDistributionFit): Fitted distribution parameters for the Eosin signal.
        visualization_path (Path): Path to the saved visualization image.

    Returns:
        SlideStainStatistics: Immutable container for one slide's fitted stain statistics.
    """

    slide_path: Path
    slide_width: int
    slide_height: int
    analysis_width: int
    analysis_height: int
    n_tissue_pixels_available: int
    n_tissue_pixels_used: int
    hematoxylin: StainDistributionFit
    eosin: StainDistributionFit
    visualization_path: Path


def parse_hsv_triplet(values: Sequence[int]) -> tuple[int, int, int]:
    """Validate and return an HSV threshold triplet.

    Args:
        values (Sequence[int]): Three integer HSV values from the command line.

    Returns:
        tuple[int, int, int]: Validated HSV triplet.
    """
    if len(values) != 3:
        raise argparse.ArgumentTypeError("HSV thresholds must contain exactly three integers.")
    if any(value < 0 or value > 255 for value in values):
        raise argparse.ArgumentTypeError("HSV threshold values must be in the range [0, 255].")
    return int(values[0]), int(values[1]), int(values[2])


def collect_slide_paths(input_path: Path) -> list[Path]:
    """Collect OME-TIFF slide paths from a file or directory.

    Args:
        input_path (Path): Path to one tissue slide or a directory containing `*.ome.tiff` files.

    Returns:
        list[Path]: Sorted list of slide paths to analyze.
    """
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.ome.tiff"))
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def thumbnail_size(dimensions: tuple[int, int], max_side: int) -> tuple[int, int]:
    """Compute a thumbnail size that preserves aspect ratio.

    Args:
        dimensions (tuple[int, int]): Original `(width, height)` dimensions.
        max_side (int): Maximum allowed width or height for the thumbnail.

    Returns:
        tuple[int, int]: Thumbnail `(width, height)` size.
    """
    if max_side < 1:
        raise ValueError("max_side must be at least 1.")

    width, height = dimensions
    scale = min(max_side / width, max_side / height, 1.0)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def read_slide_thumbnail(slide_path: Path, max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Read a whole-slide image as an RGB thumbnail.

    Args:
        slide_path (Path): Path to the slide file.
        max_side (int): Maximum width or height of the returned thumbnail.

    Returns:
        tuple[np.ndarray, tuple[int, int]]: RGB thumbnail array and full-resolution `(width, height)`.
    """
    slide = openslide.OpenSlide(str(slide_path))
    try:
        full_dimensions = slide.dimensions
        preview = slide.get_thumbnail(thumbnail_size(full_dimensions, max_side)).convert("RGB")
    finally:
        slide.close()

    return np.asarray(preview), full_dimensions


def create_tissue_mask(
    rgb_image: np.ndarray,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
) -> np.ndarray:
    """Create a binary tissue mask from an RGB image with HSV thresholding.

    Args:
        rgb_image (np.ndarray): RGB image array with shape `(height, width, 3)`.
        hsv_lower (tuple[int, int, int]): Lower HSV threshold.
        hsv_upper (tuple[int, int, int]): Upper HSV threshold.

    Returns:
        np.ndarray: Boolean tissue mask with shape `(height, width)`.
    """
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv_image, hsv_lower, hsv_upper) > 0


def separate_he_signals(rgb_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate Hematoxylin and Eosin signals with HED color deconvolution.

    Args:
        rgb_image (np.ndarray): RGB image array with shape `(height, width, 3)`.

    Returns:
        tuple[np.ndarray, np.ndarray]: Hematoxylin and Eosin signal arrays.
    """
    hed_image = rgb2hed(rgb_image)
    hematoxylin = hed_image[:, :, 0]
    eosin = hed_image[:, :, 1]
    return hematoxylin, eosin


def sample_tissue_values(
    signal: np.ndarray,
    tissue_mask: np.ndarray,
    max_pixels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Sample stain-signal values from masked tissue pixels.

    Args:
        signal (np.ndarray): Stain signal array with shape `(height, width)`.
        tissue_mask (np.ndarray): Boolean tissue mask with shape `(height, width)`.
        max_pixels (int): Maximum number of tissue pixels to use.
        rng (np.random.Generator): Random number generator for reproducible sampling.

    Returns:
        tuple[np.ndarray, int]: Sampled signal values and the number of available tissue pixels.
    """
    if max_pixels < 2:
        raise ValueError("max_pixels must be at least 2 for distribution fitting.")

    tissue_indices = np.flatnonzero(tissue_mask)
    n_available = int(tissue_indices.size)
    if n_available < 2:
        raise ValueError("At least two tissue pixels are required for distribution fitting.")

    if n_available > max_pixels:
        tissue_indices = rng.choice(tissue_indices, size=max_pixels, replace=False)

    sampled_values = signal.reshape(-1)[tissue_indices]
    sampled_values = sampled_values[np.isfinite(sampled_values)]
    if sampled_values.size < 2:
        raise ValueError("At least two finite stain-signal values are required for distribution fitting.")

    return sampled_values, n_available


def estimate_distribution_mode(distribution: stats.rv_continuous, params: tuple[float, ...]) -> float:
    """Estimate the mode of a fitted positive distribution from its PDF.

    Args:
        distribution (stats.rv_continuous): SciPy continuous distribution object.
        params (tuple[float, ...]): Fitted distribution parameters including fixed `loc` and fitted `scale`.

    Returns:
        float: Estimated distribution mode.
    """
    upper = float(distribution.ppf(0.999, *params))
    if not np.isfinite(upper) or upper <= 0:
        upper = float(params[-1] * 10)

    lower = max(float(distribution.ppf(1e-6, *params)), np.finfo(float).tiny)
    if not np.isfinite(lower) or lower <= 0 or lower >= upper:
        lower = max(upper * 1e-6, np.finfo(float).tiny)

    x_values = np.geomspace(lower, upper, 3000)
    pdf_values = distribution.pdf(x_values, *params)
    if not np.any(np.isfinite(pdf_values)):
        return float("nan")
    return float(x_values[int(np.nanargmax(pdf_values))])


def fit_best_positive_distribution(
    values: np.ndarray,
    fit_percentile: float,
    candidate_names: Sequence[str],
) -> StainDistributionFit:
    """Fit candidate positive distributions and return the best AIC fit.

    Args:
        values (np.ndarray): One-dimensional sampled stain signal values from tissue pixels.
        fit_percentile (float): Upper positive-signal percentile retained for distribution fitting.
        candidate_names (Sequence[str]): Candidate distribution names from `POSITIVE_DISTRIBUTIONS`.

    Returns:
        StainDistributionFit: Best fitted positive-distribution parameters and descriptors.
    """
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 2:
        raise ValueError("At least two finite stain-signal values are required for distribution fitting.")

    positive_values = finite_values[finite_values > 0]
    positive_fraction = float(positive_values.size / finite_values.size)
    if positive_values.size < 2:
        raise ValueError("At least two strictly positive stain-signal values are required for distribution fitting.")
    if fit_percentile <= 0 or fit_percentile > 100:
        raise ValueError("fit_percentile must be in the range (0, 100].")

    if fit_percentile < 100:
        fit_upper = float(np.percentile(positive_values, fit_percentile))
        fit_values = positive_values[positive_values <= fit_upper]
    else:
        fit_values = positive_values
    if fit_values.size < 2:
        raise ValueError("At least two positive values are required after percentile filtering.")

    best_fit: StainDistributionFit | None = None
    fit_errors = []
    for candidate_name in candidate_names:
        if candidate_name not in POSITIVE_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution '{candidate_name}'. Available: {sorted(POSITIVE_DISTRIBUTIONS)}"
            )

        distribution = POSITIVE_DISTRIBUTIONS[candidate_name]
        try:
            params = distribution.fit(fit_values, floc=0)
            log_pdf = distribution.logpdf(fit_values, *params)
            finite_log_pdf = log_pdf[np.isfinite(log_pdf)]
            if finite_log_pdf.size == 0:
                raise ValueError("No finite log-likelihood values.")

            n_parameters = len(params) - 1
            log_likelihood = float(np.sum(finite_log_pdf))
            aic = 2 * n_parameters - 2 * log_likelihood
            mean, variance = distribution.stats(*params, moments="mv")
            quantiles = distribution.ppf([0.25, 0.5, 0.75, 0.95, 0.99], *params)
            std = math.sqrt(max(float(variance), 0.0))
            mode = estimate_distribution_mode(distribution, tuple(float(param) for param in params))

            descriptors = [float(mean), std, mode, *[float(quantile) for quantile in quantiles]]
            if not all(np.isfinite(value) for value in descriptors):
                raise ValueError("Distribution descriptors are not finite.")

            shape = float(params[0])
            second_shape = float(params[1]) if distribution.numargs > 1 else float("nan")
            candidate_fit = StainDistributionFit(
                distribution=candidate_name,
                positive_fraction=positive_fraction,
                shape=shape,
                second_shape=second_shape,
                scale=float(params[-1]),
                mean=float(mean),
                std=std,
                mode=mode,
                q25=float(quantiles[0]),
                median=float(quantiles[1]),
                q75=float(quantiles[2]),
                q95=float(quantiles[3]),
                q99=float(quantiles[4]),
                fit_percentile=fit_percentile,
                aic=float(aic),
            )
        except Exception as exc:
            fit_errors.append(f"{candidate_name}: {exc}")
            continue

        if best_fit is None or candidate_fit.aic < best_fit.aic:
            best_fit = candidate_fit

    if best_fit is None:
        raise RuntimeError("All candidate distribution fits failed: " + "; ".join(fit_errors))
    return best_fit


def fitted_distribution_pdf(x_values: np.ndarray, fit: StainDistributionFit) -> np.ndarray:
    """Evaluate a fitted positive-distribution probability curve.

    Args:
        x_values (np.ndarray): One-dimensional x-axis values.
        fit (StainDistributionFit): Fitted positive-distribution parameters.

    Returns:
        np.ndarray: Probability curve values for `x_values`.
    """
    distribution = POSITIVE_DISTRIBUTIONS[fit.distribution]
    if distribution.numargs > 1:
        return distribution.pdf(x_values, fit.shape, fit.second_shape, loc=0, scale=fit.scale)
    return distribution.pdf(x_values, fit.shape, loc=0, scale=fit.scale)


def robust_display_limits(signal: np.ndarray, tissue_mask: np.ndarray) -> tuple[float, float]:
    """Compute robust display limits for a stain signal.

    Args:
        signal (np.ndarray): Stain signal array with shape `(height, width)`.
        tissue_mask (np.ndarray): Boolean tissue mask with shape `(height, width)`.

    Returns:
        tuple[float, float]: Lower and upper display limits.
    """
    tissue_values = signal[tissue_mask & np.isfinite(signal)]
    if tissue_values.size == 0:
        return float(np.nanmin(signal)), float(np.nanmax(signal))

    lower, upper = np.percentile(tissue_values, [1, 99])
    if lower == upper:
        upper = lower + 1e-6
    return float(lower), float(upper)


def safe_output_stem(slide_path: Path) -> str:
    """Create a filesystem-safe output stem for one slide path.

    Args:
        slide_path (Path): Path to the analyzed tissue slide.

    Returns:
        str: Safe output stem containing the slide stem and a short path hash.
    """
    digest = hashlib.sha1(str(slide_path).encode("utf-8")).hexdigest()[:8]
    safe_stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in slide_path.stem)
    return f"{safe_stem}_{digest}"


def plot_signal_with_distribution(
    axis: plt.Axes,
    values: np.ndarray,
    fit: StainDistributionFit,
    title: str,
    color: str,
    bins: int,
    xmax_percentile: float,
) -> None:
    """Plot positive stain-signal values with a fitted distribution count overlay.

    Args:
        axis (plt.Axes): Matplotlib axis to draw on.
        values (np.ndarray): Stain signal values used for the fit.
        fit (StainDistributionFit): Fitted positive-distribution parameters.
        title (str): Plot title.
        color (str): Histogram and curve color.
        bins (int): Number of histogram bins.
        xmax_percentile (float): Upper percentile used to limit the displayed histogram x-axis.

    Returns:
        None: The function draws on `axis`.
    """
    positive_values = values[np.isfinite(values) & (values > 0)]
    if positive_values.size < 2:
        raise ValueError("At least two positive values are required to plot a fitted distribution.")
    if xmax_percentile <= 0 or xmax_percentile > 100:
        raise ValueError("xmax_percentile must be in the range (0, 100].")

    x_min = float(np.min(positive_values))
    x_max = float(np.percentile(positive_values, xmax_percentile))
    if x_max <= x_min:
        x_max = float(np.max(positive_values))

    display_values = positive_values[positive_values <= x_max]
    counts, bin_edges, _ = axis.hist(
        display_values,
        bins=bins,
        range=(x_min, x_max),
        alpha=0.35,
        color=color,
    )
    x_values = np.linspace(float(bin_edges[0]), float(bin_edges[-1]), 800)
    bin_width = float(np.mean(np.diff(bin_edges)))
    scaled_pdf = fitted_distribution_pdf(x_values, fit) * positive_values.size * bin_width
    axis.plot(x_values, scaled_pdf, color=color, linewidth=2)
    axis.axvline(fit.mean, color="black", linestyle="--", linewidth=1)
    axis.set_title(
        f"{title}\nmean={fit.mean:.4f}, std={fit.std:.4f}, "
        f"positive={fit.positive_fraction:.2%}, fit<=p{fit.fit_percentile:g}"
    )
    axis.set_xlabel("Separated stain signal")
    axis.set_ylabel("Pixel count")
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(bottom=0, top=max(float(np.max(counts)), float(np.max(scaled_pdf))) * 1.1)
    axis.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
    if xmax_percentile < 100:
        axis.text(
            0.98,
            0.95,
            f"x-axis <= p{xmax_percentile:g}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )


def create_thresholded_rgb_stain_image(
    rgb_image: np.ndarray,
    signal: np.ndarray,
    tissue_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Create an RGB image showing only pixels above a stain-signal threshold.

    Args:
        rgb_image (np.ndarray): RGB image array with shape `(height, width, 3)`.
        signal (np.ndarray): Separated stain signal array with shape `(height, width)`.
        tissue_mask (np.ndarray): Boolean tissue mask with shape `(height, width)`.
        threshold (float): Minimum stain signal value to keep in the RGB visualization.

    Returns:
        np.ndarray: RGB image where selected pixels keep their original color and all other pixels are white.
    """
    selected_pixels = tissue_mask & np.isfinite(signal) & (signal >= threshold)
    thresholded_rgb = np.full_like(rgb_image, fill_value=255)
    thresholded_rgb[selected_pixels] = rgb_image[selected_pixels]
    return thresholded_rgb


def save_visualization(
    slide_path: Path,
    rgb_image: np.ndarray,
    tissue_mask: np.ndarray,
    hematoxylin: np.ndarray,
    eosin: np.ndarray,
    hematoxylin_values: np.ndarray,
    eosin_values: np.ndarray,
    hematoxylin_fit: StainDistributionFit,
    eosin_fit: StainDistributionFit,
    output_dir: Path,
    histogram_bins: int,
    histogram_xmax_percentile: float,
    hematoxylin_rgb_threshold_stds: float,
    eosin_rgb_threshold_stds: float,
) -> Path:
    """Save a visualization of H&E separation and selected distribution fitting.

    Args:
        slide_path (Path): Path to the analyzed tissue slide.
        rgb_image (np.ndarray): RGB analysis image.
        tissue_mask (np.ndarray): Boolean tissue mask.
        hematoxylin (np.ndarray): Hematoxylin signal array.
        eosin (np.ndarray): Eosin signal array.
        hematoxylin_values (np.ndarray): Hematoxylin values used for distribution fitting.
        eosin_values (np.ndarray): Eosin values used for distribution fitting.
        hematoxylin_fit (StainDistributionFit): Fitted distribution parameters for Hematoxylin.
        eosin_fit (StainDistributionFit): Fitted distribution parameters for Eosin.
        output_dir (Path): Directory where the visualization PNG will be written.
        histogram_bins (int): Number of histogram bins.
        histogram_xmax_percentile (float): Upper percentile used to limit histogram x-axes.
        hematoxylin_rgb_threshold_stds (float): Number of fitted standard deviations above the
            Hematoxylin mean used for the RGB threshold visualization.
        eosin_rgb_threshold_stds (float): Number of fitted standard deviations above the Eosin mean
            used for the RGB threshold visualization.

    Returns:
        Path: Path to the saved visualization PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_output_stem(slide_path)}_he_best_distribution_fit.png"

    hematoxylin_threshold = hematoxylin_fit.mean + hematoxylin_rgb_threshold_stds * hematoxylin_fit.std
    eosin_threshold = eosin_fit.mean + eosin_rgb_threshold_stds * eosin_fit.std
    hematoxylin_rgb = create_thresholded_rgb_stain_image(
        rgb_image,
        hematoxylin,
        tissue_mask,
        hematoxylin_threshold,
    )
    eosin_rgb = create_thresholded_rgb_stain_image(
        rgb_image,
        eosin,
        tissue_mask,
        eosin_threshold,
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(hematoxylin_rgb)
    axes[0, 1].set_title(
        f"RGB Pixels: Hematoxylin >= Mean + {hematoxylin_rgb_threshold_stds:g} Std"
    )
    axes[0, 1].axis("off")

    axes[0, 2].imshow(eosin_rgb)
    axes[0, 2].set_title(f"RGB Pixels: Eosin >= Mean + {eosin_rgb_threshold_stds:g} Std")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(tissue_mask, cmap="gray")
    axes[1, 0].set_title("Tissue Mask")
    axes[1, 0].axis("off")

    plot_signal_with_distribution(
        axes[1, 1],
        hematoxylin_values,
        hematoxylin_fit,
        f"Hematoxylin {hematoxylin_fit.distribution} Fit",
        "purple",
        histogram_bins,
        histogram_xmax_percentile,
    )
    plot_signal_with_distribution(
        axes[1, 2],
        eosin_values,
        eosin_fit,
        f"Eosin {eosin_fit.distribution} Fit",
        "deeppink",
        histogram_bins,
        histogram_xmax_percentile,
    )

    fig.suptitle(str(slide_path), fontsize=11)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def analyze_slide(
    slide_path: Path,
    analysis_max_side: int,
    max_fit_pixels: int,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    output_dir: Path,
    histogram_bins: int,
    histogram_xmax_percentile: float,
    fit_percentile: float,
    candidate_distributions: Sequence[str],
    random_state: int,
    hematoxylin_rgb_threshold_stds: float,
    eosin_rgb_threshold_stds: float,
) -> SlideStainStatistics:
    """Analyze one tissue slide and return fitted H&E positive-distribution statistics.

    Args:
        slide_path (Path): Path to one tissue slide.
        analysis_max_side (int): Maximum width or height used for analysis and visualization.
        max_fit_pixels (int): Maximum number of tissue pixels used per distribution fit.
        hsv_lower (tuple[int, int, int]): Lower HSV threshold for tissue masking.
        hsv_upper (tuple[int, int, int]): Upper HSV threshold for tissue masking.
        output_dir (Path): Directory where visualization PNGs will be written.
        histogram_bins (int): Number of histogram bins.
        histogram_xmax_percentile (float): Upper percentile used to limit histogram x-axes.
        fit_percentile (float): Upper positive-signal percentile retained for distribution fitting.
        candidate_distributions (Sequence[str]): Candidate distribution names for AIC-based selection.
        random_state (int): Random seed for reproducible sampling.
        hematoxylin_rgb_threshold_stds (float): Number of fitted standard deviations above the
            Hematoxylin mean used for the RGB threshold visualization.
        eosin_rgb_threshold_stds (float): Number of fitted standard deviations above the Eosin mean
            used for the RGB threshold visualization.

    Returns:
        SlideStainStatistics: Fitted H&E positive-distribution statistics for the slide.
    """
    rgb_image, full_dimensions = read_slide_thumbnail(slide_path, analysis_max_side)
    tissue_mask = create_tissue_mask(rgb_image, hsv_lower, hsv_upper)
    hematoxylin, eosin = separate_he_signals(rgb_image)

    rng = np.random.default_rng(random_state)
    hematoxylin_values, n_available = sample_tissue_values(
        hematoxylin,
        tissue_mask,
        max_fit_pixels,
        rng,
    )
    eosin_values, _ = sample_tissue_values(eosin, tissue_mask, max_fit_pixels, rng)

    hematoxylin_fit = fit_best_positive_distribution(
        hematoxylin_values,
        fit_percentile,
        candidate_distributions,
    )
    eosin_fit = fit_best_positive_distribution(
        eosin_values,
        fit_percentile,
        candidate_distributions,
    )
    visualization_path = save_visualization(
        slide_path=slide_path,
        rgb_image=rgb_image,
        tissue_mask=tissue_mask,
        hematoxylin=hematoxylin,
        eosin=eosin,
        hematoxylin_values=hematoxylin_values,
        eosin_values=eosin_values,
        hematoxylin_fit=hematoxylin_fit,
        eosin_fit=eosin_fit,
        output_dir=output_dir,
        histogram_bins=histogram_bins,
        histogram_xmax_percentile=histogram_xmax_percentile,
        hematoxylin_rgb_threshold_stds=hematoxylin_rgb_threshold_stds,
        eosin_rgb_threshold_stds=eosin_rgb_threshold_stds,
    )

    full_width, full_height = full_dimensions
    analysis_height, analysis_width = rgb_image.shape[:2]
    return SlideStainStatistics(
        slide_path=slide_path,
        slide_width=full_width,
        slide_height=full_height,
        analysis_width=analysis_width,
        analysis_height=analysis_height,
        n_tissue_pixels_available=n_available,
        n_tissue_pixels_used=int(hematoxylin_values.size),
        hematoxylin=hematoxylin_fit,
        eosin=eosin_fit,
        visualization_path=visualization_path,
    )


def write_statistics_csv(statistics: Sequence[SlideStainStatistics], output_csv: Path) -> Path:
    """Write fitted positive-distribution stain statistics to a CSV file.

    Args:
        statistics (Sequence[SlideStainStatistics]): Per-slide fitted stain statistics.
        output_csv (Path): Path where the CSV file will be written.

    Returns:
        Path: Path to the written CSV file.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slide_path",
        "slide_width",
        "slide_height",
        "analysis_width",
        "analysis_height",
        "n_tissue_pixels_available",
        "n_tissue_pixels_used",
        "hematoxylin_distribution",
        "hematoxylin_positive_fraction",
        "hematoxylin_zero_fraction",
        "hematoxylin_shape",
        "hematoxylin_second_shape",
        "hematoxylin_scale",
        "hematoxylin_mean",
        "hematoxylin_std",
        "hematoxylin_mode",
        "hematoxylin_q25",
        "hematoxylin_median",
        "hematoxylin_q75",
        "hematoxylin_q95",
        "hematoxylin_q99",
        "hematoxylin_fit_percentile",
        "hematoxylin_aic",
        "eosin_distribution",
        "eosin_positive_fraction",
        "eosin_zero_fraction",
        "eosin_shape",
        "eosin_second_shape",
        "eosin_scale",
        "eosin_mean",
        "eosin_std",
        "eosin_mode",
        "eosin_q25",
        "eosin_median",
        "eosin_q75",
        "eosin_q95",
        "eosin_q99",
        "eosin_fit_percentile",
        "eosin_aic",
        "visualization_path",
    ]

    with output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for stat in statistics:
            writer.writerow(
                {
                    "slide_path": str(stat.slide_path),
                    "slide_width": stat.slide_width,
                    "slide_height": stat.slide_height,
                    "analysis_width": stat.analysis_width,
                    "analysis_height": stat.analysis_height,
                    "n_tissue_pixels_available": stat.n_tissue_pixels_available,
                    "n_tissue_pixels_used": stat.n_tissue_pixels_used,
                    "hematoxylin_distribution": stat.hematoxylin.distribution,
                    "hematoxylin_positive_fraction": stat.hematoxylin.positive_fraction,
                    "hematoxylin_zero_fraction": 1.0 - stat.hematoxylin.positive_fraction,
                    "hematoxylin_shape": stat.hematoxylin.shape,
                    "hematoxylin_second_shape": stat.hematoxylin.second_shape,
                    "hematoxylin_scale": stat.hematoxylin.scale,
                    "hematoxylin_mean": stat.hematoxylin.mean,
                    "hematoxylin_std": stat.hematoxylin.std,
                    "hematoxylin_mode": stat.hematoxylin.mode,
                    "hematoxylin_q25": stat.hematoxylin.q25,
                    "hematoxylin_median": stat.hematoxylin.median,
                    "hematoxylin_q75": stat.hematoxylin.q75,
                    "hematoxylin_q95": stat.hematoxylin.q95,
                    "hematoxylin_q99": stat.hematoxylin.q99,
                    "hematoxylin_fit_percentile": stat.hematoxylin.fit_percentile,
                    "hematoxylin_aic": stat.hematoxylin.aic,
                    "eosin_distribution": stat.eosin.distribution,
                    "eosin_positive_fraction": stat.eosin.positive_fraction,
                    "eosin_zero_fraction": 1.0 - stat.eosin.positive_fraction,
                    "eosin_shape": stat.eosin.shape,
                    "eosin_second_shape": stat.eosin.second_shape,
                    "eosin_scale": stat.eosin.scale,
                    "eosin_mean": stat.eosin.mean,
                    "eosin_std": stat.eosin.std,
                    "eosin_mode": stat.eosin.mode,
                    "eosin_q25": stat.eosin.q25,
                    "eosin_median": stat.eosin.median,
                    "eosin_q75": stat.eosin.q75,
                    "eosin_q95": stat.eosin.q95,
                    "eosin_q99": stat.eosin.q99,
                    "eosin_fit_percentile": stat.eosin.fit_percentile,
                    "eosin_aic": stat.eosin.aic,
                    "visualization_path": str(stat.visualization_path),
                }
            )

    return output_csv


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for H&E stain statistics analysis.

    Args:
        None.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Separate Hematoxylin and Eosin signals from tissue slides, select the best positive "
            "distribution for each stain signal, and save statistics plus visualizations."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=DEFAULT_EXAMPLE_SLIDE,
        help="Path to one .ome.tiff tissue slide or a directory containing .ome.tiff files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV path for fitted positive-distribution statistics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-slide visualization PNGs.",
    )
    parser.add_argument(
        "--analysis-max-side",
        type=int,
        default=5120,
        help="Maximum width or height used for analysis and visualization.",
    )
    parser.add_argument(
        "--max-fit-pixels",
        type=int,
        default=500_000,
        help="Maximum number of tissue pixels sampled for each distribution fit.",
    )
    parser.add_argument(
        "--hsv-lower",
        nargs=3,
        type=int,
        default=(50, 50, 50),
        metavar=("H", "S", "V"),
        help="Lower HSV threshold for tissue masking.",
    )
    parser.add_argument(
        "--hsv-upper",
        nargs=3,
        type=int,
        default=(180, 255, 220),
        metavar=("H", "S", "V"),
        help="Upper HSV threshold for tissue masking.",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=400,
        help="Number of bins for selected-distribution fit histograms.",
    )
    parser.add_argument(
        "--histogram-xmax-percentile",
        type=float,
        default=99.5,
        help=(
            "Upper positive-signal percentile used for histogram x-axis display. "
            "Use --fit-percentile to control the fitted signal range."
        ),
    )
    parser.add_argument(
        "--fit-percentile",
        type=float,
        default=100.0,
        help=(
            "Upper positive-signal percentile retained for distribution fitting. "
            "The default keeps the full positive signal."
        ),
    )
    parser.add_argument(
        "--candidate-distributions",
        nargs="+",
        default=list(DEFAULT_DISTRIBUTION_CANDIDATES),
        help=(
            "Positive distributions to compare by AIC. Available: "
            f"{', '.join(sorted(POSITIVE_DISTRIBUTIONS))}."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--hematoxylin-rgb-threshold-stds",
        type=float,
        default=1.0,
        help=(
            "Number of fitted Hematoxylin standard deviations above the mean used for the "
            "RGB Hematoxylin panel. Higher values are more selective for nuclei."
        ),
    )
    parser.add_argument(
        "--eosin-rgb-threshold-stds",
        type=float,
        default=0.0,
        help=(
            "Number of fitted Eosin standard deviations above the mean used for the RGB Eosin panel."
        ),
    )
    return parser


def main() -> None:
    """Run H&E stain statistics analysis from the command line.

    Args:
        None.

    Returns:
        None: Results are written to disk and progress is printed to standard output.
    """
    parser = build_parser()
    args = parser.parse_args()

    hsv_lower = parse_hsv_triplet(args.hsv_lower)
    hsv_upper = parse_hsv_triplet(args.hsv_upper)
    slide_paths = collect_slide_paths(args.input_path)
    if not slide_paths:
        raise FileNotFoundError(f"No .ome.tiff files found in: {args.input_path}")

    statistics = []
    for slide_path in slide_paths:
        print(f"Analyzing {slide_path}")
        statistics.append(
            analyze_slide(
                slide_path=slide_path,
                analysis_max_side=args.analysis_max_side,
                max_fit_pixels=args.max_fit_pixels,
                hsv_lower=hsv_lower,
                hsv_upper=hsv_upper,
                output_dir=args.output_dir,
                histogram_bins=args.histogram_bins,
                histogram_xmax_percentile=args.histogram_xmax_percentile,
                fit_percentile=args.fit_percentile,
                candidate_distributions=args.candidate_distributions,
                random_state=args.random_state,
                hematoxylin_rgb_threshold_stds=args.hematoxylin_rgb_threshold_stds,
                eosin_rgb_threshold_stds=args.eosin_rgb_threshold_stds,
            )
        )

    output_csv = write_statistics_csv(statistics, args.output_csv)
    print(f"Wrote fitted H&E stain statistics to {output_csv}")
    print(f"Wrote visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
