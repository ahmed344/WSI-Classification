from pathlib import Path
from typing import Optional
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from PIL import Image


def collect_slides(processed_root: str | Path) -> list[dict[str, str]]:
    """
    Collect slide metadata by scanning category and slide subdirectories for `.ome.tiff` files.

    Args:
        processed_root (str | Path): Root directory containing category folders and slide subfolders.

    Returns:
        list[dict[str, str]]: List of slide metadata dictionaries with `name`, `path`, `category`, and `slide_dir`.
    """
    slides: list[dict[str, str]] = []
    processed_path = Path(processed_root)

    for category_dir in os.listdir(processed_path):
        category_path = processed_path / category_dir
        if category_path.is_dir():
            for slide_dir in os.listdir(category_path):
                slide_dir_path = category_path / slide_dir
                if slide_dir_path.is_dir():
                    for slide_file in slide_dir_path.glob("*.ome.tiff"):
                        slides.append(
                            {
                                "name": slide_file.name,
                                "path": str(slide_file),
                                "category": category_dir,
                                "slide_dir": slide_dir,
                            }
                        )

    return slides


def create_tiles(
    slide_path: str,
    output_csv_path: str | Path,
    tile_size: tuple[int, int] = (448, 448),
    hsv_lower: tuple[int, int, int] = (0, 10, 0),
    hsv_upper: tuple[int, int, int] = (180, 255, 220),
    quantile: float = 0.75,
) -> tuple[pd.DataFrame, Path]:
    """
    Create tile center coordinates from a whole-slide image and save them as a CSV file.

    Args:
        slide_path (str): Path to the input whole-slide image.
        output_csv_path (str | Path): Path where the tile coordinates CSV is written.
        tile_size (tuple[int, int]): Tile size `(width, height)` used for center-grid sampling.
        hsv_lower (tuple[int, int, int]): Lower HSV threshold used for tissue masking.
        hsv_upper (tuple[int, int, int]): Upper HSV threshold used for tissue masking.
        quantile (float): Quantile value used for thresholding the tissue mask.

    Returns:
        tuple[pd.DataFrame, Path]: Generated tile coordinates DataFrame and saved CSV path.
    """
    slide = openslide.OpenSlide(slide_path)
    try:
        img = slide.read_region((0, 0), 0, slide.dimensions).convert("RGB")
        hsv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_img, hsv_lower, hsv_upper)

        delta = (tile_size[0] // 2, tile_size[1] // 2)
        slide_width, slide_height = slide.dimensions
        x_coords, y_coords = [], []

        for y in range(delta[1], slide_height, tile_size[1]):
            for x in range(delta[0], slide_width, tile_size[0]):
                if np.quantile(mask[y - delta[1]:y + delta[1], x - delta[0]:x + delta[0]], quantile) > 0:
                    x_coords.append(x)
                    y_coords.append(y)
    finally:
        slide.close()

    tiles = pd.DataFrame({"id": range(len(x_coords)), "x": x_coords, "y": y_coords})
    csv_path = Path(output_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tiles.to_csv(csv_path, index=False)
    return tiles, csv_path


def visualize_tile_coordinates(
    tiles_csv_path: str,
    slide_path: Optional[str] = None,
    preview_max_side: int = 2500,
    point_size: float = 2.0,
    point_color: str = "black",
    point_alpha: float = 0.65,
    save: bool = False,
    output_png_path: Optional[str] = None,
    png_compression_level: int = 6,
) -> Optional[Path]:
    """
    Display tile coordinates from a `*_tiles.csv` file on a compressed slide preview, with optional PNG export.

    Args:
        tiles_csv_path (str): Path to CSV containing level-0 `x` and `y` coordinates.
        slide_path (Optional[str]): Optional explicit slide path; if `None`, infer from CSV stem.
        preview_max_side (int): Maximum width/height in pixels for the preview image.
        point_size (float): Marker size for plotted coordinates.
        point_color (str): Marker color for plotted coordinates.
        point_alpha (float): Marker opacity in range `[0.0, 1.0]`.
        save (bool): If `True`, save the overlay as PNG.
        output_png_path (Optional[str]): Optional output PNG path used when `save=True`.
        png_compression_level (int): PNG compression level in range `[0, 9]`.

    Returns:
        Optional[Path]: Saved PNG path when `save=True`, otherwise `None`.
    """
    csv_path = Path(tiles_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Tiles CSV not found: {csv_path}")

    tiles = pd.read_csv(csv_path)
    if "x" not in tiles.columns or "y" not in tiles.columns:
        raise ValueError("Tiles CSV must contain 'x' and 'y' columns.")

    if slide_path is None:
        if not csv_path.stem.endswith("_tiles"):
            raise ValueError("Cannot infer slide path because CSV name does not end with '_tiles.csv'.")
        inferred_slide_path = csv_path.parent / f"{csv_path.stem[:-len('_tiles')]}.ome.tiff"
    else:
        inferred_slide_path = Path(slide_path)

    if not inferred_slide_path.exists():
        raise FileNotFoundError(f"Slide not found: {inferred_slide_path}")

    slide = openslide.OpenSlide(str(inferred_slide_path))
    try:
        slide_width, slide_height = slide.dimensions
        scale = min(preview_max_side / slide_width, preview_max_side / slide_height, 1.0)
        preview_size = (max(1, int(slide_width * scale)), max(1, int(slide_height * scale)))
        preview_rgb = np.array(slide.get_thumbnail(preview_size).convert("RGB"))
    finally:
        slide.close()

    preview_h, preview_w = preview_rgb.shape[:2]
    scale_x = preview_w / slide_width
    scale_y = preview_h / slide_height
    x_preview = tiles["x"].astype(float) * scale_x
    y_preview = tiles["y"].astype(float) * scale_y

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(preview_rgb)
    ax.scatter(x_preview, y_preview, s=point_size, c=point_color, alpha=point_alpha)
    ax.axis("off")

    if not save:
        plt.close(fig)
        return None

    if png_compression_level < 0 or png_compression_level > 9:
        raise ValueError("png_compression_level must be in [0, 9].")

    if output_png_path is None:
        stem = csv_path.stem[:-len("_tiles")] if csv_path.stem.endswith("_tiles") else csv_path.stem
        output_path = csv_path.parent / f"{stem}_tiles_overlay.png"
    else:
        output_path = Path(output_png_path)
        if output_path.suffix.lower() != ".png":
            raise ValueError("output_png_path must end with .png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    rendered = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]
    Image.fromarray(rendered).save(output_path, format="PNG", compress_level=png_compression_level)
    plt.close(fig)
    return output_path


def ensure_tiles_and_qc(
    slide_path: str,
    output_dir: str | Path,
    slide_base_name: str,
    tile_size: tuple[int, int],
    qc_dir: str | Path,
    quantile: float = 0.75,
) -> Path:
    """
    Ensure tiles CSV and QC overlay PNG exist for a slide without overwriting existing files.

    Args:
        slide_path (str): Path to the slide `.ome.tiff`.
        output_dir (str | Path): Slide output directory where tiles CSV should be stored.
        slide_base_name (str): Base slide name used for output file naming.
        tile_size (tuple[int, int]): Tile size passed to tile creation.
        qc_dir (str | Path): Directory where QC overlay PNG files are stored.
        quantile (float): Quantile threshold used when creating missing tiles.

    Returns:
        Path: Path to the slide tiles CSV file.
    """
    output_path = Path(output_dir)
    tiles_path = output_path / f"{slide_base_name}_tiles.csv"
    if not tiles_path.exists():
        print(f"Creating tiles for {slide_base_name}")
        create_tiles(
            slide_path=slide_path,
            output_csv_path=tiles_path,
            tile_size=tile_size,
            quantile=quantile,
        )

    qc_path = Path(qc_dir)
    qc_path.mkdir(parents=True, exist_ok=True)
    qc_png_path = qc_path / f"{slide_base_name}_tiles_overlay.png"
    if not qc_png_path.exists():
        visualize_tile_coordinates(
            tiles_csv_path=str(tiles_path),
            slide_path=slide_path,
            save=True,
            output_png_path=str(qc_png_path),
        )

    return tiles_path
