import openslide
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
import gc
from pathlib import Path
from typing import Optional
import pandas as pd
from PIL import Image


def crop_tissues(
    slide_path: str,
    level: int = -1,
    min_area_ratio: float = 0.01,
    min_hole_ratio: float = 0.005,
    hsv_lower: tuple = (50, 50, 50),
    hsv_upper: tuple = (180, 255, 220),
    tile_size: tuple = (256, 256),
    pyramidal_levels: int = 6,
    pyramidal_downsample: int = 2,
    results_dir: str = ".",
    show_steps: bool = False,
    show_results: bool = False):
    """
    Given a whole-slide image path, segment HE tissues from the white background and crop each large tissue region.

    Args:
        slide_path (str): Path to the whole-slide image file.
        level (int): Pyramid level to process default -1 (lowest resolution).
        min_area_ratio (float): Minimum area ratio (relative to image size) to keep a tissue region default 0.01.
        min_hole_ratio (float): Minimum size ratio (relative to image size) to keep a whole tissue region default 0.005.
        hsv_lower (tuple): Lower bound for HSV thresholding default (50, 50, 50) for H&E on white background.
        hsv_upper (tuple): Upper bound for HSV thresholding default (180, 255, 220) for H&E on white background.
        tile_size (tuple): Tile size for saving OME-TIFF files default (256, 256).
        pyramidal_levels (int): Number of pyramid levels for OME-TIFF saving default 6.
        pyramidal_downsample (int): Downsampling factor for pyramid levels default 2.
        results_dir (str): Directory path to save the cropped tissue OME-TIFF files.
        show_steps (bool): If True, show intermediate segmentation steps default False.
        show_results (bool): If True, show cropped tissue regions default False.
        
    Returns:
        cropped_tissues (list of np.ndarray): List of cropped tissue image arrays.
    """
    # Extract slide name (without extension) to create a directory for this slide
    slide_name = Path(slide_path).stem
    
    # Create a directory for this slide within results_dir
    slide_output_dir = Path(results_dir) / slide_name
    slide_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load slide and get image array
    slide = openslide.OpenSlide(slide_path)

    # Prepare metadata for OME-TIFF
    metadata = {
        "PhysicalSizeX": slide.properties.get("openslide.mpp-x"),
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": slide.properties.get("openslide.mpp-y"),
        "PhysicalSizeYUnit": "µm",
        "ObjectivePower": slide.properties.get("openslide.objective-power")
    }

    # If level is -1, use the lowest resolution level
    if level == -1:
        level = slide.level_count - 1

    # Get an image array from the slide at the specified level
    img = slide.read_region(location=(0, 0),
                            level=level,
                            size=slide.level_dimensions[level])

    # Convert the image to rgb array
    img_array = np.array(img.convert("RGB"))

    # Calculate image area
    img_area = img_array.shape[0] * img_array.shape[1]

    # Compute downsample factor
    downsample = slide.level_downsamples[level]

    # Convert RGB to HSV for color segmentation
    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Threshold to segment tissue (non-white)
    mask = cv2.inRange(hsv_img, hsv_lower, hsv_upper)

    # Fill holes in the mask
    min_hole_length = max(1, int(img_array.shape[0]*min_hole_ratio))  # Ensure at least size 1
    min_hole_size = (min_hole_length, min_hole_length)
    mask_filled = cv2.morphologyEx(src=mask, 
                                    op=cv2.MORPH_CLOSE,
                                    kernel=cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,
                                                                     ksize=min_hole_size))

    # Remove small artifacts using connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
    min_area = img_area * min_area_ratio
    mask_cleaned = np.zeros_like(mask_filled)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask_cleaned[labels == i] = 255

    # Find contours of large tissue regions
    mask_bin = (mask_cleaned > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Plot the contours on white background
    contour_img = np.ones_like(img_array) * 255
    rectangles_img = img_array.copy()

    # Loop through contours
    cropped_tissues = []
    for cnt in contours:
        # Draw contour
        cv2.drawContours(image=contour_img,
                        contours=[cnt],
                        contourIdx=-1,
                        color=np.random.randint(0, 128, size=3).tolist(),
                        thickness= 1+contour_img.shape[0]//1000)
        
        # Draw bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(image=rectangles_img,
                        contours=[np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])],
                        contourIdx=-1,
                        color=(0, 0, 0),
                        thickness= 1+contour_img.shape[0]//1000)
        
        # Convert coordinates to full resolution
        x_full, y_full, w_full, h_full = int(x*downsample), int(y*downsample), int(w*downsample), int(h*downsample)

        # Read the full resolution crop from the slide
        full_crop = slide.read_region(location=(x_full, y_full),
                                      level=0,
                                      size=(w_full, h_full))
        
        # Convert to RGB
        full_crop = full_crop.convert("RGB")

        # Convert to numpy array
        full_crop = np.array(full_crop)

        # Create pyramidal levels
        pyramid = [full_crop]
        for i in range(1, pyramidal_levels):
            prev = pyramid[-1]
            # Downsample by 2 each time
            down = prev[::pyramidal_downsample, ::pyramidal_downsample, :]
            pyramid.append(down)
        
        # Save as OME-TIFF in the slide-specific directory
        output_path = slide_output_dir / f"{x_full}_{y_full}.ome.tiff"
        print(output_path)
        with tifffile.TiffWriter(str(output_path), bigtiff=True, ome=True) as tif:
            tif.write(data=full_crop, subifds=len(pyramid) - 1, tile=tile_size, compression='JPEG', metadata=metadata)
            for level in pyramid[1:]:
                tif.write(level,tile=tile_size, compression='JPEG', subfiletype=1)
        
        # Crop and save each tissue region
        cropped_tissues.append(full_crop[::10, ::10, :])

        # clean up memory
        del full_crop
        gc.collect()

    if show_steps:
        fig, ax = plt.subplots(1, 7, figsize=(20, 6))
        ax[0].imshow(img_array)
        ax[0].set_title('Original Image')
        ax[1].imshow(hsv_img)
        ax[1].set_title('HSV Image')
        ax[2].imshow(mask, cmap='gray')
        ax[2].set_title('Tissue Mask')
        ax[3].imshow(mask_filled, cmap='gray')
        ax[3].set_title('Filled Tissue Mask')
        ax[4].imshow(mask_cleaned, cmap='gray')
        ax[4].set_title('Cleaned Tissue Mask')
        ax[5].imshow(contour_img)
        ax[5].set_title('Detected Tissue Contours')
        ax[6].imshow(rectangles_img)
        ax[6].set_title('Bounding Rectangles')

        for a in ax:
            a.axis('off')
        plt.show()

    if show_results:
        n_tissues = len(cropped_tissues)
        if n_tissues > 0:
            fig, ax = plt.subplots(1, n_tissues, figsize=(4*n_tissues, 4))
            
            # If only one tissue, wrap ax in a list so we can iterate
            if n_tissues == 1:
                ax = [ax]
                
            for i, crop in enumerate(cropped_tissues):
                ax[i].imshow(crop)
                ax[i].set_title(f'Tissue {i+1}')
                ax[i].axis('off')
            plt.show()

    return cropped_tissues


def create_tiles(
    slide_path: str,
    output_csv_path: str | Path,
    tile_size: tuple[int, int] = (448, 448),
    hsv_lower: tuple[int, int, int] = (0, 10, 0),
    hsv_upper: tuple[int, int, int] = (180, 255, 220),
    quantile: float = 0.5,
) -> tuple[pd.DataFrame, Path]:
    """
    Create tile center coordinates from a whole-slide image and save them as a CSV file.

    Args:
        slide_path (str): Path to the input whole-slide image.
        output_csv_path (str | Path): Path where the tile coordinates CSV will be written.
        tile_size (tuple[int, int]): Tile size `(width, height)` used for center-grid sampling.
        hsv_lower (tuple[int, int, int]): Lower HSV threshold used for tissue masking.
        hsv_upper (tuple[int, int, int]): Upper HSV threshold used for tissue masking.
        quantile (float): Quantile value used for thresholding the tissue mask default 0.5 (median).
    Returns:
        tuple[pd.DataFrame, Path]: A tuple containing the generated tiles DataFrame
        with columns `id`, `x`, and `y`, and the saved CSV path.
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
        tiles_csv_path (str): Path to the CSV file containing level-0 coordinates in `x` and `y` columns.
        slide_path (Optional[str]): Optional explicit slide path. If `None`, infer `<stem>.ome.tiff` from `*_tiles.csv`.
        preview_max_side (int): Maximum width/height in pixels for the compressed preview image.
        point_size (float): Marker size for plotted coordinates.
        point_color (str): Marker color for plotted coordinates.
        point_alpha (float): Marker opacity in range `[0.0, 1.0]`.
        save (bool): If `True`, save overlay as PNG. If `False`, only display the figure.
        output_png_path (Optional[str]): Optional output PNG path when `save=True`. If `None`, auto-generate next to CSV.
        png_compression_level (int): PNG compression level in range `[0, 9]` when saving.

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
    plt.show()

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