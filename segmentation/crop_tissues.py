# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import cv2
import tifffile
import openslide
import gc

import matplotlib.pyplot as plt

# %% [markdown]
# # Hyperparameters

# %%
slide_path = '/workspaces/WSI-Classification/data/HE-MYO/Raw/DYSTROPHIC/Rome_1216-186-93_DGK_CAPN3 - 2024-12-18 15.22.04.ndpi'
path_results = '/workspaces/WSI-Classification/data/HE-MYO/Processed/Dystrophic/'

# %%
level=8  # The level to read the image at
min_area_ratio=0.01  # The minimum area ratio of the tissue to the whole slide
min_whole_ratio=0.005  # The minimum ratio of the whole slide to the tissue

LEVELS = 6  # Number of pyramid levels
DOWNSAMPLE = 2  # Downsampling factor
TILE_SIZE = (256, 256)  # Tile size for compression
LOWER_THRESHOLD = (0, 10, 0)  # Lower threshold for HSV color segmentation
UPPER_THRESHOLD = (180, 255, 220)  # Upper threshold for HSV color segmentation

# %% [markdown]
# # Load the whole-slide image

# %%
# Load the whole-slide image
slide = openslide.OpenSlide(slide_path)

#  Slide dimensions
print("Dimensions: ", slide.dimensions)
print("Level count: ", slide.level_count)
print("Level dimensions: ", slide.level_dimensions)

# Get thumbnail at lowest resolution
slide.get_thumbnail(size=slide.level_dimensions[-1])

# %% [markdown]
# # Read the image at the specified level as an RGB image array

# %%
# get an image array from the slide at the specified level
img = slide.read_region(location=(0, 0),
                        level=level,
                        size=slide.level_dimensions[level])

# Convert the image to rgb array
img_array = np.array(img.convert("RGB"))

# Calculate image area
img_area = img_array.shape[0] * img_array.shape[1]

# %% [markdown]
# # Extract some metadata

# %%
# Extract downsample factor
downsample = slide.level_downsamples[level]

# Define metadata for OME-TIFF
metadata = {
        "PhysicalSizeX": slide.properties.get("openslide.mpp-x"),
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": slide.properties.get("openslide.mpp-y"),
        "PhysicalSizeYUnit": "µm",
        "ObjectivePower": slide.properties.get("openslide.objective-power")
    }
metadata

# %% [markdown]
# # Peform tissue segmentation then cropp the tissues

# %%
# Convert RGB to HSV for color segmentation
hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

# Threshold to segment tissue (non-white)
mask = cv2.inRange(hsv_img, LOWER_THRESHOLD, UPPER_THRESHOLD)

# Fill holes in the mask
min_whole_length = int(img_array.shape[0]*min_whole_ratio)
min_whole_size = (min_whole_length, min_whole_length)
mask_filled = cv2.morphologyEx(src=mask, 
                               op=cv2.MORPH_CLOSE,
                               kernel=cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,
                                                                ksize=min_whole_size))

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

    # Convert to numpy array and downsample for visualization
    full_crop = np.array(full_crop)

    pyramid = [full_crop]
    for i in range(1, LEVELS):
        prev = pyramid[-1]
        # Downsample by 2 each time
        down = prev[::DOWNSAMPLE, ::DOWNSAMPLE, :]
        pyramid.append(down)
    
    # Save as OME-TIFF
    with tifffile.TiffWriter(f"{path_results}tissue_crop_{x_full}_{y_full}.ome.tiff", bigtiff=True, ome=True) as tif:
        tif.write(data=full_crop, subifds=len(pyramid) - 1, tile=TILE_SIZE, compression='JPEG', metadata=metadata)
        for level in pyramid[1:]:
            tif.write(level,tile=TILE_SIZE, compression='JPEG', subfiletype=1)
    
    # Crop and save each tissue region
    cropped_tissues.append(full_crop[::10, ::10, :])

    # Clean up memory
    del full_crop
    gc.collect()

# %% [markdown]
# # Visualize the cropped tissues

# %%
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

fig, ax = plt.subplots(1, len(cropped_tissues), figsize=(4*len(cropped_tissues), 4))
for i, crop in enumerate(cropped_tissues):
    ax[i].imshow(crop)
    ax[i].set_title(f'Tissue {i+1}')
    ax[i].axis('off')
plt.show()
