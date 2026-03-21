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
import os
import pandas as pd
from utils import create_tiles, visualize_tile_coordinates
import gc
from pathlib import Path

# %% [markdown]
# # Create tile coordinates (if missing)

# %%
TILE_SIZE = (448, 448)
Source_Directory = Path("/workspaces/WSI-Classification/data/HE-MYO/Processed/Myopathic")

for slide_dir in sorted(Source_Directory.iterdir()):
    if not slide_dir.is_dir():
        continue

    slide_paths = sorted(slide_dir.glob("*.ome.tiff"))
    if not slide_paths:
        continue

    print(f"Processing directory: {slide_dir.name}")
    for slide_path in slide_paths:
        slide_base_name = slide_path.name.removesuffix(".ome.tiff")
        tiles_csv_path = slide_path.with_name(f"{slide_base_name}_tiles.csv")

        if tiles_csv_path.exists():
            print(f"  - Coordinates already exist: {tiles_csv_path.name}")
        else:
            print(f"  - Creating coordinates: {tiles_csv_path.name}")
            create_tiles(
                slide_path=str(slide_path),
                output_csv_path=tiles_csv_path,
                tile_size=TILE_SIZE,
                quantile=0.75
            )
            print(f"  - Visualizing newly created coordinates: {tiles_csv_path.name}")
            visualize_tile_coordinates(
                tiles_csv_path=str(tiles_csv_path),
                slide_path=str(slide_path),
                save=False,
            )
            gc.collect()

print("Terminated")

# %% [markdown]
# # Visualize tile coordinates

# %%
Source_Directory = Path("/workspaces/WSI-Classification/data/HE-MYO/Processed/Neurogenic")

for slide_dir in sorted(Source_Directory.iterdir()):
    if not slide_dir.is_dir():
        continue

    tiles_csv_files = sorted(slide_dir.glob("*_tiles.csv"))
    if not tiles_csv_files:
        continue

    print(f"Processing directory: {slide_dir.name}")
    for tiles_csv_path in tiles_csv_files:
        slide_base_name = tiles_csv_path.name.removesuffix("_tiles.csv")
        inferred_slide_path = tiles_csv_path.with_name(f"{slide_base_name}.ome.tiff")
        print(f"  - Visualizing: {tiles_csv_path.name}")
        visualize_tile_coordinates(
            tiles_csv_path=str(tiles_csv_path),
            slide_path=str(inferred_slide_path),
            save=False,
        )
    gc.collect()

print("Terminated")
