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
from utils import crop_tissues

# %% [markdown]
# # Crop to directories according to a metadata table

# %%
# Assign the path
path = '/workspaces/WSI-Classification/data/HE-MYO/Raw/DYSTROPHIC/'
path_results = '/workspaces/WSI-Classification/data/HE-MYO/Processed'

# List the slides
slides = [slide for slide in os.listdir(path) if slide.startswith('P')]
slides_ids = [slide[:7] for slide in slides]

# Load the metadata
metadata_df = pd.read_csv('/workspaces/WSI-Classification/data/HE-MYO/Raw/Scanner Ahmed_04_11/metadata.csv', sep=';')

# Select relevant columns
metadata_df = metadata_df[['N Dossier', 'Categorie']]

# Select specific categories
metadata_df = metadata_df[metadata_df['Categorie'].isin(['Healthy', 'Myopathic', 'Dystrophic', 'Inflammatory', 'Neurogenic'])]

# Process the slides
for slide_name in slides:
    # Check if the slide is in the metadata
    if slide_name[:7] not in metadata_df['N Dossier'].values:
        print(f"Skipping {slide_name} as it's not in metadata")
        continue

    print(f"Processing {slide_name}")
    slide_path = path + slide_name
    results_dir = path_results + '/' + metadata_df[metadata_df['N Dossier'] == slide_name[:7]]['Categorie'].values[0] + '/'

    # Crop the slide
    crop_tissues(slide_path, level=-1, min_area_ratio=0.005, show_steps=True, show_results=True, results_dir=results_dir)

# %% [markdown]
# # Crop to specific directory

# %%
path = '/workspaces/WSI-Classification/data/HE-MYO/Raw/DYSTROPHIC/'
results_dir = '/workspaces/WSI-Classification/data/HE-MYO/Processed/Dystrophic/'

# List the slides
slides = [slide for slide in os.listdir(path)]

for slide_name in slides:
    print(f"Processing {slide_name}")
    slide_path = path + slide_name
    crop_tissues(slide_path, level=-1, min_area_ratio=0.005, show_steps=True, show_results=True, results_dir=results_dir)
