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
path = '/data/aabdelrahman/HE-MYO/Raw/Giessen/HE Muscle biopsy/'
path_results = '/data/aabdelrahman/HE-MYO/Processed/'

# List the slides
slides = [slide for slide in os.listdir(path)]
slides_ids = [slide[:-5] for slide in slides]

# Load the metadata
metadata_df = pd.read_excel('/data/aabdelrahman/HE-MYO/Raw/Giessen/Cooperation_Malfatti_Sept.25_II.xlsx')

# Remove nan values
metadata_df = metadata_df.dropna()

# Strip whitespace from all string columns
metadata_df = metadata_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Change Category "Control" to "Healthy"
metadata_df['Category'] = metadata_df['Category'].replace('Control', 'Healthy')

# Select specific categories
metadata_df = metadata_df[metadata_df['Category'].isin(['Healthy', 'Myopathic', 'Dystrophic', 'Inflammatory', 'Neurogenic'])]

# Process the slides
for slide_name in slides:
    # Check if the slide is in the metadata
    if slide_name[:-5] not in metadata_df['Number'].values:
        print(f"Skipping {slide_name} as it's not in metadata")
        continue

    # Check if the slide is already processed
    if os.path.exists(path_results + '/' + metadata_df[metadata_df['Number'] == slide_name[:-5]]['Category'].values[0] + '/' + slide_name[:-5]):
        print(f"Skipping {slide_name} as it's already processed")
        continue

    print(f"Processing {slide_name}")
    slide_path = path + slide_name
    results_dir = path_results + '/' + metadata_df[metadata_df['Number'] == slide_name[:-5]]['Category'].values[0] + '/'
    # Crop the slide
    crop_tissues(slide_path, level=6, min_area_ratio=0.005, show_steps=True, show_results=True, results_dir=results_dir)

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
