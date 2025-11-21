import numpy as np
import pandas as pd
import torch
import openslide
import timm
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import WSI_tiles

# Determine the device to run the model on (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
path = "/workspaces/WSI-Classification/data/HE-MYO/Processed/Dystrophic"
TILE_SIZE= (448, 448)
BATCH_SIZE = 64

# liset the slides
slides = os.listdir(path)

# # Create the model
# model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)

# Load the whole model
model = torch.load("H-Optimus-0.pth", map_location=device, weights_only=False)

# Put the model on the device
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Create the transform function from the model configuration
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.707223, 0.578729, 0.703617), 
        std=(0.211883, 0.230117, 0.177517)
    ),
])

for slide_name in slides:
    # Check if the embedding features already exist
    if os.path.exists(f"{path}/{slide_name.split('.')[0]}_features.pkl"):
        print(f"{slide_name.split('.')[0]} already processid")
        continue
    
    # Print the slide name
    print(f"Processing {slide_name}")

    # Load the slide with openslide
    slide = openslide.OpenSlide(f"/workspaces/WSI-Classification/data/HE-MYO/Processed/Dystrophic/{slide_name}")

    # Calculate tile centers for 448x448 tiles without intersection
    slide_width, slide_height = slide.dimensions
    x_coords, y_coords = [], []
    for y in range(TILE_SIZE[1]//2, slide_height, TILE_SIZE[1]):
        for x in range(TILE_SIZE[0]//2, slide_width, TILE_SIZE[0]):
            x_coords.append(x)
            y_coords.append(y)

    # Create DataFrame with id, x, and y columns
    tiles = pd.DataFrame({'id': range(len(x_coords)), 'x': x_coords, 'y': y_coords})

    # Create the dataset
    dataset = WSI_tiles(slide=slide, tiles=tiles, transform=transform, size=TILE_SIZE)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create the embedding
    feature_emb = []

    # Iterate over the dataloader with tqdm
    for batch in tqdm(dataloader, desc="Processing batches"):
        
        # Get the embedding from the model
        with torch.inference_mode():
            output = model(batch.to(device))
        
        # Append the embedding to the list
        feature_emb.append(output.cpu().numpy())

    # Concatenate the embeddings
    feature_emb = np.concatenate(feature_emb)

    # Transform the embedding features into a dataframe
    feature_emb_df = pd.DataFrame(feature_emb, index=tiles.index)

    # Clean up the GPU memory
    torch.cuda.empty_cache()

    # Save the pixels
    tiles.to_pickle(f"{path}/{slide_name.split('.')[0]}_tiles.pkl")

    # Save the embedding features to a pickle file
    feature_emb_df.to_pickle(f"{path}/{slide_name.split('.')[0]}_features.pkl")