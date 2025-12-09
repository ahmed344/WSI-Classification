import numpy as np
import pandas as pd
import torch
import cv2
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
path = "/workspaces/WSI-Classification/data/HE-MYO/Processed/Inflammatory"
TILE_SIZE= (448, 448)
BATCH_SIZE = 64

# liset the slides
slides = os.listdir(path)

# # Create the model
# model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)

# Load the whole model
model = torch.load("/workspaces/WSI-Classification/models/H-Optimus-0.pth", map_location=device, weights_only=False)

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
        print(f"{slide_name.split('.')[0]} already processed")
        continue
    
    # Print the slide name
    print(f"Processing {slide_name}")

    # Load the slide with openslide
    slide = openslide.OpenSlide(f"{path}/{slide_name}")

    # Read the slide as RGB
    img = slide.read_region((0, 0), 0, slide.dimensions).convert("RGB")

    # Convert RGB to HSV for color segmentation
    hsv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)

    # Threshold to segment tissue (non-white)
    lower = (0, 10, 0)
    upper = (180, 255, 220)
    mask = cv2.inRange(hsv_img, lower, upper)

    # Calculate tile centers for 448x448 tiles without intersection
    delta = (TILE_SIZE[0] // 2, TILE_SIZE[1] // 2)
    slide_width, slide_height = slide.dimensions
    x_coords, y_coords = [], []
    for y in range(delta[1], slide_height, TILE_SIZE[1]):
        for x in range(delta[0], slide_width, TILE_SIZE[0]):
            if np.median(mask[y-delta[1]:y+delta[1], x-delta[0]:x+delta[0]]) > 0:
                x_coords.append(x)
                y_coords.append(y)

    # Create DataFrame with id, x, and y columns
    tiles = pd.DataFrame({'id': range(len(x_coords)), 'x': x_coords, 'y': y_coords})

    # Save the pixels
    tiles.to_csv(f"{path}/{slide_name.split('.')[0]}_tiles.csv")

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
        feature_emb.append(output.cpu())

    # Concatenate the embeddings into a single tensor
    feature_emb = torch.cat(feature_emb, dim=0)

    # Save the embedding features as a tensor using torch.save
    torch.save(feature_emb, f"{path}/{slide_name.split('.')[0]}_features.pt")

    # Clean up the GPU memory
    torch.cuda.empty_cache()