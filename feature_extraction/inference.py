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
from pathlib import Path

from dataset import WSI_tiles

# Determine the device to run the model on (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
path = "/workspaces/WSI-Classification/data/HE-MYO/Processed/"
TILE_SIZE= (448, 448)
BATCH_SIZE = 64

# List the slides  all category directories (Dystrophic, Healthy, Inflammatory, Myopathic, Neurogenic)
slides = []
for category_dir in os.listdir(path):
    category_path = Path(path) / category_dir
    if category_path.is_dir():
        # For each category, scan slide directories
        for slide_dir in os.listdir(category_path):
            slide_dir_path = category_path / slide_dir
            if slide_dir_path.is_dir():
                # Find .ome.tiff files in the slide directory
                for slide_file in slide_dir_path.glob("*.ome.tiff"):
                    slides.append({
                        'name': slide_file.name,
                        'path': str(slide_file),
                        'category': category_dir,
                        'slide_dir': slide_dir
                    })

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

for slide_info in slides:
    slide_name = slide_info['name']
    slide_path = slide_info['path']
    category = slide_info['category']
    slide_dir = slide_info['slide_dir']
    
    # Determine output directory (slide's own directory)
    output_dir = Path(path) / category / slide_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if the embedding features already exist
    slide_base_name = Path(slide_name).stem
    features_path = output_dir / f"{slide_base_name}_features.pt"
    if features_path.exists():
        print(f"{slide_base_name} already processed")
        continue
    
    # Print the slide name with category
    print(f"Processing [{category}] {slide_name}")

    # Load the slide with openslide
    slide = openslide.OpenSlide(slide_path)

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
    tiles_path = output_dir / f"{slide_base_name}_tiles.csv"
    tiles.to_csv(tiles_path)

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
    torch.save(feature_emb, str(features_path))

    # Clean up the GPU memory
    torch.cuda.empty_cache()