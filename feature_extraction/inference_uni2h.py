import torch
import timm
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms

from dataset import WSI_tiles
from utils import collect_slides, ensure_tiles_and_qc

# Determine the device to run the model on (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
path = "/workspaces/WSI-Classification/data/HE-MYO/Processed/"
TILE_SIZE= (448, 448)
BATCH_SIZE = 512
QC_DIR = Path("/workspaces/WSI-Classification/data/HE-MYO/Quality_controls/create_tiles")

# List the slides  all category directories (Dystrophic, Healthy, Inflammatory, Myopathic, Neurogenic)
slides = collect_slides(path)

# pretrained=True needed to load UNI2-h weights (and download weights for the first time)
timm_kwargs = {
            'model_name': 'hf-hub:MahmoodLab/UNI2-h',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }

# Create the model
model = timm.create_model(pretrained=True, **timm_kwargs)
# Put the model on the device
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Create the transform function from the model configuration
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

for slide_info in slides:
    slide_name = slide_info['name']
    slide_path = slide_info['path']
    category = slide_info['category']
    slide_dir = slide_info['slide_dir']
    
    # Determine output directory (slide's own directory)
    output_dir = Path(path) / category / slide_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if the embedding features already exist
    slide_base_name = Path(slide_name).name.removesuffix(".ome.tiff")
    features_path = output_dir / f"{slide_base_name}_features_uni2h.pt"
    if features_path.exists():
        print(f"{slide_base_name} already processed")
        continue

    tiles_path = ensure_tiles_and_qc(
        slide_path=slide_path,
        output_dir=output_dir,
        slide_base_name=slide_base_name,
        tile_size=TILE_SIZE,
        qc_dir=QC_DIR,
        quantile=0.75,
    )
    
    # Print the slide name with category
    print(f"Processing [{category}] {slide_base_name}")

    # Load the tiles
    tiles = pd.read_csv(tiles_path)

    # Create the dataset
    dataset = WSI_tiles(slide_path=slide_path, tiles=tiles, transform=transform, size=TILE_SIZE)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

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