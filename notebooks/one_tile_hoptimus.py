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
import torch
import timm
import openslide

from torchvision import transforms
from huggingface_hub import login, snapshot_download

import matplotlib.pyplot as plt

# %%
print(timm.__version__)

# %%
# Determine the device to run the model on (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %%
# Read the huggingface token from a fileReviewing the codebase to understand the current structure and data format.
with open("huggingface_token.key", "r") as f:
    token = f.read()

# Login to huggingface hub with the token
login(token)

# %%
# Download the model from huggingface hub
snapshot_download(repo_id="bioptimus/H-optimus-0", local_dir=".", local_dir_use_symlinks=False, token=token)

# %%
# Create the model
model = timm.create_model("hf-hub:bioptimus/H-optimus-0", cache_dir="/workspaces/WSI-Classification/", pretrained=True, init_values=1e-5, dynamic_img_size=False)
# model = torch.load("H-Optimus-0.pth", map_location=device, weights_only=False)

# Put the model on the device
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# %%
# Save the whole model
torch.save(model, "H-Optimus-0.pth")

# %%
# Load the whole model
model = torch.load("H-Optimus-0.pth", map_location=device, weights_only=False)

# Set the model to evaluation mode
model.eval()
# %%
# Create the transform function as specified in the model huggingface config
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.707223, 0.578729, 0.703617), 
        std=(0.211883, 0.230117, 0.177517)
    ),
])
transform

# %%
# Determine the slide id
slide = "tissue_crop_4352_30208"

# Load the slide with openslide
slide = openslide.OpenSlide(f"/workspaces/WSI-Classification/data/HE-MYO/Processed/Dystrophic/{slide}.ome.tiff")

# Make a thumbnail of the slide
slide.get_thumbnail(size=(256, 256))

# %%
# Read a 426*426 tile image of the slide
tile = slide.read_region(location=(2000, 2000), level=0, size=(426, 426))

# Convert to RGB
tile = tile.convert("RGB")

# Transform the image to the model suitable input tensor
image = transform(tile).unsqueeze(dim=0).to(device)

# Extracted features (torch.Tensor) with shape [1,1536]
with torch.inference_mode():
    feature_emb = model(image) 

# %%
# Plot both the original image and the image seen by the model
fig, ax = plt.subplots(1, 3, figsize=(17, 5))

ax[0].imshow(tile)
ax[0].set_title("Original Image")

ax[1].imshow(image.cpu().squeeze().permute(1, 2, 0))
ax[1].set_title("Image as seen by the model")

ax[2].plot(feature_emb.cpu().squeeze().detach().numpy())
ax[2].set_title("Extracted features")

plt.show()
