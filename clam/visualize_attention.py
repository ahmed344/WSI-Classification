"""
Attention visualization module for CLAM-MB model.

This module provides functions to visualize attention weights from the CLAM-MB model
as heatmaps overlaid on tissue slides. It generates multi-branch attention visualizations
showing how the model attends to different regions of the tissue.
"""
import os
from typing import List, Dict, Any, Optional
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import openslide

from clam_dataset import WSIFeatureDataset, collate_fn
from clam_model import CLAM_MB
from config_loader import load_config


def load_tile_coordinates(
    data_root: str,
    slide_name: str,
    tissue_name: str,
    class_folder: str
) -> Optional[np.ndarray]:
    """
    Load tile coordinates for a tissue from a slide directory.
    
    Expected file location: data_root/class_folder/slide_name/tissue_name_tiles.csv
    
    Args:
        data_root (str): Root directory containing class folders.
        slide_name (str): Name of the slide directory.
        tissue_name (str): Name of the tissue (used to construct filename).
        class_folder (str): Name of the class folder (category).
    
    Returns:
        Optional[np.ndarray]: Array of shape [n_tiles, 2] containing (x, y) coordinates
            for each tile. Returns None if file not found or cannot be read.
    """
    slide_path = os.path.join(data_root, class_folder, slide_name)
    if not os.path.isdir(slide_path):
        return None
    
    # Load tile coordinates from CSV file
    tiles_path = os.path.join(slide_path, f"{tissue_name}_tiles.csv")
    if os.path.exists(tiles_path):
        try:
            tiles_df = pd.read_csv(tiles_path)
            return tiles_df[['x', 'y']].values
        except Exception as e:
            print(f"Warning: Could not load tile coordinates from {tiles_path}: {e}")
            return None
    
    return None


def load_slide_thumbnail(
    data_root: str,
    slide_name: str,
    tissue_name: str,
    class_folder: str,
    thumbnail_size: int = 512
) -> Optional[np.ndarray]:
    """
    Load thumbnail image of a tissue slide.
    
    Expected file location: data_root/class_folder/slide_name/tissue_name.ome.tiff
    
    Args:
        data_root (str): Root directory containing class folders.
        slide_name (str): Name of the slide directory.
        tissue_name (str): Name of the tissue (used to construct filename).
        class_folder (str): Name of the class folder (category).
        thumbnail_size (int): Size of the thumbnail to generate (width and height).
            Defaults to 512.
    
    Returns:
        Optional[np.ndarray]: Thumbnail image as numpy array of shape
            [thumbnail_size, thumbnail_size, 3] (RGB). Returns None if file not found
            or cannot be loaded.
    """
    slide_path = os.path.join(data_root, class_folder, slide_name)
    if not os.path.isdir(slide_path):
        return None
    
    # Load thumbnail from tissue slide file
    tissue_slide_path = os.path.join(slide_path, f"{tissue_name}.ome.tiff")
    if os.path.exists(tissue_slide_path):
        try:
            slide = openslide.OpenSlide(tissue_slide_path)
            thumbnail = slide.get_thumbnail(size=(thumbnail_size, thumbnail_size))
            slide.close()
            return np.array(thumbnail)
        except Exception as e:
            print(f"Warning: Could not load tissue thumbnail from {tissue_slide_path}: {e}")
            return None
    
    return None


def visualize_attention_branches(
    attention_weights_list: List[np.ndarray],
    avg_attention: np.ndarray,
    tile_coords: np.ndarray,
    slide_name: str,
    tissue_name: str,
    class_name: str,
    predicted_class: str,
    class_folders: List[str],
    output_path: str,
    data_root: str,
    class_folder: str,
    tile_size: int = 448,
    thumbnail_size: int = 512
) -> None:
    """
    Visualize attention weights from all branches as heatmaps.
    
    Creates a multi-panel figure showing:
    1. Original tissue thumbnail
    2. Average attention across all branches
    3. Individual attention for each branch (one per class)
    
    Args:
        attention_weights_list (List[np.ndarray]): List of attention weight arrays,
            one per branch. Each array has shape [n_tiles].
        avg_attention (np.ndarray): Average attention weights across all branches.
            Shape [n_tiles].
        tile_coords (np.ndarray): Tile coordinates array of shape [n_tiles, 2]
            containing (x, y) coordinates for each tile.
        slide_name (str): Name of the slide directory.
        tissue_name (str): Name of the tissue.
        class_name (str): True class label for this tissue.
        predicted_class (str): Predicted class label from the model.
        class_folders (List[str]): List of all class names (for branch labels).
        output_path (str): Full path where the figure should be saved.
        data_root (str): Root directory containing slide files.
        class_folder (str): Class folder name for this slide (used to load thumbnail).
        tile_size (int): Size of each tile in pixels. Defaults to 448.
        thumbnail_size (int): Size of thumbnail to generate. Defaults to 512.
    """
    if tile_coords is None or len(tile_coords) == 0:
        print(f"Warning: No tile coordinates found for {slide_name}/{tissue_name}")
        return
    
    # Load thumbnail image
    thumbnail = load_slide_thumbnail(
        data_root, slide_name, tissue_name, class_folder, thumbnail_size
    )
    
    # Calculate number of subplots: thumbnail + average + one per branch
    n_branches = len(attention_weights_list)
    n_subplots = 1 + 1 + n_branches  # thumbnail + average + branches
    
    # Create figure with horizontal layout
    fig, axes = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 4))
    
    # Calculate coordinate ranges for consistent axis limits across all plots
    x_min, x_max = tile_coords[:, 0].min(), tile_coords[:, 0].max()
    y_min, y_max = tile_coords[:, 1].min(), tile_coords[:, 1].max()
    # Add padding equal to half tile size for better visualization
    x_min -= tile_size // 2
    x_max += tile_size // 2
    y_min -= tile_size // 2
    y_max += tile_size // 2
    
    subplot_idx = 0
    
    # Plot thumbnail (first subplot)
    if thumbnail is not None:
        axes[subplot_idx].imshow(thumbnail)
        axes[subplot_idx].set_title('Original', fontsize=10)
        axes[subplot_idx].axis('off')
    else:
        # Show placeholder if thumbnail unavailable
        axes[subplot_idx].text(
            0.5, 0.5, 'Thumbnail\nNot Available',
            ha='center', va='center', fontsize=10
        )
        axes[subplot_idx].axis('off')
    subplot_idx += 1
    
    # Plot average attention (second subplot)
    # Normalize attention weights to [0, 1] for color mapping
    avg_norm = (avg_attention - avg_attention.min()) / (
        avg_attention.max() - avg_attention.min() + 1e-8
    )
    
    # Draw colored rectangles for each tile based on attention weight
    for j, (x, y) in enumerate(tile_coords):
        rect = patches.Rectangle(
            (x - tile_size // 2, y - tile_size // 2),
            tile_size, tile_size,
            linewidth=0, edgecolor='none',
            facecolor=plt.cm.viridis(avg_norm[j])  # Use viridis colormap
        )
        axes[subplot_idx].add_patch(rect)
    
    # Set axis limits and properties
    axes[subplot_idx].set_xlim(x_min, x_max)
    axes[subplot_idx].set_ylim(y_max, y_min)  # Invert y-axis (image coordinates)
    axes[subplot_idx].set_aspect('equal', adjustable='box')
    axes[subplot_idx].axis('off')
    axes[subplot_idx].set_title('Average', fontsize=10)
    
    # Add colorbar for average attention
    sm_avg = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=avg_attention.min(), vmax=avg_attention.max())
    )
    sm_avg.set_array([])
    cbar_avg = plt.colorbar(
        sm_avg, ax=axes[subplot_idx], orientation='vertical', pad=0.05
    )
    cbar_avg.ax.xaxis.set_label_position('top')
    subplot_idx += 1
    
    # Plot each branch's attention (one subplot per class)
    for i, (attn_weights, class_name_branch) in enumerate(
        zip(attention_weights_list, class_folders)
    ):
        ax = axes[subplot_idx]
        
        # Normalize this branch's attention weights
        attn_norm = (attn_weights - attn_weights.min()) / (
            attn_weights.max() - attn_weights.min() + 1e-8
        )
        
        # Draw colored rectangles for each tile
        for j, (x, y) in enumerate(tile_coords):
            rect = patches.Rectangle(
                (x - tile_size // 2, y - tile_size // 2),
                tile_size, tile_size,
                linewidth=0, edgecolor='none',
                facecolor=plt.cm.viridis(attn_norm[j])
            )
            ax.add_patch(rect)
        
        # Set axis limits and properties
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Invert y-axis
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.set_title(class_name_branch, fontsize=10)
        
        # Add colorbar for this branch
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis,
            norm=plt.Normalize(vmin=attn_weights.min(), vmax=attn_weights.max())
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05)
        
        subplot_idx += 1
    
    # Set figure title with slide/tissue info and predictions
    fig.suptitle(
        f'Slide: {slide_name} | Tissue: {tissue_name}\n'
        f'True: {class_name} | Predicted: {predicted_class}',
        fontsize=12, fontweight='bold'
    )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    tqdm.write(f"Saved attention heatmap to {output_path}")


def evaluate_with_attention(
    model: CLAM_MB,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    data_root: str,
    output_dir: str,
    tile_size: int = 448
) -> List[Dict[str, Any]]:
    """
    Evaluate model and extract attention weights for visualization.
    
    Processes batches from the dataloader, extracts attention weights from the model,
    and generates attention heatmaps for each tissue sample.
    
    Args:
        model (CLAM_MB): Trained CLAM-MB model in evaluation mode.
        dataloader (DataLoader): DataLoader providing batches of tissue samples.
        device (torch.device): Device to run inference on (CPU or CUDA).
        class_names (List[str]): List of class names for labeling.
        data_root (str): Root directory containing slide files.
        output_dir (str): Directory where attention heatmaps should be saved.
        tile_size (int): Size of each tile in pixels. Defaults to 448.
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries, one per sample, containing:
            - 'slide_name' (str): Name of the slide.
            - 'tissue_name' (str): Name of the tissue.
            - 'true_class' (str): True class label.
            - 'predicted_class' (str): Predicted class label.
            - 'max_attention' (float): Maximum attention weight value.
            - 'mean_attention' (float): Mean attention weight value.
            - 'attention_entropy' (float): Entropy of attention distribution.
    """
    model.eval()
    all_results: List[Dict[str, Any]] = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Extracting attention')
        for batch in pbar:
            # Move batch to device
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['masks'].to(device)
            slide_names = batch['slide_names']
            tissue_names = batch.get('tissue_names', [None] * len(slide_names))
            
            # Forward pass through model
            outputs = model(features, masks)
            logits = outputs['logits']
            attention_weights_list = outputs['attention_weights']
            
            # Get predictions from logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Process each tissue sample in the batch
            for i, slide_name in enumerate(slide_names):
                tissue_name = tissue_names[i] if i < len(tissue_names) else None
                
                # Compute average attention across all branches
                avg_attention = torch.stack([
                    aw[i] for aw in attention_weights_list
                ]).mean(dim=0).cpu().numpy()
                
                # Get individual branch attentions
                branch_attentions = [
                    aw[i].cpu().numpy() for aw in attention_weights_list
                ]
                
                # Remove padding: only keep valid tiles
                valid_mask = masks[i].cpu().numpy()
                avg_attention = avg_attention[valid_mask]
                branch_attentions = [
                    attn[valid_mask] for attn in branch_attentions
                ]
                
                # Get true and predicted class labels
                true_label = int(labels[i].cpu().numpy())
                pred_label = int(preds[i])
                true_class = class_names[true_label]
                pred_class = class_names[pred_label]
                
                # Find class folder and load tile coordinates
                class_folder = None
                tile_coords = None
                for cf in class_names:
                    if tissue_name:
                        tile_coords = load_tile_coordinates(
                            data_root, slide_name, tissue_name, cf
                        )
                        if tile_coords is not None:
                            class_folder = cf
                            break
                
                # Generate visualization if coordinates are available
                if tile_coords is not None:
                    # Filter coordinates to valid tiles only
                    tile_coords = tile_coords[valid_mask]
                    
                    # Create safe filenames (replace problematic characters)
                    slide_safe_name = slide_name.replace('/', '_').replace(' ', '_')
                    tissue_safe_name = (
                        tissue_name.replace('/', '_').replace(' ', '_')
                        if tissue_name else 'unknown'
                    )
                    
                    # Generate output path and create visualization
                    multi_path = os.path.join(
                        output_dir,
                        f'{slide_safe_name}_{tissue_safe_name}_attention_branches.png'
                    )
                    visualize_attention_branches(
                        branch_attentions, avg_attention, tile_coords,
                        slide_name, tissue_name, true_class, pred_class,
                        class_names, multi_path, data_root, class_folder,
                        tile_size=tile_size
                    )
                
                # Store results for summary
                all_results.append({
                    'slide_name': slide_name,
                    'tissue_name': tissue_name,
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'max_attention': float(avg_attention.max()),
                    'mean_attention': float(avg_attention.mean()),
                    'attention_entropy': float(
                        -np.sum(avg_attention * np.log(avg_attention + 1e-8))
                    )
                })
    
    return all_results


def main() -> None:
    """
    Main function to generate attention visualizations.
    
    Loads model checkpoint, creates dataset, and generates attention heatmaps
    for all samples in the specified split (train or val).
    """
    # Load configuration from config.yml
    config = load_config()
    
    # Resolve checkpoint path
    checkpoint_path = config['paths']['checkpoint']
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            config['checkpoint_dir'], 'best_model.pth'
        )
    
    # Resolve output directory
    output_dir = config['paths']['attention_output']
    if output_dir is None:
        output_dir = os.path.join(config['output_dir'], 'attention_heatmaps')
    
    # Get visualization parameters from config
    vis_config = config.get('visualization', {})
    split = vis_config.get('split', 'val')
    max_slides = vis_config.get('max_slides', None)
    tile_size = vis_config.get('tile_size', 448)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model checkpoint
    print(f'Loading checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    
    # Extract model configuration from checkpoint
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        input_dim = model_config.get('input_dim', config['input_dim'])
        hidden_dim = model_config.get('hidden_dim', config['hidden_dim'])
        num_classes = model_config.get('num_classes', config['num_classes'])
        k_clusters = model_config.get('k_clusters', config['k_clusters'])
        class_folders = checkpoint.get('class_folders', None)
    elif 'args' in checkpoint:
        model_args = checkpoint['args']
        input_dim = model_args.get('input_dim', config['input_dim'])
        hidden_dim = model_args.get('hidden_dim', config['hidden_dim'])
        num_classes = model_args.get('num_classes', config['num_classes'])
        k_clusters = model_args.get('k_clusters', config['k_clusters'])
        class_folders = checkpoint.get('class_folders', None)
    else:
        # Fallback to config file values
        input_dim = config['input_dim']
        hidden_dim = config['hidden_dim']
        num_classes = config['num_classes']
        k_clusters = config['k_clusters']
        class_folders = None
    
    # Auto-detect class folders if not in checkpoint
    if class_folders is None:
        class_folders = sorted([
            d for d in os.listdir(config['data_root'])
            if os.path.isdir(os.path.join(config['data_root'], d))
        ])
    
    print(f'Classes: {class_folders}')
    
    # Create model with extracted configuration
    print('Creating model...')
    model = CLAM_MB(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        k_clusters=k_clusters
    ).to(device)
    
    # Load model weights from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded successfully')
    
    # Create dataset for specified split
    print(f'Loading {split} dataset...')
    dataset = WSIFeatureDataset(
        config['data_root'],
        class_folders=class_folders,
        split=split,
        train_ratio=config['train_ratio'],
        random_seed=config['random_seed']
    )
    
    # Optionally limit number of samples for faster processing
    if max_slides:
        dataset.indices = dataset.indices[:max_slides]
    
    print(f'Number of samples: {len(dataset)}')
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Extract attention weights and generate visualizations
    print('Extracting attention weights and creating heatmaps...')
    results = evaluate_with_attention(
        model, dataloader, device, class_folders,
        config['data_root'], output_dir, tile_size
    )
    
    # Save summary JSON file
    summary_path = os.path.join(output_dir, f'attention_summary_{split}.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nAttention summary saved to {summary_path}')
    
    print(f'\nGenerated {len(results)} attention heatmaps')
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    main()
