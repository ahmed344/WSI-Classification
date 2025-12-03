import os
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


def load_tile_coordinates(data_root, slide_name, class_folder):
    """Load tile coordinates for a slide"""
    tiles_path = os.path.join(data_root, class_folder, f"{slide_name}_tiles.pkl")
    if os.path.exists(tiles_path):
        tiles_df = pd.read_pickle(tiles_path)
        return tiles_df[['x', 'y']].values
    return None


def load_slide_thumbnail(data_root, slide_name, class_folder, thumbnail_size=512):
    """Load thumbnail of the original slide"""
    slide_path = os.path.join(data_root, class_folder, f"{slide_name}.ome.tiff")
    if os.path.exists(slide_path):
        try:
            slide = openslide.OpenSlide(slide_path)
            thumbnail = slide.get_thumbnail(size=(thumbnail_size, thumbnail_size))
            slide.close()
            return np.array(thumbnail)
        except Exception as e:
            print(f"Warning: Could not load slide thumbnail: {e}")
            return None
    return None


def visualize_attention_branches(attention_weights_list, avg_attention, tile_coords, slide_name, 
                                class_name, predicted_class, class_folders, output_path, 
                                data_root, tile_size=448, thumbnail_size=512):
    """
    Visualize attention weights from all branches with original thumbnail and average in one row
    
    Args:
        attention_weights_list: list of attention weight arrays, one per branch
        avg_attention: average attention weights across all branches
        tile_coords: numpy array of tile coordinates
        slide_name: name of the slide
        class_name: true class name
        predicted_class: predicted class name
        class_folders: list of class names
        output_path: path to save the figure
        data_root: root directory containing slide files
        tile_size: size of each tile in pixels
        thumbnail_size: size of thumbnail to generate
    """
    if tile_coords is None or len(tile_coords) == 0:
        print(f"Warning: No tile coordinates found for {slide_name}")
        return
    
    # Find class folder for this slide
    class_folder = None
    for cf in class_folders:
        tiles_path = os.path.join(data_root, cf, f"{slide_name}_tiles.pkl")
        if os.path.exists(tiles_path):
            class_folder = cf
            break
    
    if class_folder is None:
        print(f"Warning: Could not find class folder for {slide_name}")
        return
    
    # Load thumbnail
    thumbnail = load_slide_thumbnail(data_root, slide_name, class_folder, thumbnail_size)
    
    n_branches = len(attention_weights_list)
    # Total subplots: thumbnail + average + n_branches = 1 + 1 + n_branches
    n_subplots = 1 + 1 + n_branches  # thumbnail + average + branches
    
    fig, axes = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 4))
    
    # Get coordinate ranges for consistent axis limits
    x_min, x_max = tile_coords[:, 0].min(), tile_coords[:, 0].max()
    y_min, y_max = tile_coords[:, 1].min(), tile_coords[:, 1].max()
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
        axes[subplot_idx].text(0.5, 0.5, 'Thumbnail\nNot Available', 
                              ha='center', va='center', fontsize=10)
        axes[subplot_idx].axis('off')
    subplot_idx += 1
    
    # Plot average attention (second subplot)
    avg_norm = (avg_attention - avg_attention.min()) / (avg_attention.max() - avg_attention.min() + 1e-8)
    for j, (x, y) in enumerate(tile_coords):
        rect = patches.Rectangle(
            (x - tile_size // 2, y - tile_size // 2),
            tile_size, tile_size,
            linewidth=0, edgecolor='none',
            facecolor=plt.cm.viridis(avg_norm[j])
        )
        axes[subplot_idx].add_patch(rect)
    
    axes[subplot_idx].set_xlim(x_min, x_max)
    axes[subplot_idx].set_ylim(y_max, y_min)
    axes[subplot_idx].set_aspect('equal', adjustable='box')
    axes[subplot_idx].axis('off')
    axes[subplot_idx].set_title('Average', fontsize=10)
    
    # Add colorbar for average
    sm_avg = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(vmin=avg_attention.min(),
                                                      vmax=avg_attention.max()))
    sm_avg.set_array([])
    cbar_avg = plt.colorbar(sm_avg, ax=axes[subplot_idx], orientation='vertical', pad=0.05)
    cbar_avg.ax.xaxis.set_label_position('top')
    subplot_idx += 1
    
    # Plot each branch
    for i, (attn_weights, class_name_branch) in enumerate(zip(attention_weights_list, class_folders)):
        ax = axes[subplot_idx]
        attn_norm = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min() + 1e-8)
        
        # Create filled rectangles for each tile
        for j, (x, y) in enumerate(tile_coords):
            rect = patches.Rectangle(
                (x - tile_size // 2, y - tile_size // 2),
                tile_size, tile_size,
                linewidth=0, edgecolor='none',
                facecolor=plt.cm.viridis(attn_norm[j])
            )
            ax.add_patch(rect)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.set_title(class_name_branch, fontsize=10)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(vmin=attn_weights.min(),
                                                      vmax=attn_weights.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05)
        
        subplot_idx += 1
    
    fig.suptitle(f'{slide_name}\nTrue: {class_name} | Predicted: {predicted_class}',
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    tqdm.write(f"Saved attention heatmap to {output_path}")


def evaluate_with_attention(model, dataloader, device, class_names, data_root, output_dir, tile_size=448):
    """Evaluate model and extract attention weights"""
    model.eval()
    all_results = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Extracting attention')
        for batch in pbar:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['masks'].to(device)
            slide_names = batch['slide_names']
            
            # Forward pass
            outputs = model(features, masks)
            logits = outputs['logits']
            attention_weights_list = outputs['attention_weights']
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Process each slide in batch
            for i, slide_name in enumerate(slide_names):
                # Get attention weights (average across branches for single heatmap)
                avg_attention = torch.stack([aw[i] for aw in attention_weights_list]).mean(dim=0).cpu().numpy()
                branch_attentions = [aw[i].cpu().numpy() for aw in attention_weights_list]
                
                # Get valid tiles (remove padding)
                valid_mask = masks[i].cpu().numpy()
                avg_attention = avg_attention[valid_mask]
                branch_attentions = [attn[valid_mask] for attn in branch_attentions]
                
                # Get true and predicted classes
                true_label = int(labels[i].cpu().numpy())
                pred_label = int(preds[i])
                true_class = class_names[true_label]
                pred_class = class_names[pred_label]
                
                # Find which class folder this slide belongs to
                class_folder = None
                tile_coords = None
                for cf in class_names:
                    tiles_path = os.path.join(data_root, cf, f"{slide_name}_tiles.pkl")
                    if os.path.exists(tiles_path):
                        class_folder = cf
                        tile_coords = load_tile_coordinates(data_root, slide_name, cf)
                        break
                
                if tile_coords is not None:
                    # Filter to valid tiles
                    tile_coords = tile_coords[valid_mask]
                    
                    # Create output paths
                    slide_safe_name = slide_name.replace('/', '_').replace(' ', '_')
                    
                    # Multi-branch attention heatmaps with thumbnail and average
                    multi_path = os.path.join(output_dir, f'{slide_safe_name}_attention_branches.png')
                    visualize_attention_branches(
                        branch_attentions, avg_attention, tile_coords, slide_name,
                        true_class, pred_class, class_names, multi_path, 
                        data_root, tile_size=tile_size
                    )
                
                all_results.append({
                    'slide_name': slide_name,
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'max_attention': float(avg_attention.max()),
                    'mean_attention': float(avg_attention.mean()),
                    'attention_entropy': float(-np.sum(avg_attention * np.log(avg_attention + 1e-8)))
                })
    
    return all_results


def main():
    # Load configuration
    config = load_config()
    
    # Get checkpoint path
    checkpoint_path = config['paths']['checkpoint']
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
    
    # Get output directory
    output_dir = config['paths']['attention_output']
    if output_dir is None:
        output_dir = os.path.join(config['output_dir'], 'attention_heatmaps')
    
    # Get visualization parameters
    vis_config = config.get('visualization', {})
    split = vis_config.get('split', 'val')
    max_slides = vis_config.get('max_slides', None)
    tile_size = vis_config.get('tile_size', 448)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load checkpoint
    print(f'Loading checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model arguments
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
        input_dim = config['input_dim']
        hidden_dim = config['hidden_dim']
        num_classes = config['num_classes']
        k_clusters = config['k_clusters']
        class_folders = None
    
    if class_folders is None:
        class_folders = sorted([d for d in os.listdir(config['data_root'])
                               if os.path.isdir(os.path.join(config['data_root'], d))])
    
    print(f'Classes: {class_folders}')
    
    # Create model
    print('Creating model...')
    model = CLAM_MB(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        k_clusters=k_clusters
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded successfully')
    
    # Create dataset
    print(f'Loading {split} dataset...')
    dataset = WSIFeatureDataset(
        config['data_root'],
        class_folders=class_folders,
        split=split,
        train_ratio=config['train_ratio'],
        random_seed=config['random_seed']
    )
    
    if max_slides:
        # Limit dataset size
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
    
    # Extract attention and create visualizations
    print('Extracting attention weights and creating heatmaps...')
    results = evaluate_with_attention(
        model, dataloader, device, class_folders, config['data_root'], output_dir, tile_size
    )
    
    # Save summary
    summary_path = os.path.join(output_dir, f'attention_summary_{split}.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nAttention summary saved to {summary_path}')
    
    print(f'\nGenerated {len(results)} attention heatmaps')
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    main()

