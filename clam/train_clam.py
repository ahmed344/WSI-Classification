"""
Training script for CLAM-MB (Multi-Branch) model.

This module provides training functionality for the CLAM-MB model using a mixed
loss function combining classification loss and clustering loss. It includes
early stopping, learning rate scheduling, and checkpoint saving.
"""
from typing import Dict, Any, List
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json

from clam_dataset import WSIFeatureDataset, collate_fn
from clam_model import CLAM_MB, compute_clustering_loss
from config_loader import load_config

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


def train_epoch(
    model: CLAM_MB,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    clustering_weight: float = 0.1
) -> Dict[str, float]:
    """
    Train the model for one epoch using mixed loss function.
    
    The total loss is computed as:
        total_loss = classification_loss + clustering_weight * clustering_loss
    
    Args:
        model (CLAM_MB): The CLAM-MB model to train.
        dataloader (DataLoader): DataLoader providing batches of training samples.
        criterion (nn.Module): Loss function for classification (typically CrossEntropyLoss).
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run training on (CPU or CUDA).
        clustering_weight (float): Weight for clustering loss component. Defaults to 0.1.
    
    Returns:
        Dict[str, float]: Dictionary containing training metrics:
            - 'loss' (float): Average total loss for the epoch.
            - 'cls_loss' (float): Average classification loss for the epoch.
            - 'cluster_loss' (float): Average clustering loss for the epoch.
            - 'accuracy' (float): Classification accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_cluster_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    
    # Iterate over batches
    for batch in dataloader:
        # Move batch data to device
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        masks = batch['masks'].to(device)
        
        # Forward pass through model
        outputs = model(features, masks)
        logits = outputs['logits']
        cluster_assignments = outputs['cluster_assignments']
        
        # Compute classification loss
        cls_loss = criterion(logits, labels)
        
        # Compute clustering loss (encourages confident cluster assignments)
        cluster_loss = compute_clustering_loss(cluster_assignments, masks)
        
        # Combine losses with weighted sum
        total_loss = cls_loss + clustering_weight * cluster_loss
        
        # Backward pass: compute gradients
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Accumulate statistics for epoch summary
        running_loss += total_loss.item()
        running_cls_loss += cls_loss.item()
        running_cluster_loss += cluster_loss.item()
        
        # Collect predictions for accuracy calculation
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Compute epoch averages
    epoch_loss = running_loss / len(dataloader)
    epoch_cls_loss = running_cls_loss / len(dataloader)
    epoch_cluster_loss = running_cluster_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': epoch_loss,
        'cls_loss': epoch_cls_loss,
        'cluster_loss': epoch_cluster_loss,
        'accuracy': accuracy
    }


def validate(
    model: CLAM_MB,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    clustering_weight: float = 0.1
) -> Dict[str, Any]:
    """
    Validate the model using mixed loss function.
    
    The total loss is computed as:
        total_loss = classification_loss + clustering_weight * clustering_loss
    
    Args:
        model (CLAM_MB): The CLAM-MB model to validate.
        dataloader (DataLoader): DataLoader providing batches of validation samples.
        criterion (nn.Module): Loss function for classification (typically CrossEntropyLoss).
        device (torch.device): Device to run validation on (CPU or CUDA).
        clustering_weight (float): Weight for clustering loss component. Defaults to 0.1.
    
    Returns:
        Dict[str, Any]: Dictionary containing validation metrics:
            - 'loss' (float): Average total loss for the epoch.
            - 'cls_loss' (float): Average classification loss for the epoch.
            - 'cluster_loss' (float): Average clustering loss for the epoch.
            - 'accuracy' (float): Classification accuracy for the epoch.
            - 'confusion_matrix' (List[List[int]]): Confusion matrix as nested list.
            - 'predictions' (List[int]): List of predicted class indices.
            - 'labels' (List[int]): List of true class indices.
    """
    model.eval()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_cluster_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    
    # Disable gradient computation for validation
    with torch.no_grad():
        for batch in dataloader:
            # Move batch data to device
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['masks'].to(device)
            
            # Forward pass through model
            outputs = model(features, masks)
            logits = outputs['logits']
            cluster_assignments = outputs['cluster_assignments']
            
            # Compute classification loss
            cls_loss = criterion(logits, labels)
            
            # Compute clustering loss
            cluster_loss = compute_clustering_loss(cluster_assignments, masks)
            
            # Combine losses with weighted sum
            total_loss = cls_loss + clustering_weight * cluster_loss
            
            # Accumulate statistics
            running_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_cluster_loss += cluster_loss.item()
            
            # Collect predictions for metrics
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Compute epoch averages
    epoch_loss = running_loss / len(dataloader)
    epoch_cls_loss = running_cls_loss / len(dataloader)
    epoch_cluster_loss = running_cluster_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Compute confusion matrix for detailed analysis
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': epoch_loss,
        'cls_loss': epoch_cls_loss,
        'cluster_loss': epoch_cluster_loss,
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'labels': all_labels
    }


def main() -> None:
    """
    Main training function.
    
    Loads configuration, creates datasets and model, and runs training loop with
    early stopping, learning rate scheduling, and checkpoint saving.
    """
    # Load configuration from config.yml
    config = load_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Determine device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Create training and validation datasets
    print('Loading datasets...')
    train_dataset = WSIFeatureDataset(
        config['data_root'],
        split='train',
        train_ratio=config['train_ratio'],
        random_seed=config['random_seed']
    )
    val_dataset = WSIFeatureDataset(
        config['data_root'],
        split='val',
        train_ratio=config['train_ratio'],
        random_seed=config['random_seed']
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Classes: {train_dataset.class_folders}')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,  # Shuffle training data
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid pickle issues with custom dataset
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,  # Don't shuffle validation data
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model with configuration parameters
    print('Creating model...')
    model = CLAM_MB(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        k_clusters=config['k_clusters']
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Setup loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    # Reduce learning rate when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Initialize training history tracking
    history: Dict[str, Dict[str, List[float]]] = {
        'train': {'loss': [], 'cls_loss': [], 'cluster_loss': [], 'accuracy': []},
        'val': {'loss': [], 'cls_loss': [], 'cluster_loss': [], 'accuracy': []}
    }
    
    # Initialize best metrics and early stopping counter
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    # Training loop
    print('Starting training...')
    for epoch in tqdm(range(config['epochs'])):
        # Train for one epoch
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            config['clustering_weight']
        )
        
        # Validate for one epoch
        val_metrics = validate(
            model, val_loader, criterion, device, config['clustering_weight']
        )
        
        # Update learning rate based on validation loss
        scheduler.step(val_metrics['loss'])
        
        # Save metrics to history
        for key in ['loss', 'cls_loss', 'cluster_loss', 'accuracy']:
            history['train'][key].append(train_metrics[key])
            history['val'][key].append(val_metrics[key])
        
        # Print epoch metrics
        tqdm.write(
            f'Loss: {train_metrics["loss"]:.3e}, {val_metrics["loss"]:.3e} | '
            f'Cls Loss: {train_metrics["cls_loss"]:.3e}, {val_metrics["cls_loss"]:.3e} | '
            f'Cluster Loss: {train_metrics["cluster_loss"]:.3e}, {val_metrics["cluster_loss"]:.3e} | '
            f'Accuracy: {train_metrics["accuracy"]:.4f}, {val_metrics["accuracy"]:.4f}'
        )
        
        # Save best model if validation improved
        if val_metrics['loss'] < best_val_loss or val_metrics['accuracy'] > best_val_acc:
            # Update best metrics
            best_val_loss = val_metrics['loss']
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0  # Reset patience counter
            
            # Create checkpoint dictionary
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'history': history,
                'config': config,
                'class_folders': train_dataset.class_folders
            }
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                config['checkpoint_dir'], 'best_model.pth'
            )
            torch.save(checkpoint, checkpoint_path)
            
            # Save detailed classification report
            report = classification_report(
                val_metrics['labels'],
                val_metrics['predictions'],
                target_names=train_dataset.class_folders,
                labels=list(range(len(train_dataset.class_folders))),
                output_dict=True,
                zero_division=0
            )
            report_path = os.path.join(
                config['checkpoint_dir'], 'best_model_report.json'
            )
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            # No improvement, increment patience counter
            patience_counter += 1
        
        # Early stopping: stop if no improvement for 'patience' epochs
        if patience_counter >= config['patience']:
            tqdm.write(f'Early stopping at epoch {epoch+1}')
            break
    
    # Save final model checkpoint (regardless of performance)
    final_checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'history': history,
        'config': config,
        'class_folders': train_dataset.class_folders
    }
    final_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
    torch.save(final_checkpoint, final_path)
    
    # Save training history as JSON
    history_path = os.path.join(
        config['checkpoint_dir'], 'training_history.json'
    )
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training history curves
    fig, ax = plt.subplots(nrows=4, figsize=(10, 15))
    for i, key in enumerate(['loss', 'cls_loss', 'cluster_loss', 'accuracy']):
        ax[i].plot(history['train'][key], label='Train')
        ax[i].plot(history['val'][key], label='Val')
        ax[i].set_ylabel(key)
        ax[i].legend()
        # Use log scale for loss metrics (not accuracy)
        if key != 'accuracy':
            ax[i].set_yscale('log')
    ax[-1].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(
        os.path.join(config['checkpoint_dir'], 'training_history.png')
    )
    plt.close()
    
    # Print final summary
    print(f'\nTraining completed!')
    print(f'Best validation loss: {best_val_loss:.3e}')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'Checkpoints saved to: {config["checkpoint_dir"]}')


if __name__ == '__main__':
    main()
