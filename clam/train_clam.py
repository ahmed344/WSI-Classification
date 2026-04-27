"""
Training script for CLAM-MB (Multi-Branch) model.

This module provides training functionality for the CLAM-MB model using a mixed
loss function combining classification loss and clustering loss. It includes
early stopping, learning rate scheduling, and checkpoint saving.
"""
from typing import Dict, Any, List, Tuple
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)
import json

from clam_dataset import WSIFeatureDataset, collate_fn
from clam_model import CLAM_MB, compute_clustering_loss
from config_loader import load_config, resolve_feature_file_suffix

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


def get_class_sample_counts(dataset: WSIFeatureDataset) -> Dict[str, int]:
    """
    Calculate sample counts per class for a dataset split.

    Args:
        dataset (WSIFeatureDataset): Dataset instance containing split indices and tissue metadata.

    Returns:
        Dict[str, int]: Mapping from class name to number of samples in the dataset split.
    """
    class_counts: Dict[str, int] = {class_name: 0 for class_name in dataset.class_folders}
    for tissue_idx in dataset.indices:
        class_name = dataset.tissues[tissue_idx]['class']
        class_counts[class_name] += 1
    return class_counts


def compute_class_weights(dataset: WSIFeatureDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training split.

    Args:
        dataset (WSIFeatureDataset): Training dataset containing class names and split indices.

    Returns:
        torch.Tensor: Float tensor of shape [num_classes] ordered by
            dataset.class_folders, where larger values correspond to rarer classes.
    """
    class_counts = get_class_sample_counts(dataset)
    total_samples = sum(class_counts.values())
    num_classes = len(dataset.class_folders)
    class_weights: List[float] = []
    for class_name in dataset.class_folders:
        count = class_counts[class_name]
        weight = total_samples / (num_classes * count) if count > 0 else 0.0
        class_weights.append(weight)
    return torch.tensor(class_weights, dtype=torch.float32)


def compute_sample_weights(
    dataset: WSIFeatureDataset,
    class_weights: torch.Tensor
) -> List[float]:
    """
    Compute per-sample weights for weighted random sampling.

    Args:
        dataset (WSIFeatureDataset): Dataset containing tissue metadata and split indices.
        class_weights (torch.Tensor): Class-weight tensor of shape [num_classes]
            aligned with dataset.class_folders.

    Returns:
        List[float]: Per-sample weights aligned with dataset.indices for use in
            WeightedRandomSampler.
    """
    class_weight_map: Dict[str, float] = {
        class_name: float(class_weights[class_idx].item())
        for class_idx, class_name in enumerate(dataset.class_folders)
    }
    sample_weights: List[float] = []
    for tissue_idx in dataset.indices:
        class_name = dataset.tissues[tissue_idx]['class']
        sample_weights.append(class_weight_map[class_name])
    return sample_weights


def get_clustering_weight_for_epoch(
    epoch_idx: int,
    weight_start: float,
    weight_end: float,
    warmup_epochs: int
) -> float:
    """
    Compute clustering loss weight for a given epoch using linear warmup.

    Args:
        epoch_idx (int): Zero-based epoch index.
        weight_start (float): Initial clustering weight used at first epoch.
        weight_end (float): Final clustering weight after warmup.
        warmup_epochs (int): Number of epochs for linear ramp-up. If <= 1,
            the function returns weight_end directly.

    Returns:
        float: Clustering weight value to use for the specified epoch.
    """
    if warmup_epochs <= 1:
        return float(weight_end)
    progress = min(max(epoch_idx, 0), warmup_epochs - 1) / (warmup_epochs - 1)
    return float(weight_start + progress * (weight_end - weight_start))


def resolve_best_checkpoint_metric(
    metric_name: str
) -> Tuple[str, bool, str]:
    """
    Resolve checkpoint metric config into validation key and optimization direction.

    Args:
        metric_name (str): User-selected checkpoint metric name from config.

    Returns:
        Tuple[str, bool, str]: Tuple containing:
            - validation metric key used in `val_metrics`
            - boolean flag indicating whether metric should be maximized
            - normalized config metric name
    """
    normalized_name = str(metric_name).strip().lower()
    metric_mapping: Dict[str, Tuple[str, bool, str]] = {
        'balanced_accuracy': ('balanced_accuracy', True, 'balanced_accuracy'),
        'accuracy': ('accuracy', True, 'accuracy'),
        'loss': ('loss', False, 'loss'),
        'classification_loss': ('cls_loss', False, 'classification_loss'),
        'clustering_loss': ('cluster_loss', False, 'clustering_loss')
    }
    if normalized_name not in metric_mapping:
        valid_metrics = ', '.join(metric_mapping.keys())
        raise ValueError(
            f"Invalid best_checkpoint_metric '{metric_name}'. "
            f"Valid options are: {valid_metrics}."
        )
    return metric_mapping[normalized_name]


def train_epoch(
    model: CLAM_MB,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer_cls: optim.Optimizer,
    optimizer_cluster: optim.Optimizer,
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
        optimizer_cls (optim.Optimizer): Optimizer for updating classification branch parameters.
        optimizer_cluster (optim.Optimizer): Optimizer for updating clustering branch parameters.
        device (torch.device): Device to run training on (CPU or CUDA).
        clustering_weight (float): Weight for clustering loss component. Defaults to 0.1.
    
    Returns:
        Dict[str, float]: Dictionary containing training metrics:
            - 'loss' (float): Average total loss for the epoch.
            - 'cls_loss' (float): Average classification loss for the epoch.
            - 'cluster_loss' (float): Average clustering loss for the epoch.
            - 'accuracy' (float): Classification accuracy for the epoch.
            - 'balanced_accuracy' (float): Balanced accuracy for the epoch.
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
        optimizer_cls.zero_grad()
        optimizer_cluster.zero_grad()
        total_loss.backward()
        optimizer_cls.step()
        optimizer_cluster.step()
        
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
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    
    return {
        'loss': epoch_loss,
        'cls_loss': epoch_cls_loss,
        'cluster_loss': epoch_cluster_loss,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy
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
            - 'balanced_accuracy' (float): Balanced accuracy for the epoch.
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
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    
    # Compute confusion matrix for detailed analysis
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': epoch_loss,
        'cls_loss': epoch_cls_loss,
        'cluster_loss': epoch_cluster_loss,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
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
    feature_file_suffix = resolve_feature_file_suffix(config)
    
    # Determine device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Create training and validation datasets
    print('Loading datasets...')
    print(
        f"Using feature model '{config['feature_model']}' "
        f"with suffix '{feature_file_suffix}'"
    )
    train_dataset = WSIFeatureDataset(
        config['data_root'],
        split='train',
        train_ratio=config['train_ratio'],
        random_seed=config['random_seed'],
        feature_file_suffix=feature_file_suffix
    )
    val_dataset = WSIFeatureDataset(
        config['data_root'],
        split='val',
        train_ratio=config['train_ratio'],
        random_seed=config['random_seed'],
        feature_file_suffix=feature_file_suffix
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Classes: {train_dataset.class_folders}')
    train_class_counts = get_class_sample_counts(train_dataset)
    val_class_counts = get_class_sample_counts(val_dataset)
    print('Training samples per class:')
    for class_name in train_dataset.class_folders:
        print(f'  {class_name}: {train_class_counts[class_name]}')
    print('Validation samples per class:')
    for class_name in train_dataset.class_folders:
        print(f'  {class_name}: {val_class_counts[class_name]}')
    
    # Create data loaders
    use_weighted_sampler = bool(config.get('use_weighted_sampler', True))
    use_class_weighted_loss = bool(config.get('use_class_weighted_loss', False))
    (
        best_checkpoint_metric_key,
        maximize_best_checkpoint_metric,
        best_checkpoint_metric_name
    ) = resolve_best_checkpoint_metric(
        config.get('best_checkpoint_metric', 'balanced_accuracy')
    )
    class_weights = compute_class_weights(train_dataset)
    if use_weighted_sampler:
        sample_weights = compute_sample_weights(train_dataset, class_weights)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 to avoid pickle issues with custom dataset
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
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
    checkpoint_metric_direction = (
        'max' if maximize_best_checkpoint_metric else 'min'
    )
    print(f'Weighted sampler enabled: {use_weighted_sampler}')
    print(f'Class-weighted CE loss enabled: {use_class_weighted_loss}')
    print(
        'Best checkpoint metric: '
        f'{best_checkpoint_metric_name} ({checkpoint_metric_direction})'
    )
    
    # Create model with configuration parameters
    print('Creating model...')
    model = CLAM_MB(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        k_clusters=config['k_clusters'],
        attention_hidden_dim=config.get('attention_hidden_dim'),
        attention_dim=config.get('attention_dim'),
        cluster_head_hidden_dim=config.get('cluster_head_hidden_dim'),
        feature_projection_dropout=config['feature_projection_dropout'],
        attention_branch_feature_dropout=config['attention_branch_feature_dropout'],
        clustering_branch_feature_dropout=config['clustering_branch_feature_dropout'],
        final_classifier_dropout=config['final_classifier_dropout']
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Setup loss function, optimizer, and learning rate scheduler
    if use_class_weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    lr_cls = config.get('lr_cls', config.get('learning_rate', 1e-4))
    lr_cluster = config.get('lr_cluster', lr_cls)
    cls_params = (
        list(model.feature_projection.parameters())
        + list(model.attention_branches.parameters())
        + list(model.classifier.parameters())
    )
    cluster_params = list(model.clustering_branch.parameters())
    weight_decay_cls = config['weight_decay_cls']
    weight_decay_cluster = config['weight_decay_cluster']
    optimizer_cls = optim.Adam(
        [
            {
                'params': cls_params,
                'lr': lr_cls,
                'weight_decay': weight_decay_cls
            }
        ]
    )
    optimizer_cluster = optim.Adam(
        [
            {
                'params': cluster_params,
                'lr': lr_cluster,
                'weight_decay': weight_decay_cluster
            }
        ]
    )
    # Reduce branch-specific learning rates when their validation losses plateau.
    scheduler_cls = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_cls,
        mode='min',
        factor=float(config.get('lr_scheduler_factor_cls', 0.5)),
        patience=int(config.get('lr_scheduler_patience_cls', 5))
    )
    scheduler_cluster = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_cluster,
        mode='min',
        factor=float(config.get('lr_scheduler_factor_cluster', 0.5)),
        patience=int(config.get('lr_scheduler_patience_cluster', 5))
    )
    
    # Initialize training history tracking
    history: Dict[str, Dict[str, List[float]]] = {
        'train': {
            'loss': [], 'cls_loss': [], 'cluster_loss': [],
            'accuracy': [], 'balanced_accuracy': []
        },
        'val': {
            'loss': [], 'cls_loss': [], 'cluster_loss': [],
            'accuracy': [], 'balanced_accuracy': []
        }
    }
    
    # Initialize best validation metrics and early stopping counter
    best_val_cls_loss = float('inf')
    best_val_balanced_acc = 0.0
    if maximize_best_checkpoint_metric:
        best_val_checkpoint_metric = -float('inf')
    else:
        best_val_checkpoint_metric = float('inf')
    best_epoch = 1
    patience_counter = 0
    
    # Training loop
    print('Starting training...')
    cluster_weight_start = config.get(
        'clustering_weight_start',
        config.get('clustering_weight', 0.1)
    )
    cluster_weight_end = config.get(
        'clustering_weight_end',
        config.get('clustering_weight', 0.1)
    )
    cluster_warmup_epochs = config.get('clustering_warmup_epochs', 1)
    min_epochs_before_early_stopping = max(
        0, int(config.get('min_epochs_before_early_stopping', 0))
    )
    previous_lrs = {
        'cls': optimizer_cls.param_groups[0]['lr'],
        'cluster': optimizer_cluster.param_groups[0]['lr']
    }

    for epoch in tqdm(range(config['epochs'])):
        clustering_weight = get_clustering_weight_for_epoch(
            epoch_idx=epoch,
            weight_start=cluster_weight_start,
            weight_end=cluster_weight_end,
            warmup_epochs=cluster_warmup_epochs
        )
        # Train for one epoch
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer_cls, optimizer_cluster, device,
            clustering_weight
        )
        
        # Validate for one epoch
        val_metrics = validate(
            model, val_loader, criterion, device, clustering_weight
        )
        
        # Update branch-specific learning rates based on branch losses.
        scheduler_cls.step(val_metrics['cls_loss'])
        scheduler_cluster.step(val_metrics['cluster_loss'])
        
        # Save metrics to history
        for key in ['loss', 'cls_loss', 'cluster_loss', 'accuracy', 'balanced_accuracy']:
            history['train'][key].append(train_metrics[key])
            history['val'][key].append(val_metrics[key])
        
        # Print epoch metrics
        tqdm.write(
            f'Loss: {train_metrics["loss"]:.3e}, {val_metrics["loss"]:.3e} | '
            f'Cls Loss: {train_metrics["cls_loss"]:.3e}, {val_metrics["cls_loss"]:.3e} | '
            f'Cluster Loss: {train_metrics["cluster_loss"]:.3e}, {val_metrics["cluster_loss"]:.3e} | '
            f'Accuracy: {train_metrics["accuracy"]:.4f}, {val_metrics["accuracy"]:.4f} | '
            f'Bal Acc: {train_metrics["balanced_accuracy"]:.4f}, '
            f'{val_metrics["balanced_accuracy"]:.4f}'
        )
        current_lrs = {
            'cls': optimizer_cls.param_groups[0]['lr'],
            'cluster': optimizer_cluster.param_groups[0]['lr']
        }
        if (
            not np.isclose(current_lrs['cls'], previous_lrs['cls'])
            or not np.isclose(current_lrs['cluster'], previous_lrs['cluster'])
        ):
            tqdm.write(
                f'Learning rates updated: lr_cls: {current_lrs["cls"]:.2e}, '
                f'lr_cluster: {current_lrs["cluster"]:.2e}'
            )
        previous_lrs = current_lrs
        
        # Save best model using selected checkpoint metric and direction.
        current_checkpoint_metric = float(val_metrics[best_checkpoint_metric_key])
        if maximize_best_checkpoint_metric:
            is_better_metric = (
                current_checkpoint_metric > best_val_checkpoint_metric
            )
        else:
            is_better_metric = (
                current_checkpoint_metric < best_val_checkpoint_metric
            )
        is_tie_on_metric = np.isclose(
            current_checkpoint_metric, best_val_checkpoint_metric
        )
        if is_tie_on_metric:
            is_better_metric = val_metrics['cls_loss'] < best_val_cls_loss

        if is_better_metric:
            # Update best validation metrics
            best_val_cls_loss = val_metrics['cls_loss']
            best_val_balanced_acc = val_metrics['balanced_accuracy']
            best_val_checkpoint_metric = current_checkpoint_metric
            best_epoch = epoch + 1
            patience_counter = 0  # Reset patience counter
            
            # Create checkpoint dictionary
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': {
                    'cls': optimizer_cls.state_dict(),
                    'cluster': optimizer_cluster.state_dict()
                },
                'optimizer_cls_state_dict': optimizer_cls.state_dict(),
                'optimizer_cluster_state_dict': optimizer_cluster.state_dict(),
                'best_val_cls_loss': best_val_cls_loss,
                'best_val_balanced_acc': best_val_balanced_acc,
                'best_checkpoint_metric': best_checkpoint_metric_name,
                'best_checkpoint_metric_key': best_checkpoint_metric_key,
                'best_checkpoint_metric_mode': checkpoint_metric_direction,
                'best_checkpoint_metric_value': best_val_checkpoint_metric,
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
        
        # Early stopping: only active after minimum epoch threshold is reached.
        if (
            (epoch + 1) >= min_epochs_before_early_stopping
            and patience_counter >= config['patience']
        ):
            tqdm.write(f'Early stopping at epoch {epoch+1}')
            break
    
    # Save final model checkpoint (regardless of performance)
    final_checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {
            'cls': optimizer_cls.state_dict(),
            'cluster': optimizer_cluster.state_dict()
        },
        'optimizer_cls_state_dict': optimizer_cls.state_dict(),
        'optimizer_cluster_state_dict': optimizer_cluster.state_dict(),
        'best_val_cls_loss': best_val_cls_loss,
        'best_val_balanced_acc': best_val_balanced_acc,
        'best_checkpoint_metric': best_checkpoint_metric_name,
        'best_checkpoint_metric_key': best_checkpoint_metric_key,
        'best_checkpoint_metric_mode': checkpoint_metric_direction,
        'best_checkpoint_metric_value': best_val_checkpoint_metric,
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
    plot_keys = ['loss', 'cls_loss', 'cluster_loss', 'accuracy', 'balanced_accuracy']
    fig, ax = plt.subplots(nrows=len(plot_keys), figsize=(10, 18))
    train_handle = None
    val_handle = None
    best_epoch_handle = None
    for i, key in enumerate(plot_keys):
        train_line, = ax[i].plot(history['train'][key], label='Train')
        val_line, = ax[i].plot(history['val'][key], label='Val')
        best_line = ax[i].axvline(
            x=best_epoch - 1,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label='Best model epoch'
        )
        if i == 0:
            train_handle = train_line
            val_handle = val_line
            best_epoch_handle = best_line
        ax[i].set_ylabel(key)
        # Use log scale for loss metrics (not accuracy)
        if key in ['loss', 'cls_loss', 'cluster_loss']:
            ax[i].set_yscale('log')
    ax[-1].set_xlabel('Epoch')
    fig.legend(
        [train_handle, val_handle, best_epoch_handle],
        ['Train', 'Val', 'Best model epoch'],
        loc='upper center',
        ncol=3,
        fontsize=12,
        frameon=True
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        os.path.join(config['checkpoint_dir'], 'training_history.png')
    )
    plt.close()
    
    # Print final summary
    print(f'\nTraining completed!')
    print(f'Best validation classification loss: {best_val_cls_loss:.3e}')
    print(f'Best validation balanced accuracy: {best_val_balanced_acc:.4f}')
    print(
        'Best validation checkpoint metric '
        f'({best_checkpoint_metric_name}): {best_val_checkpoint_metric:.6f}'
    )
    print(f'Checkpoints saved to: {config["checkpoint_dir"]}')


if __name__ == '__main__':
    main()
