import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
from datetime import datetime

from clam_dataset import WSIFeatureDataset, collate_fn
from clam_model import CLAM_MB, compute_clustering_loss
from config_loader import load_config


def train_epoch(model, dataloader, criterion, optimizer, device, clustering_weight=0.1):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_cluster_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        masks = batch['masks'].to(device)
        
        # Forward pass
        outputs = model(features, masks)
        logits = outputs['logits']
        cluster_assignments = outputs['cluster_assignments']
        
        # Classification loss
        cls_loss = criterion(logits, labels)
        
        # Clustering loss
        cluster_loss = compute_clustering_loss(cluster_assignments, masks)
        
        # Total loss
        total_loss = cls_loss + clustering_weight * cluster_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += total_loss.item()
        running_cls_loss += cls_loss.item()
        running_cluster_loss += cluster_loss.item()
        
        # Predictions
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / len(dataloader):.4f}',
            'cls': f'{running_cls_loss / len(dataloader):.4f}',
            'cluster': f'{running_cluster_loss / len(dataloader):.4f}'
        })
    
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


def validate(model, dataloader, criterion, device, clustering_weight=0.1):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_cluster_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['masks'].to(device)
            
            # Forward pass
            outputs = model(features, masks)
            logits = outputs['logits']
            cluster_assignments = outputs['cluster_assignments']
            
            # Classification loss
            cls_loss = criterion(logits, labels)
            
            # Clustering loss
            cluster_loss = compute_clustering_loss(cluster_assignments, masks)
            
            # Total loss
            total_loss = cls_loss + clustering_weight * cluster_loss
            
            # Statistics
            running_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_cluster_loss += cluster_loss.item()
            
            # Predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / len(dataloader):.4f}',
                'acc': f'{accuracy_score(all_labels, all_preds):.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_cls_loss = running_cls_loss / len(dataloader)
    epoch_cluster_loss = running_cluster_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Confusion matrix
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


def main():
    # Load configuration
    config = load_config()
    
    # Set random seed
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Create datasets
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
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid pickle issues
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    print('Creating model...')
    model = CLAM_MB(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        k_clusters=config['k_clusters']
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train': {'loss': [], 'cls_loss': [], 'cluster_loss': [], 'accuracy': []},
        'val': {'loss': [], 'cls_loss': [], 'cluster_loss': [], 'accuracy': []}
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print('Starting training...')
    for epoch in range(config['epochs']):
        print(f'\nEpoch {epoch+1}/{config["epochs"]}')
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, config['clustering_weight']
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, config['clustering_weight']
        )
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Save history
        for key in ['loss', 'cls_loss', 'cluster_loss', 'accuracy']:
            history['train'][key].append(train_metrics[key])
            history['val'][key].append(val_metrics[key])
        
        # Print metrics
        print(f'Train - Loss: {train_metrics["loss"]:.4f}, '
              f'Cls Loss: {train_metrics["cls_loss"]:.4f}, '
              f'Cluster Loss: {train_metrics["cluster_loss"]:.4f}, '
              f'Accuracy: {train_metrics["accuracy"]:.4f}')
        print(f'Val   - Loss: {val_metrics["loss"]:.4f}, '
              f'Cls Loss: {val_metrics["cls_loss"]:.4f}, '
              f'Cluster Loss: {val_metrics["cluster_loss"]:.4f}, '
              f'Accuracy: {val_metrics["accuracy"]:.4f}')
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
                'config': config,
                'class_folders': train_dataset.class_folders
            }
            
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'Saved best model (val_acc: {best_val_acc:.4f})')
            
            # Save classification report
            report = classification_report(
                val_metrics['labels'],
                val_metrics['predictions'],
                target_names=train_dataset.class_folders,
                labels=list(range(len(train_dataset.class_folders))),
                output_dict=True,
                zero_division=0
            )
            report_path = os.path.join(config['checkpoint_dir'], 'best_model_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Save final model and history
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
    
    # Save training history
    history_path = os.path.join(config['checkpoint_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'Checkpoints saved to: {config["checkpoint_dir"]}')


if __name__ == '__main__':
    main()

