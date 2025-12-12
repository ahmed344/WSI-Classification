"""
Evaluation script for CLAM-MB model.

This module provides evaluation functionality for trained CLAM-MB models,
including accuracy calculation, confusion matrix generation, and detailed
classification reports.
"""
from typing import Dict, Any, List, Optional
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import matplotlib.pyplot as plt
import seaborn as sns

from clam_dataset import WSIFeatureDataset, collate_fn
from clam_model import CLAM_MB
from config_loader import load_config


def evaluate(
    model: CLAM_MB,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Evaluate the model and compute detailed metrics.
    
    Args:
        model (CLAM_MB): Trained CLAM-MB model in evaluation mode.
        dataloader (DataLoader): DataLoader providing batches of samples to evaluate.
        device (torch.device): Device to run evaluation on (CPU or CUDA).
        class_names (List[str]): List of class names for labeling metrics.
    
    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics:
            - 'accuracy' (float): Overall classification accuracy.
            - 'confusion_matrix' (np.ndarray): Confusion matrix of shape [n_classes, n_classes].
            - 'classification_report' (Dict[str, Any]): Detailed per-class metrics including
              precision, recall, F1-score, and support.
            - 'predictions' (List[int]): List of predicted class indices.
            - 'labels' (List[int]): List of true class indices.
            - 'probabilities' (List[List[float]]): List of prediction probabilities for each sample.
            - 'slide_names' (List[str]): List of slide names for each sample.
            - 'attention_weights' (List[np.ndarray]): List of attention weight arrays (if available).
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []
    all_slide_names: List[str] = []
    all_attention_weights: List[np.ndarray] = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in dataloader:
            # Move batch data to device
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['masks'].to(device)
            slide_names = batch['slide_names']
            
            # Forward pass through model
            outputs = model(features, masks)
            logits = outputs['logits']
            attention_weights = outputs['attention_weights']
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Collect predictions, labels, and probabilities
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_slide_names.extend(slide_names)
            
            # Store attention weights (average across branches for visualization)
            if attention_weights:
                # Average attention across all branches
                avg_attention = torch.stack(attention_weights).mean(dim=0).cpu().numpy()
                all_attention_weights.extend(avg_attention)
    
    # Calculate overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Generate confusion matrix
    cm = confusion_matrix(
        all_labels, all_preds, labels=list(range(len(class_names)))
    )
    
    # Generate detailed classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        labels=list(range(len(class_names))),
        output_dict=True,
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'slide_names': all_slide_names,
        'attention_weights': all_attention_weights
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm (np.ndarray): Confusion matrix of shape [n_classes, n_classes].
        class_names (List[str]): List of class names for axis labels.
        save_path (Optional[str]): Path to save the figure. If None, displays the plot.
            Defaults to None.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Confusion matrix saved to {save_path}')
    else:
        plt.show()
    plt.close()


def print_metrics(metrics: Dict[str, Any], class_names: List[str]) -> None:
    """
    Print evaluation metrics in a formatted format.
    
    Args:
        metrics (Dict[str, Any]): Dictionary containing evaluation metrics from evaluate().
        class_names (List[str]): List of class names for labeling.
    """
    print('\n' + '='*60)
    print('EVALUATION RESULTS')
    print('='*60)
    print(f'\nOverall Accuracy: {metrics["accuracy"]:.4f}')
    
    # Print per-class metrics
    print('\nPer-class Metrics:')
    print('-'*60)
    report = metrics['classification_report']
    
    for i, class_name in enumerate(class_names):
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            print(
                f'{class_name:20s} | Precision: {precision:.4f} | '
                f'Recall: {recall:.4f} | F1: {f1:.4f} | Support: {support}'
            )
    
    # Print macro averages
    print('\nMacro Average:')
    macro = report['macro avg']
    print(
        f'Precision: {macro["precision"]:.4f} | '
        f'Recall: {macro["recall"]:.4f} | '
        f'F1: {macro["f1-score"]:.4f}'
    )
    
    # Print weighted averages
    print('\nWeighted Average:')
    weighted = report['weighted avg']
    print(
        f'Precision: {weighted["precision"]:.4f} | '
        f'Recall: {weighted["recall"]:.4f} | '
        f'F1: {weighted["f1-score"]:.4f}'
    )
    
    print('='*60)


def main() -> None:
    """
    Main evaluation function.
    
    Loads model checkpoint, creates dataset, evaluates model performance,
    and saves results including confusion matrix and classification report.
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
    output_dir = config['paths']['evaluation_output']
    if output_dir is None:
        output_dir = os.path.join(config['output_dir'], 'evaluation_results')
    
    # Get evaluation parameters from config
    eval_config = config.get('evaluation', {})
    split = eval_config.get('split', 'val')
    plot_cm = eval_config.get('plot_cm', True)
    
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
    
    print(f'Number of samples: {len(dataset)}')
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Evaluate model
    print('Evaluating model...')
    metrics = evaluate(model, dataloader, device, class_folders)
    
    # Print metrics to console
    print_metrics(metrics, class_folders)
    
    # Save results as JSON (convert numpy types to native Python types)
    results = {
        'accuracy': float(metrics['accuracy']),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'classification_report': metrics['classification_report'],
        'predictions': [int(p) for p in metrics['predictions']],
        'labels': [int(l) for l in metrics['labels']],
        'slide_names': metrics['slide_names']
    }
    
    results_path = os.path.join(output_dir, f'evaluation_{split}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {results_path}')
    
    # Plot and save confusion matrix if requested
    if plot_cm:
        cm_path = os.path.join(output_dir, f'confusion_matrix_{split}.png')
        plot_confusion_matrix(metrics['confusion_matrix'], class_folders, cm_path)
    
    # Save per-slide predictions with probabilities
    slide_results: List[Dict[str, Any]] = []
    for i, slide_name in enumerate(metrics['slide_names']):
        slide_results.append({
            'slide_name': slide_name,
            'true_label': int(metrics['labels'][i]),
            'predicted_label': int(metrics['predictions'][i]),
            'true_class': class_folders[metrics['labels'][i]],
            'predicted_class': class_folders[metrics['predictions'][i]],
            'probabilities': {
                class_folders[j]: float(metrics['probabilities'][i][j])
                for j in range(len(class_folders))
            }
        })
    
    slide_results_path = os.path.join(
        output_dir, f'slide_predictions_{split}.json'
    )
    with open(slide_results_path, 'w') as f:
        json.dump(slide_results, f, indent=2)
    print(f'Per-slide predictions saved to {slide_results_path}')


if __name__ == '__main__':
    main()
