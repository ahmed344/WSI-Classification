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


def evaluate(model, dataloader, device, class_names):
    """Evaluate the model and return detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_slide_names = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['masks'].to(device)
            slide_names = batch['slide_names']
            
            # Forward pass
            outputs = model(features, masks)
            logits = outputs['logits']
            attention_weights = outputs['attention_weights']
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_slide_names.extend(slide_names)
            
            # Store attention weights (average across branches for visualization)
            if attention_weights:
                avg_attention = torch.stack(attention_weights).mean(dim=0).cpu().numpy()
                all_attention_weights.extend(avg_attention)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
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


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
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


def print_metrics(metrics, class_names):
    """Print evaluation metrics"""
    print('\n' + '='*60)
    print('EVALUATION RESULTS')
    print('='*60)
    print(f'\nOverall Accuracy: {metrics["accuracy"]:.4f}')
    
    print('\nPer-class Metrics:')
    print('-'*60)
    report = metrics['classification_report']
    
    for i, class_name in enumerate(class_names):
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            print(f'{class_name:20s} | Precision: {precision:.4f} | '
                  f'Recall: {recall:.4f} | F1: {f1:.4f} | Support: {support}')
    
    print('\nMacro Average:')
    macro = report['macro avg']
    print(f'Precision: {macro["precision"]:.4f} | '
          f'Recall: {macro["recall"]:.4f} | '
          f'F1: {macro["f1-score"]:.4f}')
    
    print('\nWeighted Average:')
    weighted = report['weighted avg']
    print(f'Precision: {weighted["precision"]:.4f} | '
          f'Recall: {weighted["recall"]:.4f} | '
          f'F1: {weighted["f1-score"]:.4f}')
    
    print('='*60)


def main():
    # Load configuration
    config = load_config()
    
    # Get checkpoint path
    checkpoint_path = config['paths']['checkpoint']
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
    
    # Get output directory
    output_dir = config['paths']['evaluation_output']
    if output_dir is None:
        output_dir = os.path.join(config['output_dir'], 'evaluation_results')
    
    # Get evaluation parameters
    eval_config = config.get('evaluation', {})
    split = eval_config.get('split', 'val')
    plot_cm = eval_config.get('plot_cm', True)
    
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
    
    # Get class names
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
    
    # Load model weights
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
    
    print(f'Number of samples: {len(dataset)}')
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Evaluate
    print('Evaluating model...')
    metrics = evaluate(model, dataloader, device, class_folders)
    
    # Print metrics
    print_metrics(metrics, class_folders)
    
    # Save results (convert numpy types to native Python types for JSON serialization)
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
    
    # Plot confusion matrix
    if plot_cm:
        cm_path = os.path.join(output_dir, f'confusion_matrix_{split}.png')
        plot_confusion_matrix(metrics['confusion_matrix'], class_folders, cm_path)
    
    # Save per-slide predictions
    slide_results = []
    for i, slide_name in enumerate(metrics['slide_names']):
        slide_results.append({
            'slide_name': slide_name,
            'true_label': int(metrics['labels'][i]),
            'predicted_label': int(metrics['predictions'][i]),
            'true_class': class_folders[metrics['labels'][i]],
            'predicted_class': class_folders[metrics['predictions'][i]],
            'probabilities': {class_folders[j]: float(metrics['probabilities'][i][j])
                             for j in range(len(class_folders))}
        })
    
    slide_results_path = os.path.join(output_dir, f'slide_predictions_{split}.json')
    with open(slide_results_path, 'w') as f:
        json.dump(slide_results, f, indent=2)
    print(f'Per-slide predictions saved to {slide_results_path}')


if __name__ == '__main__':
    main()

