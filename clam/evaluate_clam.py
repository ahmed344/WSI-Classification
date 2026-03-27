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

from clam_dataset import WSIFeatureDataset, WSISlideBagDataset, collate_fn
from clam_model import CLAM_MB
from config_loader import load_config, resolve_feature_file_suffix


def get_class_sample_counts(dataset: Any) -> Dict[str, int]:
    """
    Calculate sample counts per class for a dataset split.

    Args:
        dataset (Any): Dataset instance containing split indices and either tissue
            metadata (`tissues`) or slide metadata (`slides`).

    Returns:
        Dict[str, int]: Mapping from class name to number of samples in the dataset split.
    """
    class_counts: Dict[str, int] = {class_name: 0 for class_name in dataset.class_folders}
    if hasattr(dataset, 'tissues'):
        for sample_idx in dataset.indices:
            class_name = dataset.tissues[sample_idx]['class']
            class_counts[class_name] += 1
        return class_counts

    if hasattr(dataset, 'slides'):
        for sample_idx in dataset.indices:
            class_name = dataset.slides[sample_idx]['class']
            class_counts[class_name] += 1
        return class_counts

    raise AttributeError(
        "Dataset must provide either 'tissues' or 'slides' metadata for class counting."
    )


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
            - 'tissue_names' (List[str]): List of tissue names for each sample.
            - 'attention_weights' (List[np.ndarray]): List of attention weight arrays (if available).
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []
    all_slide_names: List[str] = []
    all_tissue_names: List[str] = []
    all_attention_weights: List[np.ndarray] = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in dataloader:
            # Move batch data to device
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['masks'].to(device)
            slide_names = batch['slide_names']
            tissue_names = batch['tissue_names']
            
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
            all_tissue_names.extend(tissue_names)
            
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
        'tissue_names': all_tissue_names,
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
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        annot_kws={'size': 20}
    )
    ax.tick_params(axis='both', labelsize=14)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Actual', fontsize=16)
    plt.title('Confusion Matrix', fontsize=18)
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


def create_dataset_for_level(
    level: str,
    data_root: str,
    class_folders: List[str],
    split: str,
    train_ratio: float,
    random_seed: int,
    feature_file_suffix: str
) -> Any:
    """
    Create a dataset for the selected evaluation level and split.

    Args:
        level (str): Evaluation level name (`'tissue'` or `'slide'`).
        data_root (str): Root directory containing class folders.
        class_folders (List[str]): Ordered class folder names.
        split (str): Dataset split (`'train'` or `'val'`).
        train_ratio (float): Ratio of slides assigned to training split.
        random_seed (int): Seed for reproducible split generation.
        feature_file_suffix (str): Feature suffix used to select input files.

    Returns:
        Any: Instantiated dataset object for the requested level.
    """
    if level == 'tissue':
        return WSIFeatureDataset(
            data_root=data_root,
            class_folders=class_folders,
            split=split,
            train_ratio=train_ratio,
            random_seed=random_seed,
            feature_file_suffix=feature_file_suffix
        )
    if level == 'slide':
        return WSISlideBagDataset(
            data_root=data_root,
            class_folders=class_folders,
            split=split,
            train_ratio=train_ratio,
            random_seed=random_seed,
            feature_file_suffix=feature_file_suffix
        )
    raise ValueError(f"Invalid level '{level}'. Expected 'tissue' or 'slide'.")


def save_level_results(
    metrics: Dict[str, Any],
    class_folders: List[str],
    output_dir: str,
    level: str,
    split: str
) -> None:
    """
    Save evaluation summaries, confusion matrix, and per-sample predictions.

    Args:
        metrics (Dict[str, Any]): Output dictionary returned by `evaluate`.
        class_folders (List[str]): Ordered class names.
        output_dir (str): Directory where artifacts are written.
        level (str): Evaluation level label (`'tissue'` or `'slide'`).
        split (str): Dataset split label (`'train'` or `'val'`).

    Returns:
        None: This function writes files to disk.
    """
    results = {
        'accuracy': float(metrics['accuracy']),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'classification_report': metrics['classification_report'],
        'predictions': [int(p) for p in metrics['predictions']],
        'labels': [int(l) for l in metrics['labels']],
        'slide_names': metrics['slide_names'],
        'tissue_names': metrics['tissue_names']
    }
    results_path = os.path.join(output_dir, f'{level}_evaluation_{split}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {results_path}')

    predictions: List[Dict[str, Any]] = []
    for i, slide_name in enumerate(metrics['slide_names']):
        predictions.append({
            'slide_name': slide_name,
            'tissue_name': metrics['tissue_names'][i],
            'true_label': int(metrics['labels'][i]),
            'predicted_label': int(metrics['predictions'][i]),
            'true_class': class_folders[metrics['labels'][i]],
            'predicted_class': class_folders[metrics['predictions'][i]],
            'probabilities': {
                class_folders[j]: float(metrics['probabilities'][i][j])
                for j in range(len(class_folders))
            }
        })
    predictions_path = os.path.join(output_dir, f'{level}_predictions_{split}.json')
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f'Predictions saved to {predictions_path}')

    cm_path = os.path.join(output_dir, f'{level}_confusion_matrix_{split}.png')
    plot_confusion_matrix(metrics['confusion_matrix'], class_folders, cm_path)


def run_level_split_evaluation(
    model: CLAM_MB,
    config: Dict[str, Any],
    class_folders: List[str],
    feature_file_suffix: str,
    device: torch.device,
    output_dir: str,
    level: str,
    split: str
) -> None:
    """
    Run one evaluation job for a level/split pair and persist artifacts.

    Args:
        model (CLAM_MB): Loaded CLAM model in evaluation mode.
        config (Dict[str, Any]): Global configuration dictionary.
        class_folders (List[str]): Ordered class names for label decoding.
        feature_file_suffix (str): Feature suffix selected in config.
        device (torch.device): Evaluation device.
        output_dir (str): Evaluation output directory.
        level (str): Evaluation level (`'tissue'` or `'slide'`).
        split (str): Dataset split (`'train'` or `'val'`).

    Returns:
        None: This function runs evaluation and writes artifacts to disk.
    """
    print('\n' + '-' * 60)
    print(f'Running {level} evaluation on {split} split...')
    dataset = create_dataset_for_level(
        level=level,
        data_root=config['data_root'],
        class_folders=class_folders,
        split=split,
        train_ratio=config['train_ratio'],
        random_seed=config['random_seed'],
        feature_file_suffix=feature_file_suffix
    )
    class_counts = get_class_sample_counts(dataset)
    print(f'Number of {level} samples ({split}): {len(dataset)}')
    print(f'{split.capitalize()} samples per class:')
    for class_name in class_folders:
        print(f'  {class_name}: {class_counts[class_name]}')

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    metrics = evaluate(model, dataloader, device, class_folders)
    print_metrics(metrics, class_folders)
    save_level_results(metrics, class_folders, output_dir, level, split)


def main() -> None:
    """
    Main evaluation function.
    
    Loads model checkpoint, creates dataset, evaluates model performance,
    and saves results including confusion matrix and classification report.
    """
    # Load configuration from config.yml
    config = load_config()
    feature_file_suffix = resolve_feature_file_suffix(config)
    
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
    
    # Require new-format checkpoints that contain full model config.
    if 'config' not in checkpoint:
        raise KeyError(
            "Checkpoint is missing 'config'. "
            "Only new-format checkpoints are supported."
        )
    model_config = checkpoint['config']
    input_dim = model_config.get('input_dim', config['input_dim'])
    hidden_dim = model_config.get('hidden_dim', config['hidden_dim'])
    num_classes = model_config.get('num_classes', config['num_classes'])
    k_clusters = model_config.get('k_clusters', config['k_clusters'])
    architecture_source = model_config
    class_folders = checkpoint.get('class_folders', None)
    
    # Auto-detect class folders if not in checkpoint
    if class_folders is None:
        class_folders = sorted([
            d for d in os.listdir(config['data_root'])
            if os.path.isdir(os.path.join(config['data_root'], d))
        ])
    
    print(f'Classes: {class_folders}')
    print(
        f"Using feature model '{config['feature_model']}' "
        f"with suffix '{feature_file_suffix}'"
    )
    
    # Create model with extracted configuration
    print('Creating model...')
    model_kwargs: Dict[str, Any] = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'k_clusters': k_clusters
    }
    for key in [
        'attention_hidden_dim',
        'attention_dim',
        'cluster_head_hidden_dim'
    ]:
        value = architecture_source.get(key)
        if value is None:
            value = config.get(key)
        if value is not None:
            model_kwargs[key] = value

    dropout_keys = [
        'feature_projection_dropout',
        'attention_branch_feature_dropout',
        'clustering_branch_feature_dropout',
        'final_classifier_dropout'
    ]
    for key in dropout_keys:
        value = architecture_source.get(key)
        if value is None:
            value = config.get(key)
        if value is not None:
            model_kwargs[key] = value
    model = CLAM_MB(**model_kwargs).to(device)
    
    # Load model weights from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded successfully')
    
    # Run both levels on both splits.
    for split in ['train', 'val']:
        for level in ['tissue', 'slide']:
            run_level_split_evaluation(
                model=model,
                config=config,
                class_folders=class_folders,
                feature_file_suffix=feature_file_suffix,
                device=device,
                output_dir=output_dir,
                level=level,
                split=split
            )


if __name__ == '__main__':
    main()
