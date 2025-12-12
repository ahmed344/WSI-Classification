"""
Dataset module for loading WSI features and tiles for CLAM-MB training.

This module provides WSIFeatureDataset class for loading tissue-level features
from PyTorch .pt files and tile coordinates from CSV files. The dataset supports
slide-level stratification to ensure all tissues from the same slide are in the
same train/validation split.
"""
import os
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import defaultdict


class WSIFeatureDataset(Dataset):
    """
    Dataset for loading WSI features from .pt files and tiles from .csv files.
    
    Each sample is a tissue (bag) containing multiple tile features. The dataset
    supports slide-level stratification, ensuring all tissues from the same slide
    are assigned to the same train/validation split.
    
    Directory structure: processed/category/slide/tissue_name_features.pt
    """
    
    def __init__(
        self,
        data_root: str,
        class_folders: Optional[List[str]] = None,
        split: str = 'train',
        train_ratio: float = 0.9,
        random_seed: int = 42
    ) -> None:
        """
        Initialize the WSI Feature Dataset.
        
        Args:
            data_root (str): Root directory containing class folders (e.g., 'Dystrophic', 'Healthy').
            class_folders (Optional[List[str]]): List of class folder names. If None, auto-detect
                from data_root by scanning for directories.
            split (str): Dataset split to use. Must be 'train' or 'val'. Defaults to 'train'.
            train_ratio (float): Ratio of slides to use for training. Remaining slides go to validation.
                Defaults to 0.9.
            random_seed (int): Random seed for reproducible train/val split. Defaults to 42.
        """
        self.data_root: str = data_root
        self.split: str = split
        
        # Auto-detect class folders if not provided
        if class_folders is None:
            class_folders = sorted([
                d for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d))
            ])
        
        # Store class mappings
        self.class_folders: List[str] = class_folders
        self.class_to_idx: Dict[str, int] = {
            cls: idx for idx, cls in enumerate(class_folders)
        }
        self.idx_to_class: Dict[int, str] = {
            idx: cls for cls, idx in self.class_to_idx.items()
        }
        self.num_classes: int = len(class_folders)
        
        # Load all tissues grouped by slide
        # Dictionary structure: slide_key (str) -> list of tissue sample dicts
        slides_dict: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Iterate through each class folder (category)
        for class_folder in class_folders:
            class_path = os.path.join(data_root, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            # Iterate through slide directories within each class
            for slide_dir in os.listdir(class_path):
                slide_path = os.path.join(class_path, slide_dir)
                if not os.path.isdir(slide_path):
                    continue
                
                # Create unique slide key for grouping (category/slide_dir)
                # This ensures slide-level stratification
                slide_key = f"{class_folder}/{slide_dir}"
                
                # Scan slide directory for feature files
                # Expected structure: slide_dir/tissue_name_features.pt
                for item in os.listdir(slide_path):
                    # Check if item is a feature file (ends with _features.pt)
                    if item.endswith('_features.pt'):
                        # Extract tissue name from filename
                        tissue_name = item.replace('_features.pt', '')
                        feature_path = os.path.join(slide_path, item)
                        tiles_path = os.path.join(slide_path, f"{tissue_name}_tiles.csv")
                        
                        # Verify both feature and tile files exist
                        if os.path.exists(feature_path) and os.path.exists(tiles_path):
                            slides_dict[slide_key].append({
                                'slide_name': slide_dir,
                                'tissue_name': tissue_name,
                                'feature_path': feature_path,
                                'tiles_path': tiles_path,
                                'class': class_folder
                            })
        
        # Convert grouped tissues to flat list while maintaining slide grouping info
        all_tissues: List[Dict[str, Any]] = []
        slide_labels: List[str] = []  # Track which slide each tissue belongs to
        
        # Flatten tissues while preserving slide membership
        for slide_key, tissues in slides_dict.items():
            if len(tissues) == 0:
                continue
            
            # All tissues from the same slide have the same class label
            class_folder = tissues[0]['class']
            slide_label = self.class_to_idx[class_folder]
            
            # Add all tissues from this slide to the list
            for tissue in tissues:
                all_tissues.append(tissue)
                slide_labels.append(slide_key)  # Track slide membership for stratification
        
        # Group tissues by slide for stratification
        # Create mapping: slide_key -> list of tissue indices in all_tissues
        slide_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, slide_key in enumerate(slide_labels):
            slide_to_indices[slide_key].append(idx)
        
        # Get unique slides and their class labels for stratification
        unique_slides: List[str] = list(slide_to_indices.keys())
        unique_slide_labels: List[int] = [
            self.class_to_idx[all_tissues[slide_to_indices[sk][0]]['class']]
            for sk in unique_slides
        ]
        
        # Perform slide-level train/val split (not tissue-level)
        # This ensures all tissues from the same slide are in the same split
        if len(unique_slides) > 0:
            train_slide_indices, val_slide_indices = train_test_split(
                range(len(unique_slides)),
                test_size=1 - train_ratio,
                random_state=random_seed,
                stratify=unique_slide_labels  # Stratify by class to maintain class balance
            )
            
            # Map slide indices back to tissue indices
            train_tissue_indices: List[int] = []
            val_tissue_indices: List[int] = []
            
            # Collect all tissue indices for training slides
            for slide_idx in train_slide_indices:
                slide_key = unique_slides[slide_idx]
                train_tissue_indices.extend(slide_to_indices[slide_key])
            
            # Collect all tissue indices for validation slides
            for slide_idx in val_slide_indices:
                slide_key = unique_slides[slide_idx]
                val_tissue_indices.extend(slide_to_indices[slide_key])
            
            # Assign indices based on requested split
            if split == 'train':
                self.indices: List[int] = train_tissue_indices
            elif split == 'val':
                self.indices: List[int] = val_tissue_indices
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")
        else:
            self.indices: List[int] = []
        
        # Store all tissue information
        self.tissues: List[Dict[str, Any]] = all_tissues
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of tissue samples in the current split.
        """
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single tissue sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'features' (torch.Tensor): Feature tensor of shape [n_tiles, feature_dim].
                  Typically feature_dim=1536 for H-Optimus features.
                - 'label' (int): Integer class label (0-indexed).
                - 'slide_name' (str): Name of the slide directory.
                - 'tissue_name' (str): Name of the tissue (extracted from filename).
        """
        # Map dataset index to actual tissue index
        actual_idx = self.indices[idx]
        tissue_info = self.tissues[actual_idx]
        
        # Load features from PyTorch .pt file
        features_tensor = torch.load(
            tissue_info['feature_path'],
            map_location='cpu',
            weights_only=False
        )
        
        # Ensure features are a float tensor
        if not isinstance(features_tensor, torch.Tensor):
            features_tensor = torch.FloatTensor(features_tensor)
        else:
            features_tensor = features_tensor.float()
        
        # Get integer class label
        label = self.class_to_idx[tissue_info['class']]
        
        return {
            'features': features_tensor,
            'label': label,
            'slide_name': tissue_info['slide_name'],
            'tissue_name': tissue_info['tissue_name']
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable-length sequences.
    
    Pads features to the same length within a batch to enable batch processing.
    Creates masks to indicate which tiles are valid (not padding).
    
    Args:
        batch (List[Dict[str, Any]]): List of samples from __getitem__. Each sample
            contains 'features', 'label', 'slide_name', and 'tissue_name'.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'features' (torch.Tensor): Padded feature tensor of shape
              [batch_size, max_tiles, feature_dim].
            - 'labels' (torch.LongTensor): Class labels tensor of shape [batch_size].
            - 'masks' (torch.BoolTensor): Boolean mask tensor of shape [batch_size, max_tiles].
              True indicates valid tiles, False indicates padding.
            - 'slide_names' (List[str]): List of slide names, one per sample.
            - 'tissue_names' (List[str]): List of tissue names, one per sample.
    """
    # Extract features, labels, and names from batch
    features_list = [item['features'] for item in batch]
    labels = torch.LongTensor([item['label'] for item in batch])
    slide_names = [item['slide_name'] for item in batch]
    tissue_names = [item['tissue_name'] for item in batch]
    
    # Find maximum number of tiles in this batch
    max_tiles = max(f.shape[0] for f in features_list)
    feature_dim = features_list[0].shape[1]
    
    # Create padded tensors and masks
    batch_size = len(batch)
    features_padded = torch.zeros(batch_size, max_tiles, feature_dim)
    masks = torch.zeros(batch_size, max_tiles, dtype=torch.bool)
    
    # Fill padded tensor and set mask for valid tiles
    for i, features in enumerate(features_list):
        n_tiles = features.shape[0]
        # Copy actual features to padded tensor
        features_padded[i, :n_tiles] = features
        # Mark valid tiles in mask
        masks[i, :n_tiles] = True
    
    return {
        'features': features_padded,
        'labels': labels,
        'masks': masks,
        'slide_names': slide_names,
        'tissue_names': tissue_names
    }
