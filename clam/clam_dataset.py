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
        random_seed: int = 42,
        feature_file_suffix: str = '_features.pt',
        max_tiles_per_tissue: Optional[int] = None
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
            feature_file_suffix (str): Feature filename suffix to match when scanning
                slide directories (e.g., '_features.pt' or '_features_hoptimus.pt').
                Defaults to '_features.pt'.
            max_tiles_per_tissue (Optional[int]): Maximum number of tile features to
                return per tissue. If None, all tiles are returned. Defaults to None.
        """
        self.data_root: str = data_root
        self.split: str = split
        self.feature_file_suffix: str = feature_file_suffix
        if len(self.feature_file_suffix) == 0:
            raise ValueError("feature_file_suffix must be a non-empty string.")
        if max_tiles_per_tissue is not None and max_tiles_per_tissue <= 0:
            raise ValueError("max_tiles_per_tissue must be positive or None.")
        self.max_tiles_per_tissue: Optional[int] = max_tiles_per_tissue
        
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
                # Expected structure: slide_dir/tissue_name<feature_file_suffix>
                for item in os.listdir(slide_path):
                    # Check if item is a feature file for selected extractor
                    if item.endswith(self.feature_file_suffix):
                        # Extract tissue name from filename
                        tissue_name = item[:-len(self.feature_file_suffix)]
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
        features_tensor = subsample_features(
            features=features_tensor,
            max_tiles=self.max_tiles_per_tissue
        )
        
        # Get integer class label
        label = self.class_to_idx[tissue_info['class']]
        
        return {
            'features': features_tensor,
            'label': label,
            'slide_name': tissue_info['slide_name'],
            'tissue_name': tissue_info['tissue_name']
        }


def subsample_features(
    features: torch.Tensor,
    max_tiles: Optional[int]
) -> torch.Tensor:
    """
    Randomly subsample tile features when a bag exceeds a configured limit.

    Args:
        features (torch.Tensor): Feature tensor with shape `[n_tiles, feature_dim]`.
        max_tiles (Optional[int]): Maximum number of tiles to keep. If None or
            greater than/equal to `n_tiles`, the original feature tensor is returned.

    Returns:
        torch.Tensor: Feature tensor with at most `max_tiles` rows, preserving
            original row order for the randomly selected tiles.
    """
    if max_tiles is None or features.shape[0] <= max_tiles:
        return features

    selected_indices = torch.randperm(features.shape[0])[:max_tiles]
    selected_indices, _ = torch.sort(selected_indices)
    return features[selected_indices]


class WSISlideBagDataset(Dataset):
    """
    Dataset that builds one bag per slide by concatenating all tissue features.

    Each sample corresponds to one slide directory. All tissue feature files that
    match the selected feature suffix are loaded and concatenated along the tile
    dimension to build a single slide-level bag.
    """

    def __init__(
        self,
        data_root: str,
        class_folders: Optional[List[str]] = None,
        split: str = 'train',
        train_ratio: float = 0.9,
        random_seed: int = 42,
        feature_file_suffix: str = '_features.pt'
    ) -> None:
        """
        Initialize the slide-level bag dataset.

        Args:
            data_root (str): Root directory containing class folders.
            class_folders (Optional[List[str]]): Class folder names. If None,
                class folders are auto-detected from `data_root`.
            split (str): Dataset split name. Must be `'train'` or `'val'`.
            train_ratio (float): Ratio of slides assigned to train split.
            random_seed (int): Random seed used in reproducible split.
            feature_file_suffix (str): Suffix used to select feature files.

        Returns:
            None: This constructor initializes dataset metadata and split indices.
        """
        self.data_root: str = data_root
        self.split: str = split
        self.feature_file_suffix: str = feature_file_suffix
        if len(self.feature_file_suffix) == 0:
            raise ValueError("feature_file_suffix must be a non-empty string.")

        if class_folders is None:
            class_folders = sorted([
                d for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d))
            ])

        self.class_folders: List[str] = class_folders
        self.class_to_idx: Dict[str, int] = {
            cls: idx for idx, cls in enumerate(class_folders)
        }
        self.idx_to_class: Dict[int, str] = {
            idx: cls for cls, idx in self.class_to_idx.items()
        }
        self.num_classes: int = len(class_folders)

        # slide_key -> slide sample metadata
        slide_samples: Dict[str, Dict[str, Any]] = {}
        unique_slides: List[str] = []
        unique_slide_labels: List[int] = []

        for class_folder in class_folders:
            class_path = os.path.join(data_root, class_folder)
            if not os.path.isdir(class_path):
                continue

            for slide_dir in os.listdir(class_path):
                slide_path = os.path.join(class_path, slide_dir)
                if not os.path.isdir(slide_path):
                    continue

                feature_paths: List[str] = []
                tissue_names: List[str] = []
                for item in os.listdir(slide_path):
                    if not item.endswith(self.feature_file_suffix):
                        continue

                    tissue_name = item[:-len(self.feature_file_suffix)]
                    feature_path = os.path.join(slide_path, item)
                    tiles_path = os.path.join(slide_path, f"{tissue_name}_tiles.csv")
                    if os.path.exists(feature_path) and os.path.exists(tiles_path):
                        feature_paths.append(feature_path)
                        tissue_names.append(tissue_name)

                if len(feature_paths) == 0:
                    continue

                slide_key = f"{class_folder}/{slide_dir}"
                sorted_pairs = sorted(
                    zip(tissue_names, feature_paths),
                    key=lambda pair: pair[0]
                )
                sorted_tissue_names = [pair[0] for pair in sorted_pairs]
                sorted_feature_paths = [pair[1] for pair in sorted_pairs]

                slide_samples[slide_key] = {
                    'slide_name': slide_dir,
                    'class': class_folder,
                    'feature_paths': sorted_feature_paths,
                    'tissue_names': sorted_tissue_names
                }
                unique_slides.append(slide_key)
                unique_slide_labels.append(self.class_to_idx[class_folder])

        self.slides: List[Dict[str, Any]] = [
            slide_samples[slide_key] for slide_key in unique_slides
        ]

        if len(unique_slides) > 0:
            train_slide_indices, val_slide_indices = train_test_split(
                range(len(unique_slides)),
                test_size=1 - train_ratio,
                random_state=random_seed,
                stratify=unique_slide_labels
            )
            if split == 'train':
                self.indices: List[int] = list(train_slide_indices)
            elif split == 'val':
                self.indices = list(val_slide_indices)
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")
        else:
            self.indices = []

    def __len__(self) -> int:
        """
        Get the number of slide bags in the selected split.

        Args:
            None

        Returns:
            int: Number of slide-level samples in current split.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get one slide-level bag made from all tissue features in the slide.

        Args:
            idx (int): Index of the slide sample in split-local indexing.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - `features` (torch.Tensor): Slide bag feature tensor with shape
                  `[total_tiles_in_slide, feature_dim]`.
                - `label` (int): Integer class label for the slide.
                - `slide_name` (str): Name of the slide directory.
                - `tissue_name` (str): Synthetic identifier for compatibility
                  with collate logic.
                - `num_tissues` (int): Number of tissues concatenated for this slide.
        """
        actual_idx = self.indices[idx]
        slide_info = self.slides[actual_idx]

        features_list: List[torch.Tensor] = []
        for feature_path in slide_info['feature_paths']:
            features_tensor = torch.load(
                feature_path,
                map_location='cpu',
                weights_only=False
            )
            if not isinstance(features_tensor, torch.Tensor):
                features_tensor = torch.FloatTensor(features_tensor)
            else:
                features_tensor = features_tensor.float()
            features_list.append(features_tensor)

        concatenated_features = torch.cat(features_list, dim=0)
        label = self.class_to_idx[slide_info['class']]
        tissue_names = slide_info['tissue_names']

        return {
            'features': concatenated_features,
            'label': label,
            'slide_name': slide_info['slide_name'],
            'tissue_name': f"slide_bag::{slide_info['slide_name']}",
            'num_tissues': len(tissue_names)
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
