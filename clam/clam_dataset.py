import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class WSIFeatureDataset(Dataset):
    """
    Dataset for loading WSI features from pickle files.
    Each sample is a slide (bag) containing multiple tile features.
    """
    
    def __init__(self, data_root, class_folders=None, split='train', train_ratio=0.9, random_seed=42):
        """
        Args:
            data_root: Root directory containing class folders
            class_folders: List of class folder names. If None, auto-detect from data_root
            split: 'train' or 'val'
            train_ratio: Ratio of training data (default 0.9)
            random_seed: Random seed for train/val split
        """
        self.data_root = data_root
        self.split = split
        
        # Class mapping
        if class_folders is None:
            # Auto-detect class folders
            class_folders = sorted([d for d in os.listdir(data_root) 
                                   if os.path.isdir(os.path.join(data_root, d))])
        
        self.class_folders = class_folders
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_folders)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(class_folders)
        
        # Load all slides with their labels
        self.slides = []
        self.labels = []
        
        for class_folder in class_folders:
            class_path = os.path.join(data_root, class_folder)
            if not os.path.isdir(class_path):
                continue
                
            # Find all feature files
            feature_files = [f for f in os.listdir(class_path) if f.endswith('_features.pkl')]
            
            for feature_file in feature_files:
                slide_name = feature_file.replace('_features.pkl', '')
                feature_path = os.path.join(class_path, feature_file)
                
                # Verify file exists and is readable
                if os.path.exists(feature_path):
                    self.slides.append({
                        'slide_name': slide_name,
                        'feature_path': feature_path,
                        'class': class_folder
                    })
                    self.labels.append(self.class_to_idx[class_folder])
        
        # Split into train/val
        if len(self.slides) > 0:
            train_indices, val_indices = train_test_split(
                range(len(self.slides)),
                test_size=1 - train_ratio,
                random_state=random_seed,
                stratify=self.labels
            )
            
            if split == 'train':
                self.indices = train_indices
            elif split == 'val':
                self.indices = val_indices
        else:
            self.indices = []
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            features: torch.Tensor of shape [n_tiles, 1536]
            label: int class label
            slide_name: str slide identifier
        """
        actual_idx = self.indices[idx]
        slide_info = self.slides[actual_idx]
        
        # Load features
        features_df = pd.read_pickle(slide_info['feature_path'])
        
        # Convert to numpy array (handle both DataFrame and array formats)
        if isinstance(features_df, pd.DataFrame):
            features = features_df.values
        else:
            features = np.array(features_df)
        
        # Convert to torch tensor
        features_tensor = torch.FloatTensor(features)
        
        # Get label
        label = self.labels[actual_idx]
        
        return {
            'features': features_tensor,
            'label': label,
            'slide_name': slide_info['slide_name']
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads features to the same length within a batch.
    
    Returns:
        features: torch.Tensor [batch_size, max_tiles, feature_dim]
        labels: torch.LongTensor [batch_size]
        masks: torch.BoolTensor [batch_size, max_tiles] - True for valid tiles
        slide_names: list of slide names
    """
    features_list = [item['features'] for item in batch]
    labels = torch.LongTensor([item['label'] for item in batch])
    slide_names = [item['slide_name'] for item in batch]
    
    # Get max number of tiles in batch
    max_tiles = max(f.shape[0] for f in features_list)
    feature_dim = features_list[0].shape[1]
    
    # Pad features and create masks
    batch_size = len(batch)
    features_padded = torch.zeros(batch_size, max_tiles, feature_dim)
    masks = torch.zeros(batch_size, max_tiles, dtype=torch.bool)
    
    for i, features in enumerate(features_list):
        n_tiles = features.shape[0]
        features_padded[i, :n_tiles] = features
        masks[i, :n_tiles] = True
    
    return {
        'features': features_padded,
        'labels': labels,
        'masks': masks,
        'slide_names': slide_names
    }

