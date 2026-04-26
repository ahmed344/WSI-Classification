"""
Dataset utilities for DG-SSM-MIL tissue-level training.
"""
from collections import defaultdict
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class DGSSMMILTissueDataset(Dataset):
    """
    Tissue-level MIL dataset that loads patch features and tile coordinates.
    """

    def __init__(
        self,
        data_root: str,
        class_folders: Optional[List[str]] = None,
        split: str = "train",
        train_ratio: float = 0.8,
        random_seed: int = 42,
        feature_file_suffix: str = "_features.pt",
        coordinate_columns: Tuple[str, str] = ("x", "y"),
        coordinate_mismatch: str = "trim",
        sort_tiles_spatially: bool = True,
        normalize_coordinates: bool = True,
    ) -> None:
        """
        Initialize the tissue-level DG-SSM-MIL dataset.

        Args:
            data_root (str): Root directory containing class folders.
            class_folders (Optional[List[str]]): Class folder names. If None,
                directories under `data_root` are auto-detected.
            split (str): Dataset split, either `train` or `val`.
            train_ratio (float): Fraction of slides assigned to the training split.
            random_seed (int): Random seed for reproducible slide-level splitting.
            feature_file_suffix (str): Suffix used to find feature tensors.
            coordinate_columns (Tuple[str, str]): CSV columns containing x and y.
            coordinate_mismatch (str): Handling when feature and coordinate counts
                differ; either `error` or `trim`.
            sort_tiles_spatially (bool): Whether to sort tiles by y then x.
            normalize_coordinates (bool): Whether to center and scale coordinates
                per tissue before returning them.

        Returns:
            None: This constructor initializes dataset metadata in-place.
        """
        self.data_root = data_root
        self.split = split
        self.feature_file_suffix = feature_file_suffix
        self.coordinate_columns = coordinate_columns
        self.coordinate_mismatch = coordinate_mismatch
        self.sort_tiles_spatially = sort_tiles_spatially
        self.normalize_coordinates = normalize_coordinates

        if split not in {"train", "val"}:
            raise ValueError("split must be either 'train' or 'val'.")
        if coordinate_mismatch not in {"error", "trim"}:
            raise ValueError("coordinate_mismatch must be either 'error' or 'trim'.")
        if not feature_file_suffix:
            raise ValueError("feature_file_suffix must be a non-empty string.")

        if class_folders is None:
            class_folders = sorted(
                item
                for item in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, item))
            )

        self.class_folders = class_folders
        self.class_to_idx = {
            class_name: class_idx for class_idx, class_name in enumerate(class_folders)
        }
        self.idx_to_class = {
            class_idx: class_name for class_name, class_idx in self.class_to_idx.items()
        }
        self.num_classes = len(class_folders)

        self.tissues = self._discover_tissues()
        self.indices = self._build_split_indices(train_ratio, random_seed)

    def _discover_tissues(self) -> List[Dict[str, Any]]:
        """
        Discover feature tensors with matching coordinate CSV files.

        Args:
            None: Discovery uses paths and class folders stored on the dataset.

        Returns:
            List[Dict[str, Any]]: Tissue metadata dictionaries for all discovered
            samples before train/validation filtering.
        """
        tissues: List[Dict[str, Any]] = []
        for class_folder in self.class_folders:
            class_path = os.path.join(self.data_root, class_folder)
            if not os.path.isdir(class_path):
                continue
            for slide_dir in sorted(os.listdir(class_path)):
                slide_path = os.path.join(class_path, slide_dir)
                if not os.path.isdir(slide_path):
                    continue
                for item in sorted(os.listdir(slide_path)):
                    if not item.endswith(self.feature_file_suffix):
                        continue
                    tissue_name = item[: -len(self.feature_file_suffix)]
                    feature_path = os.path.join(slide_path, item)
                    tiles_path = os.path.join(slide_path, f"{tissue_name}_tiles.csv")
                    if not os.path.exists(tiles_path):
                        continue
                    tissues.append(
                        {
                            "class": class_folder,
                            "slide_name": slide_dir,
                            "slide_key": f"{class_folder}/{slide_dir}",
                            "tissue_name": tissue_name,
                            "feature_path": feature_path,
                            "tiles_path": tiles_path,
                        }
                    )
        return tissues

    def _build_split_indices(
        self,
        train_ratio: float,
        random_seed: int,
    ) -> List[int]:
        """
        Build split-local tissue indices with slide-level grouping.

        Args:
            train_ratio (float): Fraction of slides assigned to training.
            random_seed (int): Random seed for deterministic splitting.

        Returns:
            List[int]: Indices into `self.tissues` for the requested split.
        """
        slide_to_indices: Dict[str, List[int]] = defaultdict(list)
        for tissue_idx, tissue in enumerate(self.tissues):
            slide_to_indices[tissue["slide_key"]].append(tissue_idx)

        unique_slides = list(slide_to_indices.keys())
        if not unique_slides:
            return []

        slide_labels = [
            self.class_to_idx[self.tissues[slide_to_indices[slide_key][0]]["class"]]
            for slide_key in unique_slides
        ]
        stratify_labels = slide_labels if _can_stratify(slide_labels, train_ratio) else None
        train_slide_indices, val_slide_indices = train_test_split(
            range(len(unique_slides)),
            test_size=1.0 - train_ratio,
            random_state=random_seed,
            stratify=stratify_labels,
        )

        selected_slide_indices = (
            train_slide_indices if self.split == "train" else val_slide_indices
        )
        selected_tissue_indices: List[int] = []
        for slide_idx in selected_slide_indices:
            selected_tissue_indices.extend(slide_to_indices[unique_slides[slide_idx]])
        return selected_tissue_indices

    def __len__(self) -> int:
        """
        Return the number of tissues in the selected split.

        Args:
            None: Length is computed from split indices.

        Returns:
            int: Number of tissue samples.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load one tissue sample with features, coordinates, and label.

        Args:
            idx (int): Split-local sample index.

        Returns:
            Dict[str, Any]: Sample containing feature tensor `[n_tiles, D]`,
            coordinate tensor `[n_tiles, 2]`, integer label, and identifiers.
        """
        tissue = self.tissues[self.indices[idx]]
        features = _load_feature_tensor(tissue["feature_path"])
        coords = _load_coordinates(tissue["tiles_path"], self.coordinate_columns)
        features, coords = _align_feature_and_coordinate_lengths(
            features,
            coords,
            self.coordinate_mismatch,
            tissue["feature_path"],
            tissue["tiles_path"],
        )
        if self.sort_tiles_spatially:
            features, coords = _sort_by_spatial_position(features, coords)
        if self.normalize_coordinates:
            coords = _normalize_coordinates(coords)

        return {
            "features": features,
            "coords": coords,
            "label": self.class_to_idx[tissue["class"]],
            "slide_name": tissue["slide_name"],
            "tissue_name": tissue["tissue_name"],
            "feature_path": tissue["feature_path"],
            "tiles_path": tissue["tiles_path"],
        }


def _can_stratify(labels: Sequence[int], train_ratio: float) -> bool:
    """
    Determine whether stratified splitting is feasible.

    Args:
        labels (Sequence[int]): Slide-level class labels.
        train_ratio (float): Fraction of slides assigned to training.

    Returns:
        bool: True when every class can appear in both train and validation.
    """
    if len(labels) < 2:
        return False
    unique, counts = np.unique(np.asarray(labels), return_counts=True)
    if len(unique) < 2 or np.min(counts) < 2:
        return False
    val_count = int(np.ceil((1.0 - train_ratio) * len(labels)))
    train_count = len(labels) - val_count
    return val_count >= len(unique) and train_count >= len(unique)


def _load_feature_tensor(feature_path: str) -> torch.Tensor:
    """
    Load a feature tensor from disk.

    Args:
        feature_path (str): Path to a `.pt` feature tensor file.

    Returns:
        torch.Tensor: Float tensor of shape `[n_tiles, feature_dim]`.
    """
    tensor = torch.load(feature_path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor = tensor.float()
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D feature tensor in {feature_path}, got {tensor.shape}.")
    return tensor


def _load_coordinates(
    tiles_path: str,
    coordinate_columns: Tuple[str, str],
) -> torch.Tensor:
    """
    Load tile coordinates from a CSV file.

    Args:
        tiles_path (str): Path to the tile coordinate CSV file.
        coordinate_columns (Tuple[str, str]): Names of the x and y columns.

    Returns:
        torch.Tensor: Float tensor of shape `[n_tiles, 2]`.
    """
    tiles_df = pd.read_csv(tiles_path)
    missing_columns = [
        column for column in coordinate_columns if column not in tiles_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing coordinate columns {missing_columns} in {tiles_path}."
        )
    coords = tiles_df.loc[:, list(coordinate_columns)].to_numpy(dtype=np.float32)
    return torch.from_numpy(coords)


def _align_feature_and_coordinate_lengths(
    features: torch.Tensor,
    coords: torch.Tensor,
    coordinate_mismatch: str,
    feature_path: str,
    tiles_path: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align features and coordinates when their tile counts differ.

    Args:
        features (torch.Tensor): Feature tensor of shape `[n_features, D]`.
        coords (torch.Tensor): Coordinate tensor of shape `[n_coords, 2]`.
        coordinate_mismatch (str): Handling mode, either `error` or `trim`.
        feature_path (str): Source feature path for error messages.
        tiles_path (str): Source coordinate path for error messages.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Length-aligned features and coordinates.
    """
    if features.shape[0] == coords.shape[0]:
        return features, coords
    if coordinate_mismatch == "error":
        raise ValueError(
            "Feature/coordinate count mismatch: "
            f"{features.shape[0]} features in {feature_path}, "
            f"{coords.shape[0]} coordinates in {tiles_path}."
        )
    n_tiles = min(features.shape[0], coords.shape[0])
    if n_tiles <= 0:
        raise ValueError(f"No overlapping tiles between {feature_path} and {tiles_path}.")
    return features[:n_tiles], coords[:n_tiles]


def _sort_by_spatial_position(
    features: torch.Tensor,
    coords: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sort tiles in row-major spatial order.

    Args:
        features (torch.Tensor): Feature tensor of shape `[n_tiles, D]`.
        coords (torch.Tensor): Coordinate tensor of shape `[n_tiles, 2]`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Features and coordinates sorted by y then x.
    """
    order = np.lexsort((coords[:, 0].numpy(), coords[:, 1].numpy()))
    order_tensor = torch.as_tensor(order, dtype=torch.long)
    return features[order_tensor], coords[order_tensor]


def _normalize_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """
    Center and scale coordinates per tissue for graph construction.

    Args:
        coords (torch.Tensor): Raw coordinate tensor of shape `[n_tiles, 2]`.

    Returns:
        torch.Tensor: Normalized coordinate tensor of shape `[n_tiles, 2]`.
    """
    coords = coords.float()
    centered = coords - coords.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, keepdim=True).clamp_min(1.0)
    return centered / scale


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pad variable-length tissue bags into a batch.

    Args:
        batch (List[Dict[str, Any]]): Samples returned by
            `DGSSMMILTissueDataset.__getitem__`.

    Returns:
        Dict[str, Any]: Batch with padded `features`, padded `coords`, `labels`,
        boolean `masks`, and sample identifiers.
    """
    features_list = [item["features"] for item in batch]
    coords_list = [item["coords"] for item in batch]
    labels = torch.as_tensor([item["label"] for item in batch], dtype=torch.long)
    max_tiles = max(features.shape[0] for features in features_list)
    feature_dim = features_list[0].shape[1]
    batch_size = len(batch)

    features_padded = torch.zeros(batch_size, max_tiles, feature_dim, dtype=torch.float32)
    coords_padded = torch.zeros(batch_size, max_tiles, 2, dtype=torch.float32)
    masks = torch.zeros(batch_size, max_tiles, dtype=torch.bool)

    for sample_idx, (features, coords) in enumerate(zip(features_list, coords_list)):
        n_tiles = features.shape[0]
        features_padded[sample_idx, :n_tiles] = features
        coords_padded[sample_idx, :n_tiles] = coords
        masks[sample_idx, :n_tiles] = True

    return {
        "features": features_padded,
        "coords": coords_padded,
        "labels": labels,
        "masks": masks,
        "slide_names": [item["slide_name"] for item in batch],
        "tissue_names": [item["tissue_name"] for item in batch],
        "feature_paths": [item["feature_path"] for item in batch],
        "tiles_paths": [item["tiles_path"] for item in batch],
    }


def get_class_sample_counts(dataset: DGSSMMILTissueDataset) -> Dict[str, int]:
    """
    Count split samples per class.

    Args:
        dataset (DGSSMMILTissueDataset): Dataset with split indices and tissue metadata.

    Returns:
        Dict[str, int]: Mapping from class folder name to sample count.
    """
    class_counts = {class_name: 0 for class_name in dataset.class_folders}
    for tissue_idx in dataset.indices:
        class_counts[dataset.tissues[tissue_idx]["class"]] += 1
    return class_counts


def compute_class_weights(dataset: DGSSMMILTissueDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for a dataset split.

    Args:
        dataset (DGSSMMILTissueDataset): Dataset used to count class frequencies.

    Returns:
        torch.Tensor: Float tensor of shape `[num_classes]`.
    """
    class_counts = get_class_sample_counts(dataset)
    total_samples = sum(class_counts.values())
    weights: List[float] = []
    for class_name in dataset.class_folders:
        count = class_counts[class_name]
        weight = total_samples / (len(dataset.class_folders) * count) if count > 0 else 0.0
        weights.append(weight)
    return torch.as_tensor(weights, dtype=torch.float32)


def compute_sample_weights(
    dataset: DGSSMMILTissueDataset,
    class_weights: torch.Tensor,
) -> List[float]:
    """
    Compute per-sample weights for weighted random sampling.

    Args:
        dataset (DGSSMMILTissueDataset): Dataset with split indices.
        class_weights (torch.Tensor): Class weights aligned to dataset class order.

    Returns:
        List[float]: Per-sample weights aligned with `dataset.indices`.
    """
    class_weight_map = {
        class_name: float(class_weights[class_idx].item())
        for class_idx, class_name in enumerate(dataset.class_folders)
    }
    return [
        class_weight_map[dataset.tissues[tissue_idx]["class"]]
        for tissue_idx in dataset.indices
    ]
