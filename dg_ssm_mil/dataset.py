"""
Dataset utilities for DG-SSM-MIL tissue-level training.
"""
from collections import defaultdict
import hashlib
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class _LegacyDGSSMMILTissueDataset(Dataset):
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
        max_tiles_per_tissue: Optional[int] = None,
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
            max_tiles_per_tissue (Optional[int]): Maximum number of contiguous
                tiles to return per tissue. If None, all tiles are returned.

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
        self.max_tiles_per_tissue = max_tiles_per_tissue

        if split not in {"train", "val"}:
            raise ValueError("split must be either 'train' or 'val'.")
        if coordinate_mismatch not in {"error", "trim"}:
            raise ValueError("coordinate_mismatch must be either 'error' or 'trim'.")
        if not feature_file_suffix:
            raise ValueError("feature_file_suffix must be a non-empty string.")
        if max_tiles_per_tissue is not None and max_tiles_per_tissue <= 0:
            raise ValueError("max_tiles_per_tissue must be positive or None.")

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
        features, coords = _select_contiguous_region(
            features,
            coords,
            self.max_tiles_per_tissue,
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


class DGSSMMILTissueDataset(Dataset):
    """
    DG-SSM-MIL bag dataset with strict spatial alignment and slide-safe metadata.
    """

    def __init__(
        self,
        data_root: str,
        class_folders: Optional[List[str]] = None,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: Optional[float] = None,
        random_seed: int = 42,
        feature_file_suffix: str = "_features.pt",
        coordinate_columns: Tuple[str, str] = ("x", "y"),
        coordinate_mismatch: str = "error",
        sort_tiles_spatially: bool = False,
        normalize_coordinates: bool = False,
        max_tiles_per_tissue: Optional[int] = None,
        skip_tissues_above_tiles: Optional[int] = None,
        expected_feature_dim: Optional[int] = None,
        bag_level: str = "tissue",
        tile_sampling: str = "random",
        feature_normalization: str = "none",
    ) -> None:
        """
        Initialize a deterministic tissue or future-compatible slide bag dataset.

        Args:
            data_root (str): Root directory containing class and slide folders.
            class_folders (Optional[List[str]]): Stable ordered class names.
            split (str): Dataset split: `train`, `val`, or `test`.
            train_ratio (float): Fraction of slides assigned to training.
            val_ratio (float): Fraction of slides assigned to validation.
            test_ratio (Optional[float]): Fraction assigned to testing.
            random_seed (int): Split and sampling seed.
            feature_file_suffix (str): Selected feature tensor suffix.
            coordinate_columns (Tuple[str, str]): Coordinate column names. The
                shared artifact contract requires `x` and `y`.
            coordinate_mismatch (str): Must be `error`; silent trimming is rejected.
            sort_tiles_spatially (bool): Legacy option. Must remain false so path H
                preserves extraction order.
            normalize_coordinates (bool): Apply isotropic centering/scaling while
                preserving Euclidean neighbor topology.
            max_tiles_per_tissue (Optional[int]): Split-specific tile cap.
            skip_tissues_above_tiles (Optional[int]): Exclude source tissues with
                more coordinate rows than this threshold, or `None` to keep all.
            expected_feature_dim (Optional[int]): Required feature width.
            bag_level (str): `tissue` now or `slide` for future slide bags.
            tile_sampling (str): Deterministic capped-bag sampling strategy.
            feature_normalization (str): Optional feature normalization mode.

        Returns:
            None: Dataset discovery and split metadata are initialized.
        """
        if tuple(coordinate_columns) != ("x", "y"):
            raise ValueError("DG-SSM-MIL currently requires coordinate columns ['x', 'y'].")
        if coordinate_mismatch != "error":
            raise ValueError("coordinate_mismatch must be 'error' to preserve alignment.")
        if sort_tiles_spatially:
            raise ValueError(
                "sort_tiles_spatially must be false; the paper preserves path H order."
            )
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test.")
        if bag_level not in {"tissue", "slide"}:
            raise ValueError("bag_level must be tissue or slide.")
        if tile_sampling not in {"random", "uniform", "first"}:
            raise ValueError("tile_sampling must be random, uniform, or first.")
        if feature_normalization not in {"none", "l2", "layer_norm"}:
            raise ValueError(
                "feature_normalization must be none, l2, or layer_norm."
            )
        if skip_tissues_above_tiles is not None and skip_tissues_above_tiles <= 0:
            raise ValueError("skip_tissues_above_tiles must be positive or None.")
        resolved_test_ratio = (
            1.0 - train_ratio - val_ratio
            if test_ratio is None
            else float(test_ratio)
        )
        ratios = (float(train_ratio), float(val_ratio), resolved_test_ratio)
        if any(ratio < 0.0 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
            raise ValueError("train, validation, and test ratios must sum to 1.")

        self.data_root = data_root
        self.split = split
        self.random_seed = int(random_seed)
        self.feature_file_suffix = feature_file_suffix
        self.expected_feature_dim = expected_feature_dim
        self.bag_level = bag_level
        self.tile_sampling = tile_sampling
        self.feature_normalization = feature_normalization
        self.normalize_coordinates = normalize_coordinates
        self.epoch = 0

        if class_folders is None:
            class_folders = sorted(
                item
                for item in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, item))
            )
        self.class_folders = list(class_folders)
        self.class_to_idx = {
            class_name: class_idx
            for class_idx, class_name in enumerate(self.class_folders)
        }
        self.idx_to_class = {
            class_idx: class_name
            for class_name, class_idx in self.class_to_idx.items()
        }
        self.num_classes = len(self.class_folders)
        discovered_tissues = _discover_tissue_records(
            data_root, self.class_folders, feature_file_suffix
        )
        all_slide_bags = _group_tissues_by_slide(discovered_tissues)
        split_slide_keys = _split_slide_keys(
            all_slide_bags, ratios, self.random_seed
        )[split]
        applied_threshold = skip_tissues_above_tiles if split == "train" else None
        selected_tissues = [
            tissue
            for tissue in discovered_tissues
            if tissue["slide_key"] in split_slide_keys
        ]
        unselected_tissues = [
            tissue
            for tissue in discovered_tissues
            if tissue["slide_key"] not in split_slide_keys
        ]
        kept_selected, self.skipped_tissues = _filter_oversized_tissues(
            selected_tissues, applied_threshold
        )
        self.tissues = sorted(
            [*unselected_tissues, *kept_selected],
            key=lambda tissue: (
                tissue["class_name"],
                tissue["slide_name"],
                tissue["tissue_name"],
            ),
        )
        slide_bags = _group_tissues_by_slide(self.tissues)
        if bag_level == "tissue":
            self._bags = [
                {
                    "slide_name": tissue["slide_name"],
                    "slide_key": tissue["slide_key"],
                    "class_name": tissue["class_name"],
                    "tissues": [tissue],
                }
                for tissue in self.tissues
            ]
        else:
            self._bags = slide_bags
        self.indices = [
            bag_idx
            for bag_idx, bag in enumerate(self._bags)
            if bag["slide_key"] in split_slide_keys
        ]
        self.max_tiles_per_tissue = max_tiles_per_tissue
        self.skip_tissues_above_tiles = applied_threshold

    def __len__(self) -> int:
        """
        Return the number of bags in the selected split.

        Args:
            None: Length uses the precomputed split indices.

        Returns:
            int: Number of tissue or slide bags.
        """
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        """
        Select the deterministic sampling epoch.

        Args:
            epoch (int): Nonnegative training epoch.

        Returns:
            None: The epoch is stored for subsequent item loading.
        """
        if epoch < 0:
            raise ValueError("epoch must be nonnegative.")
        self.epoch = int(epoch)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load one aligned DG-SSM-MIL bag and expose compatibility metadata.

        Args:
            idx (int): Split-local bag index.

        Returns:
            Dict[str, Any]: Features, coordinates, tile/tissue indices, label,
            identifiers, and source paths.
        """
        bag = self._bags[self.indices[idx]]
        feature_parts = []
        coordinate_parts = []
        tile_index_parts = []
        tissue_index_parts = []
        feature_paths = []
        tiles_paths = []
        for tissue_idx, tissue in enumerate(bag["tissues"]):
            features = _load_feature_tensor(tissue["feature_path"])
            if (
                self.expected_feature_dim is not None
                and features.shape[1] != self.expected_feature_dim
            ):
                raise ValueError(
                    f"Expected feature dimension {self.expected_feature_dim} in "
                    f"{tissue['feature_path']}, got {features.shape[1]}."
                )
            coords = _load_coordinates(tissue["tiles_path"], ("x", "y"))
            if len(features) != len(coords):
                raise ValueError(
                    "Feature/coordinate row mismatch for "
                    f"'{tissue['feature_path']}': {len(features)} features "
                    f"versus {len(coords)} CSV rows."
                )
            feature_parts.append(features)
            coordinate_parts.append(coords)
            tile_index_parts.append(torch.arange(len(features), dtype=torch.long))
            tissue_index_parts.append(
                torch.full((len(features),), tissue_idx, dtype=torch.long)
            )
            feature_paths.append(tissue["feature_path"])
            tiles_paths.append(tissue["tiles_path"])

        features = torch.cat(feature_parts)
        coords = torch.cat(coordinate_parts)
        tile_indices = torch.cat(tile_index_parts)
        tissue_indices = torch.cat(tissue_index_parts)
        seed = _stable_sampling_seed(
            self.random_seed,
            bag["slide_key"],
            bag["tissues"][0]["tissue_name"] if self.bag_level == "tissue" else "",
            self.epoch if self.split == "train" else 0,
        )
        selected = _select_bag_indices(
            len(features), self.max_tiles_per_tissue, self.tile_sampling, seed
        )
        features = _normalize_bag_features(
            features[selected], self.feature_normalization
        )
        coords = coords[selected]
        tile_indices = tile_indices[selected]
        tissue_indices = tissue_indices[selected]
        if self.normalize_coordinates:
            coords = _normalize_coordinates_isotropically(coords)
        tissue_names = [tissue["tissue_name"] for tissue in bag["tissues"]]
        tissue_slices = _build_tissue_slices(tissue_indices, len(tissue_names))
        tissue_name = (
            tissue_names[0]
            if self.bag_level == "tissue"
            else f"slide_bag::{bag['slide_name']}"
        )
        provenance = {
            "bag_level": self.bag_level,
            "class_name": bag["class_name"],
            "slide_key": bag["slide_key"],
            "feature_paths": feature_paths,
            "tiles_paths": tiles_paths,
            "sampling_seed": seed,
            "epoch": self.epoch if self.split == "train" else 0,
        }
        return {
            "features": features,
            "coords": coords,
            "coordinates": coords,
            "label": self.class_to_idx[bag["class_name"]],
            "slide_name": bag["slide_name"],
            "tissue_name": tissue_name,
            "tissue_names": tissue_names,
            "tissue_slices": tissue_slices,
            "num_tissues": len(tissue_names),
            "tile_indices": tile_indices,
            "tissue_indices": tissue_indices,
            "feature_path": feature_paths[0] if len(feature_paths) == 1 else None,
            "tiles_path": tiles_paths[0] if len(tiles_paths) == 1 else None,
            "feature_paths": feature_paths,
            "tiles_paths": tiles_paths,
            "provenance": provenance,
        }


def _discover_tissue_records(
    data_root: str,
    class_folders: Sequence[str],
    feature_file_suffix: str,
) -> List[Dict[str, Any]]:
    """
    Discover aligned tissue feature and coordinate artifact paths.

    Args:
        data_root (str): Root directory containing class folders.
        class_folders (Sequence[str]): Ordered class folder names.
        feature_file_suffix (str): Selected feature filename suffix.

    Returns:
        List[Dict[str, Any]]: Stable tissue metadata records.
    """
    records = []
    for class_name in class_folders:
        class_path = os.path.join(data_root, class_name)
        if not os.path.isdir(class_path):
            continue
        for slide_name in sorted(os.listdir(class_path)):
            slide_path = os.path.join(class_path, slide_name)
            if not os.path.isdir(slide_path):
                continue
            for filename in sorted(os.listdir(slide_path)):
                if not filename.endswith(feature_file_suffix):
                    continue
                tissue_name = filename[: -len(feature_file_suffix)]
                tiles_path = os.path.join(slide_path, f"{tissue_name}_tiles.csv")
                if not os.path.isfile(tiles_path):
                    continue
                records.append(
                    {
                        "class_name": class_name,
                        "class": class_name,
                        "slide_name": slide_name,
                        "slide_key": f"{class_name}/{slide_name}",
                        "tissue_name": tissue_name,
                        "feature_path": os.path.join(slide_path, filename),
                        "tiles_path": tiles_path,
                    }
                )
    return records


def _filter_oversized_tissues(
    tissues: Sequence[Dict[str, Any]],
    max_tiles: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partition tissues by a configured coordinate-row threshold.

    Args:
        tissues (Sequence[Dict[str, Any]]): Discovered tissue records.
        max_tiles (Optional[int]): Maximum allowed tiles, or `None` to disable.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Kept and skipped
        tissue records; skipped records include their `num_tiles`.
    """
    if max_tiles is None:
        return list(tissues), []
    kept = []
    skipped = []
    for tissue in tissues:
        with open(tissue["tiles_path"], "r", encoding="utf-8") as tiles_file:
            num_tiles = max(sum(1 for _ in tiles_file) - 1, 0)
        if num_tiles > max_tiles:
            skipped.append({**tissue, "num_tiles": num_tiles})
        else:
            kept.append(tissue)
    return kept, skipped


def _group_tissues_by_slide(
    tissues: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Group tissue records into slide bags.

    Args:
        tissues (Sequence[Dict[str, Any]]): Discovered tissue records.

    Returns:
        List[Dict[str, Any]]: Stable slide-level bag records.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for tissue in tissues:
        grouped[tissue["slide_key"]].append(tissue)
    return [
        {
            "slide_name": grouped[key][0]["slide_name"],
            "slide_key": key,
            "class_name": grouped[key][0]["class_name"],
            "tissues": grouped[key],
        }
        for key in sorted(grouped)
    ]


def _split_slide_keys(
    slide_bags: Sequence[Dict[str, Any]],
    ratios: Tuple[float, float, float],
    random_seed: int,
) -> Dict[str, set[str]]:
    """
    Create deterministic class-stratified train, validation, and test slide keys.

    Args:
        slide_bags (Sequence[Dict[str, Any]]): One record per slide.
        ratios (Tuple[float, float, float]): Train, validation, and test ratios.
        random_seed (int): Seed controlling within-class shuffling.

    Returns:
        Dict[str, set[str]]: Disjoint slide-key sets for each split.
    """
    split_names = ("train", "val", "test")
    result = {split_name: set() for split_name in split_names}
    grouped: Dict[str, List[str]] = defaultdict(list)
    for bag in slide_bags:
        grouped[bag["class_name"]].append(bag["slide_key"])
    for class_idx, class_name in enumerate(sorted(grouped)):
        keys = sorted(grouped[class_name])
        generator = np.random.default_rng(random_seed + class_idx)
        generator.shuffle(keys)
        counts = _allocate_split_counts(len(keys), ratios)
        start = 0
        for split_name, count in zip(split_names, counts):
            result[split_name].update(keys[start : start + count])
            start += count
    return result


def _allocate_split_counts(
    total: int,
    ratios: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """
    Allocate an integer sample count with largest-remainder rounding.

    Args:
        total (int): Number of slides in one class.
        ratios (Tuple[float, float, float]): Requested split ratios.

    Returns:
        Tuple[int, int, int]: Train, validation, and test counts summing to total.
    """
    raw = np.asarray(ratios, dtype=np.float64) * total
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts))[: total - int(counts.sum())]:
        counts[index] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _stable_sampling_seed(*parts: Any) -> int:
    """
    Build a stable integer seed from bag identity and epoch values.

    Args:
        *parts (Any): Values contributing to the deterministic seed.

    Returns:
        int: Stable unsigned 32-bit seed.
    """
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _select_bag_indices(
    num_tiles: int,
    max_tiles: Optional[int],
    method: str,
    random_seed: int,
) -> torch.Tensor:
    """
    Select deterministic tile indices for a capped bag.

    Args:
        num_tiles (int): Number of available tiles.
        max_tiles (Optional[int]): Maximum retained tiles, or `None`.
        method (str): `random`, `uniform`, or `first`.
        random_seed (int): Seed used by random sampling.

    Returns:
        torch.Tensor: Sorted selected row indices.
    """
    if max_tiles is None or num_tiles <= max_tiles:
        return torch.arange(num_tiles, dtype=torch.long)
    if method == "first":
        return torch.arange(max_tiles, dtype=torch.long)
    if method == "uniform":
        return torch.linspace(0, num_tiles - 1, max_tiles).round().long().unique()
    generator = torch.Generator().manual_seed(random_seed)
    return torch.randperm(num_tiles, generator=generator)[:max_tiles].sort().values


def _normalize_bag_features(
    features: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """
    Apply optional per-tile feature normalization.

    Args:
        features (torch.Tensor): Feature matrix `[N, D]`.
        method (str): `none`, `l2`, or `layer_norm`.

    Returns:
        torch.Tensor: Normalized feature matrix.
    """
    if method == "l2":
        return torch.nn.functional.normalize(features, p=2, dim=-1)
    if method == "layer_norm":
        return torch.nn.functional.layer_norm(features, (features.shape[-1],))
    return features


def _build_tissue_slices(
    tissue_indices: torch.Tensor,
    num_tissues: int,
) -> List[Tuple[int, int]]:
    """
    Report selected-row ranges for tissues in a bag.

    Args:
        tissue_indices (torch.Tensor): Tissue membership for retained tiles.
        num_tissues (int): Number of source tissues.

    Returns:
        List[Tuple[int, int]]: Half-open selected-row bounds per tissue.
    """
    slices = []
    for tissue_idx in range(num_tissues):
        positions = torch.nonzero(
            tissue_indices == tissue_idx, as_tuple=False
        ).flatten()
        if positions.numel() == 0:
            slices.append((0, 0))
        else:
            slices.append((int(positions.min()), int(positions.max()) + 1))
    return slices


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


def _select_contiguous_region(
    features: torch.Tensor,
    coords: torch.Tensor,
    max_tiles: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select a spatially contiguous tile region around a random center tile.

    Args:
        features (torch.Tensor): Feature tensor of shape `[n_tiles, D]`.
        coords (torch.Tensor): Raw coordinate tensor of shape `[n_tiles, 2]`.
        max_tiles (Optional[int]): Maximum number of nearest tiles to keep. If
            None or greater than/equal to `n_tiles`, all tiles are returned.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cropped features and coordinates with
        at most `max_tiles` rows.
    """
    if max_tiles is None or features.shape[0] <= max_tiles:
        return features, coords

    center_idx = torch.randint(features.shape[0], size=(1,)).item()
    distances = torch.linalg.vector_norm(coords - coords[center_idx], dim=1)
    selected_indices = torch.topk(
        distances,
        k=max_tiles,
        largest=False,
    ).indices
    selected_indices, _ = torch.sort(selected_indices)
    return features[selected_indices], coords[selected_indices]


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


def _normalize_coordinates_isotropically(coords: torch.Tensor) -> torch.Tensor:
    """
    Center and isotropically scale coordinates without changing k-NN topology.

    Args:
        coords (torch.Tensor): Raw coordinates `[N, 2]`.

    Returns:
        torch.Tensor: Centered coordinates divided by one shared positive scale.
    """
    centered = coords.float() - coords.float().mean(dim=0, keepdim=True)
    scale = torch.linalg.vector_norm(centered, dim=1).std().clamp_min(1.0)
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
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    feature_dims = {int(item["features"].shape[1]) for item in batch}
    if len(feature_dims) != 1:
        raise ValueError("All bags in a batch must share one feature dimension.")
    batch_size = len(batch)
    max_tiles = max(int(item["features"].shape[0]) for item in batch)
    feature_dim = next(iter(feature_dims))
    features = torch.zeros(batch_size, max_tiles, feature_dim)
    coords = torch.full((batch_size, max_tiles, 2), float("nan"))
    masks = torch.zeros(batch_size, max_tiles, dtype=torch.bool)
    tile_indices = torch.full((batch_size, max_tiles), -1, dtype=torch.long)
    tissue_indices = torch.full((batch_size, max_tiles), -1, dtype=torch.long)
    for batch_idx, item in enumerate(batch):
        length = len(item["features"])
        features[batch_idx, :length] = item["features"]
        coords[batch_idx, :length] = item["coords"]
        masks[batch_idx, :length] = True
        tile_indices[batch_idx, :length] = item["tile_indices"]
        tissue_indices[batch_idx, :length] = item["tissue_indices"]
    return {
        "features": features,
        "coords": coords,
        "coordinates": coords,
        "labels": torch.as_tensor(
            [item["label"] for item in batch], dtype=torch.long
        ),
        "masks": masks,
        "tile_indices": tile_indices,
        "tissue_indices": tissue_indices,
        "slide_names": [item["slide_name"] for item in batch],
        "tissue_names": [item["tissue_name"] for item in batch],
        "bag_tissue_names": [item["tissue_names"] for item in batch],
        "tissue_slices": [item["tissue_slices"] for item in batch],
        "metadata": [
            {
                "slide_name": item["slide_name"],
                "tissue_name": item["tissue_name"],
                "tissue_names": item["tissue_names"],
                "tissue_slices": item["tissue_slices"],
                "num_tissues": item["num_tissues"],
            }
            for item in batch
        ],
        "provenance": [item["provenance"] for item in batch],
        "feature_paths": [item["feature_path"] for item in batch],
        "tiles_paths": [item["tiles_path"] for item in batch],
        "bag_feature_paths": [item["feature_paths"] for item in batch],
        "bag_tiles_paths": [item["tiles_paths"] for item in batch],
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
    for bag_idx in dataset.indices:
        class_counts[dataset._bags[bag_idx]["class_name"]] += 1
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
        class_weight_map[dataset._bags[bag_idx]["class_name"]]
        for bag_idx in dataset.indices
    ]
