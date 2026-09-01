"""Independent tissue- and slide-level bag datasets for logistic regression."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


SPLITS = ("train", "val", "test")
BAG_LEVELS = ("tissue", "slide")
NORMALIZATIONS = ("none", "l2", "layer_norm")
SAMPLING_METHODS = ("random", "uniform", "first")


class TissueRecord(TypedDict):
    """Describe one aligned tissue feature/coordinate source.

    Args:
        slide_name (str): Source slide directory name.
        tissue_name (str): Tissue basename shared by feature and CSV files.
        feature_path (str): Feature tensor path.
        tiles_path (str): Coordinate CSV path.
        class_name (str): Class directory name.
        slide_key (str): Class-qualified slide identifier.

    Returns:
        TissueRecord: Typed tissue metadata mapping.
    """

    slide_name: str
    tissue_name: str
    feature_path: str
    tiles_path: str
    class_name: str
    slide_key: str


class BagRecord(TypedDict):
    """Describe one tissue- or slide-level bag.

    Args:
        slide_name (str): Source slide directory name.
        class_name (str): Class directory name.
        slide_key (str): Class-qualified slide identifier.
        tissues (List[TissueRecord]): Ordered tissue records in the bag.

    Returns:
        BagRecord: Typed bag metadata mapping.
    """

    slide_name: str
    class_name: str
    slide_key: str
    tissues: List[TissueRecord]


class WSIBagDataset(Dataset):
    """Load deterministic raw-feature bags without depending on MIL packages.

    Args:
        data_root (str): Root containing ``class/slide`` directories.
        class_folders (Optional[Sequence[str]]): Ordered classes, or ``None`` to
            discover sorted immediate directories.
        split (str): Requested split: ``train``, ``val``, or ``test``.
        train_ratio (float): Fraction of slides assigned to training.
        val_ratio (float): Fraction of slides assigned to validation.
        test_ratio (Optional[float]): Test fraction, or ``None`` for remainder.
        random_seed (int): Base split and sampling seed.
        feature_file_suffix (str): Feature filename suffix ending in ``.pt``.
        expected_feature_dim (Optional[int]): Required raw feature width.
        bag_level (str): ``tissue`` or ``slide``.
        max_tiles_per_bag (Optional[int]): Maximum retained tiles.
        tile_sampling (str): ``random``, ``uniform``, or ``first``.
        feature_normalization (str): ``none``, ``l2``, or ``layer_norm``.

    Returns:
        WSIBagDataset: Initialized split-local bag dataset.
    """

    def __init__(
        self,
        data_root: str,
        class_folders: Optional[Sequence[str]] = None,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: Optional[float] = None,
        random_seed: int = 42,
        feature_file_suffix: str = "_features.pt",
        expected_feature_dim: Optional[int] = None,
        bag_level: str = "tissue",
        max_tiles_per_bag: Optional[int] = None,
        tile_sampling: str = "random",
        feature_normalization: str = "none",
    ) -> None:
        """Initialize discovery and the CLAM-compatible grouped split.

        Args:
            data_root (str): Root containing ``class/slide`` directories.
            class_folders (Optional[Sequence[str]]): Ordered class names.
            split (str): Requested split name.
            train_ratio (float): Training slide fraction.
            val_ratio (float): Validation slide fraction.
            test_ratio (Optional[float]): Test fraction or remainder.
            random_seed (int): Deterministic base seed.
            feature_file_suffix (str): Feature tensor suffix.
            expected_feature_dim (Optional[int]): Required tensor width.
            bag_level (str): Tissue or slide bag granularity.
            max_tiles_per_bag (Optional[int]): Optional tile cap.
            tile_sampling (str): Capped-bag sampling method.
            feature_normalization (str): Per-tile normalization method.

        Returns:
            None: Dataset state is initialized in place.
        """
        self.data_root = str(Path(data_root).expanduser().resolve())
        self.split = split
        self.random_seed = int(random_seed)
        self.feature_file_suffix = feature_file_suffix
        self.expected_feature_dim = expected_feature_dim
        self.bag_level = bag_level
        self.max_tiles_per_bag = max_tiles_per_bag
        self.tile_sampling = tile_sampling
        self.feature_normalization = feature_normalization
        self.epoch = 0
        root = Path(self.data_root)
        if not root.is_dir():
            raise ValueError(f"data_root is not a directory: {root}")
        if split not in SPLITS:
            raise ValueError(f"Invalid split '{split}'. Expected one of {SPLITS}.")
        if bag_level not in BAG_LEVELS:
            raise ValueError(f"Invalid bag_level '{bag_level}'.")
        if tile_sampling not in SAMPLING_METHODS:
            raise ValueError(f"Invalid tile_sampling '{tile_sampling}'.")
        if feature_normalization not in NORMALIZATIONS:
            raise ValueError(f"Invalid feature_normalization '{feature_normalization}'.")
        if not feature_file_suffix or not feature_file_suffix.endswith(".pt"):
            raise ValueError("feature_file_suffix must be a nonempty '.pt' suffix.")
        if expected_feature_dim is not None and expected_feature_dim <= 0:
            raise ValueError("expected_feature_dim must be positive or None.")
        if max_tiles_per_bag is not None and max_tiles_per_bag <= 0:
            raise ValueError("max_tiles_per_bag must be positive or None.")
        resolved_test = (
            1.0 - train_ratio - val_ratio if test_ratio is None else float(test_ratio)
        )
        ratios = (float(train_ratio), float(val_ratio), resolved_test)
        _validate_ratios(ratios)
        self.split_ratios = dict(zip(SPLITS, ratios))
        class_names = (
            sorted(path.name for path in root.iterdir() if path.is_dir())
            if class_folders is None
            else list(class_folders)
        )
        if not class_names or len(set(class_names)) != len(class_names):
            raise ValueError("class_folders must contain unique class names.")
        self.class_folders = class_names
        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(class_names)
        }
        self.idx_to_class = {
            index: class_name for class_name, index in self.class_to_idx.items()
        }
        self.num_classes = len(class_names)
        discovered = _discover_tissues(root, class_names, feature_file_suffix)
        slide_records = _group_by_slide(discovered)
        labels = [self.class_to_idx[item["class_name"]] for item in slide_records]
        split_indices = _split_slide_indices(labels, ratios, self.random_seed)
        selected_keys = {
            slide_records[index]["slide_key"] for index in split_indices[split]
        }
        self.tissues: List[Dict[str, Any]] = [
            {**tissue, "class": tissue["class_name"]} for tissue in discovered
        ]
        self.slides: List[Dict[str, Any]] = [
            {
                **slide,
                "class": slide["class_name"],
                "feature_paths": [item["feature_path"] for item in slide["tissues"]],
                "tissue_names": [item["tissue_name"] for item in slide["tissues"]],
            }
            for slide in slide_records
        ]
        if bag_level == "tissue":
            self._bags: List[BagRecord] = [
                {
                    "slide_name": tissue["slide_name"],
                    "class_name": tissue["class_name"],
                    "slide_key": tissue["slide_key"],
                    "tissues": [tissue],
                }
                for tissue in discovered
            ]
        else:
            self._bags = slide_records
        self.indices = [
            index
            for index, bag in enumerate(self._bags)
            if bag["slide_key"] in selected_keys
        ]

    def __len__(self) -> int:
        """Return the number of bags in this split.

        Args:
            None: This method takes no arguments.

        Returns:
            int: Number of split-local bags.
        """
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by deterministic training sampling.

        Args:
            epoch (int): Nonnegative epoch.

        Returns:
            None: Epoch state is updated in place.
        """
        if epoch < 0:
            raise ValueError("epoch must be nonnegative.")
        self.epoch = int(epoch)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load one aligned, normalized, optionally capped bag.

        Args:
            idx (int): Split-local bag index.

        Returns:
            Dict[str, Any]: Features, label, coordinates, indices, boundaries,
                metadata, and sampling provenance.
        """
        bag = self._bags[self.indices[idx]]
        feature_parts: List[torch.Tensor] = []
        coordinate_parts: List[torch.Tensor] = []
        tile_parts: List[torch.Tensor] = []
        tissue_parts: List[torch.Tensor] = []
        feature_paths: List[str] = []
        tiles_paths: List[str] = []
        for tissue_index, tissue in enumerate(bag["tissues"]):
            features = _load_features(
                Path(tissue["feature_path"]), self.expected_feature_dim
            )
            coordinates = _load_coordinates(Path(tissue["tiles_path"]))
            if features.shape[0] != coordinates.shape[0]:
                raise ValueError(
                    f"Feature/coordinate row mismatch for '{tissue['feature_path']}': "
                    f"{features.shape[0]} features versus {coordinates.shape[0]} CSV rows."
                )
            feature_parts.append(features)
            coordinate_parts.append(coordinates)
            tile_parts.append(torch.arange(features.shape[0], dtype=torch.long))
            tissue_parts.append(
                torch.full((features.shape[0],), tissue_index, dtype=torch.long)
            )
            feature_paths.append(tissue["feature_path"])
            tiles_paths.append(tissue["tiles_path"])
        features = torch.cat(feature_parts)
        coordinates = torch.cat(coordinate_parts)
        tile_indices = torch.cat(tile_parts)
        tissue_indices = torch.cat(tissue_parts)
        seed = _stable_seed(
            self.random_seed,
            bag["slide_key"],
            bag["tissues"][0]["tissue_name"] if self.bag_level == "tissue" else "",
            self.epoch if self.split == "train" else 0,
        )
        selected = _select_tile_indices(
            features.shape[0], self.max_tiles_per_bag, self.tile_sampling, seed
        )
        features = _normalize_features(features[selected], self.feature_normalization)
        coordinates = coordinates[selected]
        tile_indices = tile_indices[selected]
        tissue_indices = tissue_indices[selected]
        tissue_names = [item["tissue_name"] for item in bag["tissues"]]
        tissue_slices = _tissue_slices(tissue_indices, len(tissue_names))
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
            "label": self.class_to_idx[bag["class_name"]],
            "slide_name": bag["slide_name"],
            "tissue_name": tissue_name,
            "tissue_names": tissue_names,
            "tissue_slices": tissue_slices,
            "coordinates": coordinates,
            "coords": coordinates,
            "tile_indices": tile_indices,
            "tissue_indices": tissue_indices,
            "num_tissues": len(tissue_names),
            "provenance": provenance,
        }


def create_bag_dataset(
    config: Mapping[str, Any],
    split: str,
    class_folders: Optional[Sequence[str]] = None,
    **overrides: Any,
) -> WSIBagDataset:
    """Create a bag dataset from resolved configuration.

    Args:
        config (Mapping[str, Any]): Resolved logistic-regression configuration.
        split (str): Train, validation, or test split.
        class_folders (Optional[Sequence[str]]): Optional ordered classes.
        **overrides (Any): Dataset keyword overrides.

    Returns:
        WSIBagDataset: Configured independent bag dataset.
    """
    caps = config.get("max_tiles_per_bag", {})
    if not isinstance(caps, Mapping):
        raise ValueError("max_tiles_per_bag must be a mapping.")
    arguments: Dict[str, Any] = {
        "data_root": config["data_root"],
        "class_folders": class_folders,
        "split": split,
        "train_ratio": config["train_ratio"],
        "val_ratio": config["val_ratio"],
        "test_ratio": config["test_ratio"],
        "random_seed": config["random_seed"],
        "feature_file_suffix": config["feature_file_suffix"],
        "expected_feature_dim": config["input_dim"],
        "bag_level": config["bag_level"],
        "max_tiles_per_bag": caps.get(split),
        "tile_sampling": config["tile_sampling"],
        "feature_normalization": config["feature_normalization"],
    }
    arguments.update(overrides)
    return WSIBagDataset(**arguments)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length bags while preserving aligned provenance.

    Args:
        batch (List[Dict[str, Any]]): Nonempty dataset sample list.

    Returns:
        Dict[str, Any]: Padded features, masks, labels, coordinates, source
            indices, bag metadata, and provenance.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    dimensions = {int(item["features"].shape[1]) for item in batch}
    if len(dimensions) != 1:
        raise ValueError("All bags in a batch must share one feature dimension.")
    batch_size = len(batch)
    max_tiles = max(int(item["features"].shape[0]) for item in batch)
    feature_dim = next(iter(dimensions))
    features = torch.zeros((batch_size, max_tiles, feature_dim), dtype=torch.float32)
    masks = torch.zeros((batch_size, max_tiles), dtype=torch.bool)
    coordinates = torch.full(
        (batch_size, max_tiles, 2), float("nan"), dtype=torch.float32
    )
    tile_indices = torch.full((batch_size, max_tiles), -1, dtype=torch.long)
    tissue_indices = torch.full((batch_size, max_tiles), -1, dtype=torch.long)
    for batch_index, item in enumerate(batch):
        length = int(item["features"].shape[0])
        features[batch_index, :length] = item["features"]
        masks[batch_index, :length] = True
        coordinates[batch_index, :length] = item["coordinates"]
        tile_indices[batch_index, :length] = item["tile_indices"]
        tissue_indices[batch_index, :length] = item["tissue_indices"]
    metadata = [
        {
            "slide_name": item["slide_name"],
            "tissue_name": item["tissue_name"],
            "tissue_names": item["tissue_names"],
            "tissue_slices": item["tissue_slices"],
            "num_tissues": item["num_tissues"],
        }
        for item in batch
    ]
    return {
        "features": features,
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "masks": masks,
        "coordinates": coordinates,
        "coords": coordinates,
        "tile_indices": tile_indices,
        "tissue_indices": tissue_indices,
        "slide_names": [item["slide_name"] for item in batch],
        "tissue_names": [item["tissue_name"] for item in batch],
        "bag_tissue_names": [item["tissue_names"] for item in batch],
        "tissue_slices": [item["tissue_slices"] for item in batch],
        "metadata": metadata,
        "provenance": [item["provenance"] for item in batch],
    }


def _discover_tissues(
    root: Path, class_folders: Sequence[str], suffix: str
) -> List[TissueRecord]:
    """Discover aligned feature/CSV pairs in stable order.

    Args:
        root (Path): Dataset root.
        class_folders (Sequence[str]): Ordered class directories.
        suffix (str): Selected feature suffix.

    Returns:
        List[TissueRecord]: Stable tissue source records.
    """
    records: List[TissueRecord] = []
    for class_name in class_folders:
        class_path = root / class_name
        if not class_path.is_dir():
            continue
        for slide_path in sorted(
            (path for path in class_path.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            for feature_path in sorted(slide_path.glob(f"*{suffix}")):
                tissue_name = feature_path.name[: -len(suffix)]
                tiles_path = slide_path / f"{tissue_name}_tiles.csv"
                if tiles_path.is_file():
                    records.append(
                        {
                            "slide_name": slide_path.name,
                            "tissue_name": tissue_name,
                            "feature_path": str(feature_path),
                            "tiles_path": str(tiles_path),
                            "class_name": class_name,
                            "slide_key": f"{class_name}/{slide_path.name}",
                        }
                    )
    return records


def _group_by_slide(tissues: Sequence[TissueRecord]) -> List[BagRecord]:
    """Group tissue records into class-qualified slide bags.

    Args:
        tissues (Sequence[TissueRecord]): Stable tissue records.

    Returns:
        List[BagRecord]: Stable slide records.
    """
    grouped: Dict[str, List[TissueRecord]] = {}
    for tissue in tissues:
        grouped.setdefault(tissue["slide_key"], []).append(tissue)
    return [
        {
            "slide_name": grouped[key][0]["slide_name"],
            "class_name": grouped[key][0]["class_name"],
            "slide_key": key,
            "tissues": grouped[key],
        }
        for key in sorted(grouped)
    ]


def _split_slide_indices(
    labels: Sequence[int],
    ratios: Tuple[float, float, float],
    random_seed: int,
) -> Dict[str, List[int]]:
    """Duplicate CLAM's stratified slide split and rare-class fallback.

    Args:
        labels (Sequence[int]): Labels for stably ordered slides.
        ratios (Tuple[float, float, float]): Train, validation, and test ratios.
        random_seed (int): Deterministic split seed.

    Returns:
        Dict[str, List[int]]: Slide indices keyed by split.
    """
    if not labels:
        return {split: [] for split in SPLITS}
    indices = np.arange(len(labels))
    try:
        train_indices, remaining = train_test_split(
            indices,
            train_size=ratios[0],
            random_state=random_seed,
            stratify=np.asarray(labels),
        )
        if ratios[1] == 0.0:
            val_indices = np.asarray([], dtype=int)
            test_indices = remaining
        elif ratios[2] == 0.0:
            val_indices = remaining
            test_indices = np.asarray([], dtype=int)
        else:
            val_indices, test_indices = train_test_split(
                remaining,
                train_size=ratios[1] / (ratios[1] + ratios[2]),
                random_state=random_seed + 1,
                stratify=np.asarray(labels)[remaining],
            )
        return {
            "train": sorted(int(index) for index in train_indices),
            "val": sorted(int(index) for index in val_indices),
            "test": sorted(int(index) for index in test_indices),
        }
    except ValueError:
        return _rare_safe_split(labels, ratios, random_seed)


def _rare_safe_split(
    labels: Sequence[int],
    ratios: Tuple[float, float, float],
    random_seed: int,
) -> Dict[str, List[int]]:
    """Allocate rare classes deterministically while honoring global sizes.

    Args:
        labels (Sequence[int]): Class label per slide.
        ratios (Tuple[float, float, float]): Split ratios.
        random_seed (int): Tie-breaking seed.

    Returns:
        Dict[str, List[int]]: Disjoint slide indices by split.
    """
    target = _allocate_counts(len(labels), ratios)
    assigned: List[List[int]] = [[], [], []]
    global_counts = [0, 0, 0]
    grouped: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(index)
    rng = np.random.default_rng(random_seed)
    for label in sorted(grouped, key=lambda value: (len(grouped[value]), value)):
        class_indices = list(grouped[label])
        rng.shuffle(class_indices)
        class_counts = [0, 0, 0]
        for position, index in enumerate(class_indices, start=1):
            available = [
                split_index
                for split_index in range(3)
                if global_counts[split_index] < target[split_index]
            ]
            chosen = max(
                available,
                key=lambda split_index: (
                    ratios[split_index] * position - class_counts[split_index],
                    target[split_index] - global_counts[split_index],
                    -split_index,
                ),
            )
            assigned[chosen].append(index)
            global_counts[chosen] += 1
            class_counts[chosen] += 1
    return {
        split: sorted(assigned[index]) for index, split in enumerate(SPLITS)
    }


def _allocate_counts(
    total: int, ratios: Tuple[float, float, float]
) -> List[int]:
    """Convert split ratios into exact integer counts.

    Args:
        total (int): Number of slides.
        ratios (Tuple[float, float, float]): Split ratios.

    Returns:
        List[int]: Three counts summing to total.
    """
    raw = [total * ratio for ratio in ratios]
    counts = [int(np.floor(value)) for value in raw]
    order = sorted(
        range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True
    )
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return counts


def _validate_ratios(ratios: Tuple[float, float, float]) -> None:
    """Validate split ratios.

    Args:
        ratios (Tuple[float, float, float]): Train, validation, and test ratios.

    Returns:
        None: Validation succeeds by returning normally.
    """
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in ratios):
        raise ValueError("Split ratios must be finite values between zero and one.")
    if abs(sum(ratios) - 1.0) > 1e-8 or ratios[0] <= 0.0:
        raise ValueError("Split ratios must sum to one with positive train_ratio.")


def _load_features(path: Path, expected_dim: Optional[int]) -> torch.Tensor:
    """Load and validate one feature tensor.

    Args:
        path (Path): PyTorch feature path.
        expected_dim (Optional[int]): Required feature width.

    Returns:
        torch.Tensor: Finite nonempty float tensor shaped ``[tiles, features]``.
    """
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, Mapping) and "features" in loaded:
        loaded = loaded["features"]
    features = (
        loaded.detach().to(dtype=torch.float32, device="cpu")
        if isinstance(loaded, torch.Tensor)
        else torch.as_tensor(loaded, dtype=torch.float32)
    )
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"Feature file '{path}' must contain a nonempty 2D tensor.")
    if expected_dim is not None and features.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch for '{path}': expected {expected_dim}, "
            f"received {features.shape[1]}."
        )
    if not torch.isfinite(features).all():
        raise ValueError(f"Feature file '{path}' contains non-finite values.")
    return features.contiguous()


def _load_coordinates(path: Path) -> torch.Tensor:
    """Load x/y coordinates preserving CSV row order.

    Args:
        path (Path): CSV containing x and y columns.

    Returns:
        torch.Tensor: Coordinate tensor shaped ``[tiles, 2]``.
    """
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not {"x", "y"}.issubset(reader.fieldnames):
            raise ValueError(f"Coordinate CSV '{path}' must contain x and y columns.")
        try:
            rows = [(float(row["x"]), float(row["y"])) for row in reader]
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"Coordinate CSV '{path}' has invalid x/y data.") from error
    coordinates = torch.tensor(rows, dtype=torch.float32).reshape(-1, 2)
    if not torch.isfinite(coordinates).all():
        raise ValueError(f"Coordinate CSV '{path}' contains non-finite values.")
    return coordinates


def _select_tile_indices(
    tile_count: int, max_tiles: Optional[int], method: str, seed: int
) -> torch.Tensor:
    """Select deterministic, source-ordered tile rows.

    Args:
        tile_count (int): Source tile count.
        max_tiles (Optional[int]): Retained cap or ``None``.
        method (str): Random, uniform, or first selection.
        seed (int): Random method seed.

    Returns:
        torch.Tensor: Sorted source row indices.
    """
    if max_tiles is None or tile_count <= max_tiles:
        return torch.arange(tile_count, dtype=torch.long)
    if method == "first":
        return torch.arange(max_tiles, dtype=torch.long)
    if method == "uniform":
        return torch.linspace(0, tile_count - 1, max_tiles).round().long()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.sort(torch.randperm(tile_count, generator=generator)[:max_tiles]).values


def _normalize_features(features: torch.Tensor, method: str) -> torch.Tensor:
    """Normalize raw tile feature rows.

    Args:
        features (torch.Tensor): Tensor shaped ``[tiles, features]``.
        method (str): None, L2, or layer normalization.

    Returns:
        torch.Tensor: Normalized feature tensor.
    """
    if method == "l2":
        return F.normalize(features, p=2, dim=1)
    if method == "layer_norm":
        return F.layer_norm(features, (features.shape[1],))
    return features


def _stable_seed(base_seed: int, *parts: Any) -> int:
    """Create a process-independent sampling seed.

    Args:
        base_seed (int): User base seed.
        *parts (Any): Stable bag identifiers and epoch.

    Returns:
        int: Nonnegative torch generator seed.
    """
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63 - 1)


def _tissue_slices(
    tissue_indices: torch.Tensor, tissue_count: int
) -> List[Tuple[int, int]]:
    """Compute retained half-open tissue boundaries.

    Args:
        tissue_indices (torch.Tensor): Tissue index per retained tile.
        tissue_count (int): Number of source tissues.

    Returns:
        List[Tuple[int, int]]: Half-open row slices in tissue order.
    """
    counts = Counter(int(index) for index in tissue_indices.tolist())
    slices: List[Tuple[int, int]] = []
    start = 0
    for tissue_index in range(tissue_count):
        end = start + counts.get(tissue_index, 0)
        slices.append((start, end))
        start = end
    return slices
