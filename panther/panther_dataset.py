"""Independent tissue/slide bag discovery, splitting, and loading for PANTHER."""

from __future__ import annotations

import csv
import hashlib
import json
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


class TissueRecord(TypedDict):
    tissue_name: str
    feature_path: str
    tiles_path: str


class SlideRecord(TypedDict):
    slide_name: str
    slide_key: str
    class_name: str
    label: int
    split: str
    tissues: List[TissueRecord]


class PantherBagDataset(Dataset):
    """Load one variable-length tissue or concatenated-slide feature bag."""

    def __init__(
        self,
        records: Sequence[SlideRecord],
        input_dim: int,
        feature_normalization: str = "none",
        max_tiles_per_slide: Optional[int] = None,
        tile_sampling: str = "random",
        random_seed: int = 42,
        bag_level: str = "slide",
    ) -> None:
        if bag_level not in BAG_LEVELS:
            raise ValueError(
                f"bag_level must be one of {BAG_LEVELS}, received {bag_level}."
            )
        self.records = list(records)
        self.input_dim = int(input_dim)
        self.feature_normalization = feature_normalization
        self.max_tiles_per_slide = max_tiles_per_slide
        self.tile_sampling = tile_sampling
        self.random_seed = int(random_seed)
        self.bag_level = bag_level

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        parts = [
            _load_feature_tensor(Path(tissue["feature_path"]), self.input_dim)
            for tissue in record["tissues"]
        ]
        features = torch.cat(parts, dim=0)
        selected = select_tile_indices(
            int(features.shape[0]),
            self.max_tiles_per_slide,
            self.tile_sampling,
            stable_seed(self.random_seed, _bag_key(record, self.bag_level)),
        )
        features = features[selected]
        if self.feature_normalization == "l2":
            features = F.normalize(features, p=2, dim=1)
        elif self.feature_normalization != "none":
            raise ValueError(
                f"Unsupported feature_normalization: {self.feature_normalization}"
            )
        if not torch.isfinite(features).all():
            raise ValueError(f"Non-finite features in slide {record['slide_key']}.")
        return {
            "features": features,
            "label": int(record["label"]),
            "slide_name": record["slide_name"],
            "slide_key": record["slide_key"],
            "bag_name": _bag_name(record, self.bag_level),
            "bag_key": _bag_key(record, self.bag_level),
            "bag_level": self.bag_level,
            "tissue_names": [
                tissue["tissue_name"] for tissue in record["tissues"]
            ],
            "class_name": record["class_name"],
            "num_tissues": len(record["tissues"]),
            "num_tiles": int(features.shape[0]),
        }


# Backward-compatible public name used by visualization and downstream callers.
PantherSlideDataset = PantherBagDataset


def build_datasets(
    config: Mapping[str, Any],
    class_folders: Optional[Sequence[str]] = None,
    split_assignments: Optional[Mapping[str, str]] = None,
    bag_level: str = "slide",
) -> tuple[Dict[str, PantherBagDataset], List[str], List[SlideRecord]]:
    """Discover slides and construct deterministic tissue- or slide-bag datasets."""
    if bag_level not in BAG_LEVELS:
        raise ValueError(
            f"bag_level must be one of {BAG_LEVELS}, received {bag_level}."
        )
    classes, discovered = discover_slide_records(
        Path(str(config["data_root"])),
        str(config["feature_file_suffix"]),
        class_folders if class_folders is not None else config.get("class_folders"),
    )
    labels = [int(record["label"]) for record in discovered]
    if split_assignments is None:
        indices = split_slide_indices(
            labels,
            (
                float(config["train_ratio"]),
                float(config["val_ratio"]),
                float(config["test_ratio"]),
            ),
            int(config["random_seed"]),
        )
        assignments = {
            discovered[index]["slide_key"]: split
            for split, members in indices.items()
            for index in members
        }
    else:
        assignments = dict(split_assignments)
        found = {record["slide_key"] for record in discovered}
        missing = sorted(set(assignments) - found)
        unexpected = sorted(found - set(assignments))
        if missing or unexpected:
            raise ValueError(
                "Current data do not match the saved split manifest: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}."
            )
        if any(value not in SPLITS for value in assignments.values()):
            raise ValueError("Saved split assignments contain an invalid split.")

    records: List[SlideRecord] = []
    for source in discovered:
        record: SlideRecord = {**source, "split": assignments[source["slide_key"]]}
        records.append(record)

    common = {
        "input_dim": int(config["input_dim"]),
        "feature_normalization": str(config["feature_normalization"]),
        "max_tiles_per_slide": config.get("max_tiles_per_slide"),
        "tile_sampling": str(config["tile_sampling"]),
        "random_seed": int(config["random_seed"]),
    }
    datasets = {
        split: PantherBagDataset(
            _records_at_level(
                [record for record in records if record["split"] == split],
                bag_level,
            ),
            bag_level=bag_level,
            **common,
        )
        for split in SPLITS
    }
    return datasets, classes, records


def discover_slide_records(
    root: Path,
    feature_suffix: str,
    class_folders: Optional[Sequence[str]] = None,
) -> tuple[List[str], List[SlideRecord]]:
    """Find feature/coordinate pairs and group tissues into sorted slide bags."""
    if not root.is_dir():
        raise ValueError(f"data_root is not a directory: {root}")
    classes = (
        list(class_folders)
        if class_folders is not None
        else sorted(path.name for path in root.iterdir() if path.is_dir())
    )
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("No unique class directories were configured/discovered.")
    class_to_idx = {name: index for index, name in enumerate(classes)}
    records: List[SlideRecord] = []
    for class_name in classes:
        class_path = root / class_name
        if not class_path.is_dir():
            raise ValueError(f"Class directory does not exist: {class_path}")
        for slide_path in sorted(
            (path for path in class_path.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            tissues: List[TissueRecord] = []
            for feature_path in sorted(slide_path.glob(f"*{feature_suffix}")):
                tissue_name = feature_path.name[: -len(feature_suffix)]
                tiles_path = slide_path / f"{tissue_name}_tiles.csv"
                # Requiring the coordinate file matches CLAM's bag discovery contract.
                if tiles_path.is_file():
                    tissues.append(
                        {
                            "tissue_name": tissue_name,
                            "feature_path": str(feature_path.resolve()),
                            "tiles_path": str(tiles_path.resolve()),
                        }
                    )
            if tissues:
                records.append(
                    {
                        "slide_name": slide_path.name,
                        "slide_key": f"{class_name}/{slide_path.name}",
                        "class_name": class_name,
                        "label": class_to_idx[class_name],
                        "split": "",
                        "tissues": tissues,
                    }
                )
    if not records:
        raise ValueError(
            f"No '*{feature_suffix}' and matching '*_tiles.csv' pairs found in {root}."
        )
    return classes, records


def split_slide_indices(
    labels: Sequence[int],
    ratios: Tuple[float, float, float],
    random_seed: int,
) -> Dict[str, List[int]]:
    """Use CLAM-compatible stratification with a deterministic rare-class fallback."""
    if not labels:
        return {split: [] for split in SPLITS}
    all_indices = np.arange(len(labels))
    try:
        train, remaining = train_test_split(
            all_indices,
            train_size=ratios[0],
            random_state=random_seed,
            stratify=np.asarray(labels),
        )
        if ratios[1] == 0.0:
            val, test = np.asarray([], dtype=int), remaining
        elif ratios[2] == 0.0:
            val, test = remaining, np.asarray([], dtype=int)
        else:
            val, test = train_test_split(
                remaining,
                train_size=ratios[1] / (ratios[1] + ratios[2]),
                random_state=random_seed + 1,
                stratify=np.asarray(labels)[remaining],
            )
        return {
            "train": sorted(int(index) for index in train),
            "val": sorted(int(index) for index in val),
            "test": sorted(int(index) for index in test),
        }
    except ValueError:
        return _rare_safe_split(labels, ratios, random_seed)


def _rare_safe_split(
    labels: Sequence[int],
    ratios: Tuple[float, float, float],
    random_seed: int,
) -> Dict[str, List[int]]:
    target = _allocate_counts(len(labels), ratios)
    assigned: List[List[int]] = [[], [], []]
    global_counts = [0, 0, 0]
    grouped: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(index)
    rng = np.random.default_rng(random_seed)
    for label in sorted(grouped, key=lambda item: (len(grouped[item]), item)):
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


def _allocate_counts(total: int, ratios: Tuple[float, float, float]) -> List[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(np.floor(value)) for value in raw]
    order = sorted(
        range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True
    )
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return counts


def save_split_manifest(
    records: Sequence[SlideRecord], class_folders: Sequence[str], run_dir: Path
) -> tuple[Path, Path]:
    """Persist the exact data partition used for prototype fitting and training."""
    csv_path = run_dir / "split_manifest.csv"
    json_path = run_dir / "split_manifest.json"
    rows = [
        {
            "slide_key": record["slide_key"],
            "slide_name": record["slide_name"],
            "class_name": record["class_name"],
            "label": int(record["label"]),
            "split": record["split"],
            "num_tissues": len(record["tissues"]),
            "feature_paths": json.dumps(
                [tissue["feature_path"] for tissue in record["tissues"]]
            ),
        }
        for record in records
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "class_folders": list(class_folders),
        "assignments": {record["slide_key"]: record["split"] for record in records},
        "counts": {
            split: dict(
                Counter(
                    record["class_name"]
                    for record in records
                    if record["split"] == split
                )
            )
            for split in SPLITS
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return csv_path, json_path


def load_split_manifest(path: Path) -> tuple[List[str], Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    classes = payload.get("class_folders")
    assignments = payload.get("assignments")
    if not isinstance(classes, list) or not isinstance(assignments, dict):
        raise ValueError(f"Invalid split manifest: {path}")
    return [str(name) for name in classes], {
        str(key): str(value) for key, value in assignments.items()
    }


def class_counts(dataset: PantherBagDataset) -> Dict[str, int]:
    return dict(Counter(record["class_name"] for record in dataset.records))


def _records_at_level(
    records: Sequence[SlideRecord], bag_level: str
) -> List[SlideRecord]:
    """Expand slide records into one-record-per-tissue bags when requested."""
    if bag_level == "slide":
        return list(records)
    return [
        {**record, "tissues": [tissue]}
        for record in records
        for tissue in record["tissues"]
    ]


def _bag_name(record: SlideRecord, bag_level: str) -> str:
    if bag_level == "tissue":
        return str(record["tissues"][0]["tissue_name"])
    return str(record["slide_name"])


def _bag_key(record: SlideRecord, bag_level: str) -> str:
    if bag_level == "tissue":
        return f"{record['slide_key']}/{record['tissues'][0]['tissue_name']}"
    return str(record["slide_key"])


def select_tile_indices(
    total: int, maximum: Optional[int], method: str, seed: int
) -> torch.Tensor:
    if maximum is None or total <= maximum:
        return torch.arange(total, dtype=torch.long)
    if method == "first":
        return torch.arange(maximum, dtype=torch.long)
    if method == "uniform":
        return torch.linspace(0, total - 1, maximum).round().to(torch.long).unique()
    if method == "random":
        generator = torch.Generator().manual_seed(seed)
        return torch.randperm(total, generator=generator)[:maximum].sort().values
    raise ValueError(f"Unsupported tile sampling method: {method}")


def stable_seed(base_seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        "::".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _load_feature_tensor(path: Path, input_dim: int) -> torch.Tensor:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, Mapping) and "features" in loaded:
        loaded = loaded["features"]
    features = (
        loaded.detach().to(dtype=torch.float32, device="cpu")
        if isinstance(loaded, torch.Tensor)
        else torch.as_tensor(loaded, dtype=torch.float32)
    )
    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Feature file must contain [tiles, dim]: {path}")
    if int(features.shape[1]) != input_dim:
        raise ValueError(
            f"Feature dimension mismatch in {path}: expected {input_dim}, "
            f"received {features.shape[1]}."
        )
    return features.contiguous()
