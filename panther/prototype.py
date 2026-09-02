"""Training-only prototype sampling and CPU K-means construction."""

from __future__ import annotations

import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from sklearn.cluster import KMeans

from .panther_dataset import PantherSlideDataset, stable_seed


def fit_prototypes(
    dataset: PantherSlideDataset,
    config: Mapping[str, Any],
    destination: Path,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Sample slides evenly and fit the configured full sklearn K-means model."""
    settings = config["prototype"]
    num_prototypes = int(settings["num_prototypes"])
    target = num_prototypes * int(settings["patches_per_prototype"])
    per_slide = math.ceil(target / len(dataset))
    sampled = torch.empty((target, dataset.input_dim), dtype=torch.float32)
    offset = 0
    started = time.time()
    for index in range(len(dataset)):
        if offset >= target:
            break
        item = dataset[index]
        features = item["features"]
        generator = torch.Generator().manual_seed(
            stable_seed(int(config["random_seed"]), item["slide_key"], "prototype")
        )
        take = min(per_slide, int(features.shape[0]), target - offset)
        selected = torch.randperm(int(features.shape[0]), generator=generator)[:take]
        sampled[offset : offset + take] = features[selected]
        offset += take
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            print(
                f"Prototype sampling: {index + 1}/{len(dataset)} slides, "
                f"{offset:,}/{target:,} patches"
            )
    if offset < num_prototypes:
        raise ValueError(
            f"Only {offset} training patches were sampled for {num_prototypes} prototypes."
        )
    samples = sampled[:offset].numpy()
    print(
        f"Fitting K-means to {offset:,} patches x {dataset.input_dim} features "
        f"with {num_prototypes} prototypes..."
    )
    kmeans = KMeans(
        n_clusters=num_prototypes,
        max_iter=int(settings["max_iterations"]),
        n_init=int(settings["num_initializations"]),
        algorithm=str(settings["algorithm"]),
        random_state=int(config["random_seed"]),
        copy_x=False,
        verbose=1,
    )
    kmeans.fit(samples)
    prototypes = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32, copy=False))
    metadata: Dict[str, Any] = {
        "method": "sklearn.cluster.KMeans",
        "num_prototypes": num_prototypes,
        "feature_dim": dataset.input_dim,
        "sampled_patches": offset,
        "target_patches": target,
        "patches_per_slide_limit": per_slide,
        "inertia": float(kmeans.inertia_),
        "iterations": int(kmeans.n_iter_),
        "elapsed_seconds": float(time.time() - started),
        "random_seed": int(config["random_seed"]),
        "feature_normalization": str(config["feature_normalization"]),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Match the official artifact convention: {'prototypes': [1, C, D]}.
    with destination.open("wb") as handle:
        pickle.dump(
            {
                "prototypes": prototypes.numpy()[None, ...],
                "metadata": metadata,
            },
            handle,
        )
    return prototypes, metadata


def load_prototypes(path: Path) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Load this pipeline's official-format prototype pickle."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    array = np.asarray(payload["prototypes"]).squeeze(0)
    return torch.as_tensor(array, dtype=torch.float32), dict(payload.get("metadata", {}))
