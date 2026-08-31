"""Training-split k-means initialization for CLAM prototype histograms."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn

try:
    from .clam_dataset import WSIBagDataset
except ImportError:
    from clam_dataset import WSIBagDataset


MIN_PROTOTYPE_TEMPERATURE = 1e-6


def collect_training_tile_embeddings(
    model: nn.Module,
    train_dataset: WSIBagDataset,
    device: torch.device,
    max_tiles: int,
    batch_size: int,
    random_seed: int,
) -> np.ndarray:
    """Collect a deterministic sample of embedded training-split tiles.

    Args:
        model (nn.Module): CLAM model exposing an ``embedding`` module.
        train_dataset (WSIBagDataset): Training-split bag dataset.
        device (torch.device): Device used to run tile embedding.
        max_tiles (int): Maximum number of embedded tiles to retain.
        batch_size (int): Maximum tile count per embedding operation.
        random_seed (int): Seed controlling bag order and final-bag subsampling.

    Returns:
        np.ndarray: Float32 embedded tiles shaped ``[sample_count, hidden_dim]``.
    """
    if train_dataset.split != "train":
        raise ValueError("Prototype initialization requires a training-split dataset.")
    if max_tiles <= 0 or batch_size <= 0:
        raise ValueError("max_tiles and batch_size must be positive.")
    if random_seed < 0:
        raise ValueError("random_seed must be nonnegative.")
    embedding = getattr(model, "embedding", None)
    hidden_dim = getattr(model, "hidden_dim", None)
    if not isinstance(embedding, nn.Module) or not isinstance(hidden_dim, int):
        raise TypeError("model must expose an embedding module and integer hidden_dim.")

    train_dataset.set_epoch(0)
    bag_order = np.arange(len(train_dataset), dtype=np.int64)
    generator = np.random.default_rng(random_seed)
    generator.shuffle(bag_order)
    collected = np.empty((max_tiles, hidden_dim), dtype=np.float32)
    sample_count = 0
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for bag_index in bag_order:
                features = train_dataset[int(bag_index)]["features"]
                remaining = max_tiles - sample_count
                if remaining <= 0:
                    break
                if features.shape[0] > remaining:
                    selected = np.sort(
                        generator.choice(
                            int(features.shape[0]), size=remaining, replace=False
                        )
                    )
                    features = features[torch.from_numpy(selected)]

                for start in range(0, int(features.shape[0]), batch_size):
                    feature_batch = features[start : start + batch_size].to(device)
                    embedded = embedding(feature_batch).detach().cpu().float().numpy()
                    end = sample_count + embedded.shape[0]
                    collected[sample_count:end] = embedded
                    sample_count = end
    finally:
        model.train(was_training)

    return collected[:sample_count].copy()


def fit_prototype_centroids(
    embeddings: np.ndarray,
    num_prototypes: int,
    batch_size: int,
    random_seed: int,
) -> np.ndarray:
    """Fit deterministic mini-batch k-means centroids.

    Args:
        embeddings (np.ndarray): Embedded training tiles shaped ``[N, H]``.
        num_prototypes (int): Number of morphological prototypes.
        batch_size (int): MiniBatchKMeans update batch size.
        random_seed (int): Random state used by k-means.

    Returns:
        np.ndarray: Float32 prototype centroids shaped ``[K, H]``.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] < num_prototypes:
        raise ValueError(
            "Prototype initialization needs a two-dimensional embedding matrix "
            "with at least one tile per prototype."
        )
    if num_prototypes <= 0 or batch_size <= 0:
        raise ValueError("num_prototypes and batch_size must be positive.")
    estimator = MiniBatchKMeans(
        n_clusters=num_prototypes,
        batch_size=max(batch_size, num_prototypes),
        n_init=10,
        random_state=random_seed,
    )
    estimator.fit(embeddings)
    return np.asarray(estimator.cluster_centers_, dtype=np.float32)


def estimate_prototype_temperature(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    batch_size: int,
) -> float:
    """Estimate assignment temperature from nearest-centroid squared distances.

    Args:
        embeddings (np.ndarray): Embedded tiles shaped ``[N, H]``.
        centroids (np.ndarray): K-means centroids shaped ``[K, H]``.
        batch_size (int): Number of tiles processed per distance block.

    Returns:
        float: Positive mean nearest-centroid squared distance.
    """
    if embeddings.ndim != 2 or centroids.ndim != 2:
        raise ValueError("embeddings and centroids must be two-dimensional.")
    if embeddings.shape[1] != centroids.shape[1] or embeddings.shape[0] == 0:
        raise ValueError("embeddings and centroids must have matching nonempty widths.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    centroid_norms = np.square(centroids, dtype=np.float64).sum(axis=1)
    nearest_distance_sum = 0.0
    for start in range(0, embeddings.shape[0], batch_size):
        block = embeddings[start : start + batch_size].astype(np.float64, copy=False)
        squared_distances = (
            np.square(block).sum(axis=1, keepdims=True)
            + centroid_norms[None, :]
            - 2.0 * block @ centroids.astype(np.float64, copy=False).T
        )
        nearest_distance_sum += float(
            np.maximum(squared_distances.min(axis=1), 0.0).sum()
        )
    estimated = nearest_distance_sum / float(embeddings.shape[0])
    return max(estimated, MIN_PROTOTYPE_TEMPERATURE)


def initialize_prototype_assignment(
    model: nn.Module,
    centroids: np.ndarray,
    temperature: float,
    freeze_prototypes: bool,
) -> None:
    """Initialize a CLAM assignment layer from k-means centroids.

    Args:
        model (nn.Module): CLAM model with prototype assignment parameters.
        centroids (np.ndarray): K-means centroids shaped ``[K, H]``.
        temperature (float): Positive soft-assignment temperature.
        freeze_prototypes (bool): Whether assignment weights and bias stay fixed.

    Returns:
        None: Prototype parameters are updated in place.
    """
    assignment = getattr(model, "prototype_assignment", None)
    log_temperature = getattr(model, "log_prototype_temperature", None)
    if not isinstance(assignment, nn.Linear) or not isinstance(
        log_temperature, nn.Parameter
    ):
        raise ValueError("Model does not have an enabled prototype histogram.")
    if centroids.shape != tuple(assignment.weight.shape):
        raise ValueError(
            "Centroid shape must match prototype assignment weight shape."
        )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite.")

    centroid_tensor = torch.as_tensor(
        centroids,
        dtype=assignment.weight.dtype,
        device=assignment.weight.device,
    )
    with torch.no_grad():
        assignment.weight.copy_(2.0 * centroid_tensor)
        assignment.bias.copy_(-centroid_tensor.square().sum(dim=1))
        log_temperature.fill_(math.log(temperature))
    assignment.weight.requires_grad_(not freeze_prototypes)
    assignment.bias.requires_grad_(not freeze_prototypes)


def initialize_prototypes_from_kmeans(
    model: nn.Module,
    train_dataset: WSIBagDataset,
    config: Mapping[str, Any],
    device: torch.device,
) -> None:
    """Initialize an enabled prototype histogram from training tiles only.

    Args:
        model (nn.Module): Newly created CLAM model.
        train_dataset (WSIBagDataset): Training-split dataset.
        config (Mapping[str, Any]): Resolved CLAM configuration.
        device (torch.device): Device used for embedding collection.

    Returns:
        None: Enabled prototype parameters are initialized in place.
    """
    num_prototypes = int(config.get("pooling_num_prototypes", 0))
    if num_prototypes == 0:
        return
    max_tiles = int(config.get("prototype_kmeans_max_tiles", 200_000))
    batch_size = int(config.get("prototype_kmeans_batch_size", 4_096))
    random_seed = int(config["random_seed"])
    embeddings = collect_training_tile_embeddings(
        model=model,
        train_dataset=train_dataset,
        device=device,
        max_tiles=max_tiles,
        batch_size=batch_size,
        random_seed=random_seed,
    )
    if embeddings.shape[0] < num_prototypes:
        raise ValueError(
            f"Only {embeddings.shape[0]} training tiles were available for "
            f"{num_prototypes} prototypes."
        )
    centroids = fit_prototype_centroids(
        embeddings=embeddings,
        num_prototypes=num_prototypes,
        batch_size=batch_size,
        random_seed=random_seed,
    )
    configured_temperature = config.get("pooling_prototype_temperature")
    temperature = (
        estimate_prototype_temperature(embeddings, centroids, batch_size)
        if configured_temperature is None
        else float(configured_temperature)
    )
    initialize_prototype_assignment(
        model=model,
        centroids=centroids,
        temperature=temperature,
        freeze_prototypes=bool(config.get("pooling_freeze_prototypes", False)),
    )
    print(
        "Initialized prototype histogram from "
        f"{embeddings.shape[0]} training tiles "
        f"(K={num_prototypes}, temperature={temperature:.6g})."
    )
