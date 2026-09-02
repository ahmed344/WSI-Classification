"""Independent raw-feature logistic-regression baseline package."""

from .config_loader import (
    allocate_training_run,
    load_config,
    resolve_inference_run_paths,
)
from .dataset import WSIBagDataset, collate_fn, create_bag_dataset
from .model import (
    DEFAULT_POOLING_STATISTICS,
    RawFeatureStatisticsPooler,
    TorchLogisticRegression,
    normalize_pooling_statistics,
    pool_raw_features,
    pooling_contract,
    pooling_output_dim,
)

__all__ = [
    "DEFAULT_POOLING_STATISTICS",
    "RawFeatureStatisticsPooler",
    "TorchLogisticRegression",
    "WSIBagDataset",
    "allocate_training_run",
    "collate_fn",
    "create_bag_dataset",
    "load_config",
    "normalize_pooling_statistics",
    "pool_raw_features",
    "pooling_contract",
    "pooling_output_dim",
    "resolve_inference_run_paths",
]
