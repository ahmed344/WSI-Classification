"""Independent raw-feature logistic-regression baseline package."""

from .config_loader import (
    allocate_training_run,
    load_config,
    resolve_inference_run_paths,
)
from .dataset import WSIBagDataset, collate_fn, create_bag_dataset
from .model import (
    RawFeatureStatisticsPooler,
    TorchLogisticRegression,
    pool_raw_features,
)

__all__ = [
    "RawFeatureStatisticsPooler",
    "TorchLogisticRegression",
    "WSIBagDataset",
    "allocate_training_run",
    "collate_fn",
    "create_bag_dataset",
    "load_config",
    "pool_raw_features",
    "resolve_inference_run_paths",
]
