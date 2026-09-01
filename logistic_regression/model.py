"""Raw-feature statistics pooling and CUDA logistic regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import joblib
import numpy as np
import torch
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, torch.Tensor]


class RawFeatureStatisticsPooler:
    """Pool tiles into mask-aware population mean and standard deviation.

    Args:
        epsilon (float): Nonnegative variance floor applied before square root.

    Returns:
        RawFeatureStatisticsPooler: Callable stateless pooling object.
    """

    def __init__(self, epsilon: float = 0.0) -> None:
        """Initialize the raw-feature pooler.

        Args:
            epsilon (float): Nonnegative variance floor before square root.

        Returns:
            None: Pooling configuration is stored in place.
        """
        if epsilon < 0.0:
            raise ValueError("epsilon must be nonnegative.")
        self.epsilon = float(epsilon)

    def __call__(
        self, features: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool padded or unpadded bags into ``[mean || population_std]``.

        Args:
            features (torch.Tensor): Features shaped ``[B, T, D]`` or ``[T, D]``.
            masks (Optional[torch.Tensor]): Valid rows shaped ``[B, T]`` or
                ``[T]``; ``None`` marks every tile valid.

        Returns:
            torch.Tensor: Bag representations shaped ``[B, 2D]``.
        """
        return pool_raw_features(features, masks=masks, epsilon=self.epsilon)


def pool_raw_features(
    features: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Compute mask-aware raw-feature population moments.

    Args:
        features (torch.Tensor): Features shaped ``[B, T, D]`` or ``[T, D]``.
        masks (Optional[torch.Tensor]): Valid rows shaped ``[B, T]`` or ``[T]``.
        epsilon (float): Nonnegative value added to population variance.

    Returns:
        torch.Tensor: Concatenated mean and standard deviation shaped ``[B, 2D]``.
    """
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative.")
    if features.ndim == 2:
        features = features.unsqueeze(0)
        if masks is not None and masks.ndim == 1:
            masks = masks.unsqueeze(0)
    if features.ndim != 3 or features.shape[1] == 0 or features.shape[2] == 0:
        raise ValueError("features must have nonempty shape [B, T, D] or [T, D].")
    if not torch.is_floating_point(features):
        features = features.float()
    if not torch.isfinite(features).all():
        raise ValueError("features contain non-finite values.")
    if masks is None:
        valid = torch.ones(
            features.shape[:2], dtype=torch.bool, device=features.device
        )
    else:
        if masks.shape != features.shape[:2]:
            raise ValueError("masks shape does not match the feature bag axes.")
        valid = masks.to(device=features.device, dtype=torch.bool)
    counts = valid.sum(dim=1, keepdim=True)
    if torch.any(counts == 0):
        raise ValueError("Every bag must contain at least one valid tile.")
    weights = valid.unsqueeze(-1).to(dtype=features.dtype)
    denominator = counts.to(dtype=features.dtype)
    mean = (features * weights).sum(dim=1) / denominator
    variance = (
        (features - mean.unsqueeze(1)).square() * weights
    ).sum(dim=1) / denominator
    standard_deviation = torch.sqrt(variance.clamp_min(0.0) + epsilon)
    return torch.cat((mean, standard_deviation), dim=1)


class TorchLogisticRegression:
    """Standardized multinomial logistic regression fitted with PyTorch.

    The full training matrix, standardization, linear classifier, and optimizer
    live on the configured CUDA device during fitting. Fitted arrays are stored
    on CPU for portable Joblib checkpoints and copied to the selected device
    for inference.

    Args:
        C (float): Positive inverse L2 regularization strength.
        device (str): Required fitting device, normally ``cuda``.
        max_iter (int): Positive L-BFGS iteration limit.
        tolerance (float): Positive gradient and parameter tolerance.
        learning_rate (float): Positive L-BFGS step size.
        class_weight (Optional[str]): ``balanced`` or ``None``.
        random_seed (int): Nonnegative PyTorch random seed.

    Returns:
        TorchLogisticRegression: Unfitted GPU estimator.
    """

    def __init__(
        self,
        C: float = 1.0,
        device: str = "cuda",
        max_iter: int = 500,
        tolerance: float = 1e-5,
        learning_rate: float = 1.0,
        class_weight: Optional[str] = "balanced",
        random_seed: int = 42,
    ) -> None:
        """Initialize one GPU regularization candidate.

        Args:
            C (float): Positive inverse L2 regularization strength.
            device (str): ``cuda`` or ``cpu`` execution device.
            max_iter (int): Positive L-BFGS iteration limit.
            tolerance (float): Positive optimizer tolerance.
            learning_rate (float): Positive L-BFGS learning rate.
            class_weight (Optional[str]): ``balanced`` or ``None``.
            random_seed (int): Nonnegative random seed.

        Returns:
            None: Estimator configuration is stored in place.
        """
        if not np.isfinite(C) or C <= 0.0:
            raise ValueError("C must be positive and finite.")
        if device not in ("cuda", "cpu"):
            raise ValueError("device must be 'cuda' or 'cpu'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA fitting was requested but CUDA is unavailable.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite.")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite.")
        if class_weight not in ("balanced", None):
            raise ValueError("class_weight must be 'balanced' or None.")
        if random_seed < 0:
            raise ValueError("random_seed must be nonnegative.")
        self.C = float(C)
        self.device = device
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.learning_rate = float(learning_rate)
        self.class_weight = class_weight
        self.random_seed = int(random_seed)
        self.classes_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.fit_device_: Optional[str] = None

    def fit(
        self, bag_features: ArrayLike, labels: ArrayLike
    ) -> "TorchLogisticRegression":
        """Fit standardization and multinomial weights on the selected device.

        Args:
            bag_features (ArrayLike): Pooled bag matrix shaped ``[N, 2D]``.
            labels (ArrayLike): Integer class labels shaped ``[N]``.

        Returns:
            TorchLogisticRegression: Fitted estimator.
        """
        matrix = _as_feature_matrix(bag_features)
        targets = _as_label_vector(labels, matrix.shape[0])
        classes, encoded = np.unique(targets, return_inverse=True)
        if classes.size < 2:
            raise ValueError("Logistic regression requires at least two classes.")
        torch.manual_seed(self.random_seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.random_seed)
        execution_device = torch.device(self.device)
        feature_tensor = torch.as_tensor(
            matrix, dtype=torch.float32, device=execution_device
        )
        target_tensor = torch.as_tensor(
            encoded, dtype=torch.long, device=execution_device
        )
        mean = feature_tensor.mean(dim=0)
        scale = feature_tensor.std(dim=0, unbiased=False)
        scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
        standardized = (feature_tensor - mean) / scale
        weights = torch.zeros(
            (classes.size, matrix.shape[1]),
            dtype=torch.float32,
            device=execution_device,
            requires_grad=True,
        )
        intercept = torch.zeros(
            classes.size,
            dtype=torch.float32,
            device=execution_device,
            requires_grad=True,
        )
        loss_weights = self._class_weights(target_tensor, classes.size)
        optimizer = torch.optim.LBFGS(
            [weights, intercept],
            lr=self.learning_rate,
            max_iter=self.max_iter,
            tolerance_grad=self.tolerance,
            tolerance_change=self.tolerance,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0

        def closure() -> torch.Tensor:
            """Compute full-batch regularized cross-entropy for L-BFGS.

            Args:
                None: The closure captures standardized training tensors.

            Returns:
                torch.Tensor: Scalar differentiable objective on the fit device.
            """
            nonlocal closure_calls
            optimizer.zero_grad()
            logits = F.linear(standardized, weights, intercept)
            classification_loss = F.cross_entropy(
                logits, target_tensor, weight=loss_weights
            )
            regularization = 0.5 * weights.square().sum() / self.C
            loss = classification_loss + regularization / matrix.shape[0]
            loss.backward()
            closure_calls += 1
            return loss

        optimizer.step(closure)
        self.classes_ = classes.astype(np.int64, copy=False)
        self.mean_ = mean.detach().cpu().numpy()
        self.scale_ = scale.detach().cpu().numpy()
        self.coef_ = weights.detach().cpu().numpy()
        self.intercept_ = intercept.detach().cpu().numpy()
        self.n_iter_ = closure_calls
        self.fit_device_ = str(execution_device)
        return self

    def _class_weights(
        self, encoded_targets: torch.Tensor, class_count: int
    ) -> Optional[torch.Tensor]:
        """Compute balanced loss weights on the fitting device.

        Args:
            encoded_targets (torch.Tensor): Contiguous training labels.
            class_count (int): Number of represented classes.

        Returns:
            Optional[torch.Tensor]: Balanced weights or ``None``.
        """
        if self.class_weight is None:
            return None
        counts = torch.bincount(encoded_targets, minlength=class_count).float()
        return encoded_targets.numel() / (class_count * counts)

    def predict(self, bag_features: ArrayLike) -> np.ndarray:
        """Predict original fixed-space class labels.

        Args:
            bag_features (ArrayLike): Pooled bag matrix shaped ``[N, 2D]``.

        Returns:
            np.ndarray: Predicted labels shaped ``[N]``.
        """
        probabilities = self.predict_proba(bag_features)
        classes = self._require_fitted()["classes"]
        return classes[np.argmax(probabilities, axis=1)]

    def predict_proba(self, bag_features: ArrayLike) -> np.ndarray:
        """Predict represented-class probabilities on the selected device.

        Args:
            bag_features (ArrayLike): Pooled bag matrix shaped ``[N, 2D]``.

        Returns:
            np.ndarray: Probabilities shaped ``[N, represented_classes]``.
        """
        matrix = _as_feature_matrix(bag_features)
        fitted = self._require_fitted()
        if matrix.shape[1] != fitted["coef"].shape[1]:
            raise ValueError("bag_features width does not match the fitted model.")
        execution_device = torch.device(self.device)
        with torch.inference_mode():
            features = torch.as_tensor(
                matrix, dtype=torch.float32, device=execution_device
            )
            mean = torch.as_tensor(
                fitted["mean"], dtype=torch.float32, device=execution_device
            )
            scale = torch.as_tensor(
                fitted["scale"], dtype=torch.float32, device=execution_device
            )
            weights = torch.as_tensor(
                fitted["coef"], dtype=torch.float32, device=execution_device
            )
            intercept = torch.as_tensor(
                fitted["intercept"], dtype=torch.float32, device=execution_device
            )
            logits = F.linear((features - mean) / scale, weights, intercept)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: str | Path) -> Path:
        """Persist the fitted estimator with Joblib.

        Args:
            path (str | Path): Destination ``.joblib`` file.

        Returns:
            Path: Absolute saved artifact path.
        """
        self._require_fitted()
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "TorchLogisticRegression":
        """Load a persisted estimator.

        Args:
            path (str | Path): Existing Joblib artifact.

        Returns:
            TorchLogisticRegression: Restored fitted estimator.
        """
        source = Path(path).expanduser().resolve()
        loaded = joblib.load(source)
        if not isinstance(loaded, cls):
            raise TypeError(f"Artifact '{source}' does not contain {cls.__name__}.")
        loaded._require_fitted()
        return loaded

    def get_metadata(self) -> Dict[str, Any]:
        """Return serializable optimizer and fitted-device metadata.

        Args:
            None: This method takes no arguments.

        Returns:
            Dict[str, Any]: Estimator settings and fitted state metadata.
        """
        metadata: Dict[str, Any] = {
            "C": self.C,
            "backend": "pytorch",
            "device": self.device,
            "fit_device": self.fit_device_,
            "optimizer": "lbfgs",
            "max_iter": self.max_iter,
            "n_iter": self.n_iter_,
            "tolerance": self.tolerance,
            "learning_rate": self.learning_rate,
            "class_weight": self.class_weight,
            "random_seed": self.random_seed,
        }
        if self.classes_ is not None:
            metadata["classes"] = self.classes_.tolist()
        return metadata

    def _require_fitted(self) -> Dict[str, np.ndarray]:
        """Return fitted arrays or reject an unfitted estimator.

        Args:
            None: This method inspects estimator state.

        Returns:
            Dict[str, np.ndarray]: Classes, scaling, and linear-model arrays.
        """
        values = {
            "classes": self.classes_,
            "mean": self.mean_,
            "scale": self.scale_,
            "coef": self.coef_,
            "intercept": self.intercept_,
        }
        if any(value is None for value in values.values()):
            raise RuntimeError("TorchLogisticRegression has not been fitted.")
        return {key: np.asarray(value) for key, value in values.items()}


def _as_feature_matrix(values: ArrayLike) -> np.ndarray:
    """Convert bag representations to a finite float matrix.

    Args:
        values (ArrayLike): NumPy or torch bag representations.

    Returns:
        np.ndarray: C-contiguous float32 matrix shaped ``[N, features]``.
    """
    array = (
        values.detach().cpu().numpy()
        if isinstance(values, torch.Tensor)
        else np.asarray(values)
    )
    matrix = np.asarray(array, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("bag_features must be a nonempty 2D matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("bag_features contain non-finite values.")
    return np.ascontiguousarray(matrix)


def _as_label_vector(values: ArrayLike, expected_size: int) -> np.ndarray:
    """Convert labels to a validated one-dimensional integer vector.

    Args:
        values (ArrayLike): NumPy or torch labels.
        expected_size (int): Required number of labels.

    Returns:
        np.ndarray: Integer labels shaped ``[expected_size]``.
    """
    array = (
        values.detach().cpu().numpy()
        if isinstance(values, torch.Tensor)
        else np.asarray(values)
    )
    labels = np.asarray(array)
    if labels.ndim != 1 or labels.shape[0] != expected_size:
        raise ValueError(f"labels must have shape [{expected_size}].")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
            raise ValueError("labels must contain finite integer values.")
    return labels.astype(np.int64, copy=False)
