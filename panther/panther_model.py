"""Faithful PANTHER diagonal-GMM representation and downstream classifier.

The MAP-EM update follows the official CVPR 2024 implementation, which was
itself adapted from Differentiable EM for Set Representation Learning (DIEM).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn


class PANTHER(nn.Module):
    """Summarize each slide by mixture weights, means, and diagonal variances."""

    def __init__(
        self,
        prototypes: torch.Tensor,
        em_iterations: int = 1,
        tau: float = 1.0,
        covariance_regularizer: float = 1.0,
        variance_floor: float = 1e-6,
        output_type: str = "allcat",
        fix_prototypes: bool = True,
        em_chunk_size: Optional[int] = 65536,
    ) -> None:
        super().__init__()
        prototypes = torch.as_tensor(prototypes, dtype=torch.float32)
        if prototypes.ndim != 2 or min(prototypes.shape) <= 0:
            raise ValueError("prototypes must have shape [num_prototypes, feature_dim].")
        if output_type != "allcat":
            raise ValueError("Only the original PANTHER 'allcat' output is supported.")
        self.num_prototypes = int(prototypes.shape[0])
        self.feature_dim = int(prototypes.shape[1])
        self.em_iterations = int(em_iterations)
        self.tau = float(tau)
        self.covariance_regularizer = float(covariance_regularizer)
        self.variance_floor = float(variance_floor)
        self.output_type = output_type
        self.em_chunk_size = em_chunk_size
        self.prototypes = nn.Parameter(prototypes, requires_grad=not fix_prototypes)

    @property
    def output_dim(self) -> int:
        """Dimension of `[pi, flattened mu, flattened diagonal variance]`."""
        return self.num_prototypes + 2 * self.num_prototypes * self.feature_dim

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_assignments: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute PANTHER representations for one or more padded slide bags."""
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3 or int(features.shape[2]) != self.feature_dim:
            raise ValueError(
                f"features must have shape [batch, tiles, {self.feature_dim}]."
            )
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        else:
            mask = mask.to(device=features.device, dtype=torch.bool)
            if tuple(mask.shape) != tuple(features.shape[:2]):
                raise ValueError("mask shape must match features[:2].")

        outputs = [
            self._map_em_single(features[index, mask[index]], return_assignments)
            for index in range(int(features.shape[0]))
        ]
        result = {
            key: torch.stack([output[key] for output in outputs], dim=0)
            for key in ("representation", "mixture_weights", "means", "variances")
        }
        if return_assignments:
            lengths = {int(output["assignments"].shape[0]) for output in outputs}
            if len(lengths) != 1:
                raise ValueError(
                    "Assignments can only be returned for equal-length unpadded bags."
                )
            result["assignments"] = torch.stack(
                [output["assignments"] for output in outputs], dim=0
            )
        return result

    def _map_em_single(
        self, features: torch.Tensor, return_assignments: bool
    ) -> Dict[str, torch.Tensor]:
        if features.shape[0] == 0:
            raise ValueError("PANTHER cannot aggregate an empty slide bag.")
        prototypes = self.prototypes
        dtype = features.dtype
        device = features.device
        k = self.num_prototypes
        pi = torch.full((k,), 1.0 / k, dtype=dtype, device=device)
        mu = prototypes.to(dtype=dtype)
        variance = torch.full_like(mu, self.covariance_regularizer)
        prior_variance = torch.full_like(mu, self.covariance_regularizer)
        assignments = None

        for iteration in range(self.em_iterations):
            weight_sum = torch.zeros(k, dtype=dtype, device=device)
            weighted_sum = torch.zeros_like(mu)
            weighted_square_sum = torch.zeros_like(mu)
            assignment_parts = [] if return_assignments and iteration + 1 == self.em_iterations else None

            chunk_size = self.em_chunk_size or int(features.shape[0])
            for start in range(0, int(features.shape[0]), chunk_size):
                chunk = features[start : start + chunk_size]
                responsibilities = _responsibilities(chunk, pi, mu, variance)
                weight_sum += responsibilities.sum(dim=0)
                weighted_sum += responsibilities.transpose(0, 1) @ chunk
                weighted_square_sum += responsibilities.transpose(0, 1) @ chunk.square()
                if assignment_parts is not None:
                    assignment_parts.append(responsibilities)

            regularized_weight = weight_sum + self.tau
            pi = regularized_weight / regularized_weight.sum()
            mu = (weighted_sum + self.tau * prototypes) / regularized_weight[:, None]
            second_moment = (
                weighted_square_sum
                + self.tau * (prior_variance + prototypes.square())
            ) / regularized_weight[:, None]
            variance = (second_moment - mu.square()).clamp_min(self.variance_floor)
            if assignment_parts is not None:
                assignments = torch.cat(assignment_parts, dim=0)

        representation = torch.cat((pi.flatten(), mu.flatten(), variance.flatten()))
        result = {
            "representation": representation,
            "mixture_weights": pi,
            "means": mu,
            "variances": variance,
        }
        if return_assignments:
            assert assignments is not None
            result["assignments"] = assignments
        return result


def _responsibilities(
    features: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    """Evaluate posterior GMM component probabilities without a 3-D expansion."""
    inverse_variance = variance.reciprocal()
    mahalanobis = (
        features.square() @ inverse_variance.transpose(0, 1)
        + (mu.square() * inverse_variance).sum(dim=1).unsqueeze(0)
        - 2.0 * features @ (mu * inverse_variance).transpose(0, 1)
    )
    log_normalizer = (
        features.shape[1] * math.log(2.0 * math.pi)
        + variance.log().sum(dim=1)
    )
    log_joint = -0.5 * (mahalanobis + log_normalizer.unsqueeze(0))
    log_joint = log_joint + pi.clamp_min(torch.finfo(pi.dtype).tiny).log().unsqueeze(0)
    return torch.softmax(log_joint, dim=1)


class LinearClassifier(nn.Module):
    """Original PANTHER downstream linear head (bias disabled by default)."""

    def __init__(self, input_dim: int, num_classes: int, bias: bool = False) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes, bias=bias)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        return self.classifier(representations)

