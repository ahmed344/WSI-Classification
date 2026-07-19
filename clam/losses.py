"""Loss functions used by CLAM training."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class GeneralizedCrossEntropyLoss(nn.Module):
    """Generalized cross entropy with optional label smoothing.

    ``q=0`` uses the continuous cross-entropy limit, while positive ``q`` uses
    the noise-robust generalized cross-entropy formulation.
    """

    def __init__(
        self,
        q: float = 0.0,
        epsilon: float = 0.0,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize the classification loss.

        Args:
            q (float): Generalized cross-entropy exponent in ``[0, 1]``.
            epsilon (float): Label-smoothing strength in ``[0, 1)``.
            weight (Optional[torch.Tensor]): Optional nonnegative class weights
                shaped ``[C]``.

        Returns:
            None: The loss module is initialized in place.
        """
        super().__init__()
        if not 0.0 <= float(q) <= 1.0:
            raise ValueError("q must be between 0 and 1.")
        if not 0.0 <= float(epsilon) < 1.0:
            raise ValueError("epsilon must be in [0, 1).")
        if weight is not None:
            if weight.ndim != 1:
                raise ValueError("weight must be a one-dimensional tensor.")
            if not torch.isfinite(weight).all() or (weight < 0).any():
                raise ValueError("weight must contain finite nonnegative values.")

        self.q = float(q)
        self.epsilon = float(epsilon)
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the mean generalized cross-entropy loss.

        Args:
            logits (torch.Tensor): Unnormalized predictions shaped ``[B, C]``.
            targets (torch.Tensor): Integer class indices shaped ``[B]``.

        Returns:
            torch.Tensor: Scalar mean loss.
        """
        if logits.ndim != 2:
            raise ValueError("logits must have shape [B, C].")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets must have shape [B] matching logits.")
        if logits.shape[1] < 2:
            raise ValueError("logits must contain at least two classes.")
        if self.weight is not None and self.weight.numel() != logits.shape[1]:
            raise ValueError("weight length must equal the number of classes.")

        log_probabilities = F.log_softmax(logits, dim=1)
        target_distribution = F.one_hot(
            targets, num_classes=logits.shape[1]
        ).to(dtype=logits.dtype)
        if self.epsilon > 0.0:
            target_distribution = (
                (1.0 - self.epsilon) * target_distribution
                + self.epsilon / logits.shape[1]
            )

        if self.q == 0.0:
            sample_losses = -(target_distribution * log_probabilities).sum(dim=1)
        else:
            probabilities = log_probabilities.exp()
            generalized_losses = (1.0 - probabilities.pow(self.q)) / self.q
            sample_losses = (target_distribution * generalized_losses).sum(dim=1)

        if self.weight is None:
            return sample_losses.mean()

        sample_weights = self.weight[targets]
        denominator = sample_weights.sum()
        if denominator <= 0:
            raise ValueError("The selected targets have zero total class weight.")
        return (sample_losses * sample_weights).sum() / denominator
