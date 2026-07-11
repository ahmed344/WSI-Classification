"""Canonical batched CLAM single-branch and multi-branch models."""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class _AttentionNetwork(nn.Module):
    """Compute canonical gated or ungated CLAM attention scores."""

    def __init__(
        self,
        input_dim: int,
        attention_dim: int,
        branches: int,
        gated: bool,
        dropout: float,
    ) -> None:
        """Initialize the attention network.

        Args:
            input_dim (int): Embedded tile feature dimension.
            attention_dim (int): Hidden attention dimension.
            branches (int): Number of attention branches.
            gated (bool): Whether to use gated attention.
            dropout (float): Dropout probability in attention projections.

        Returns:
            None: The initialized module.
        """
        super().__init__()
        self.gated = gated
        self.attention_v = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        if gated:
            self.attention_u: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(input_dim, attention_dim),
                nn.Sigmoid(),
                nn.Dropout(dropout),
            )
        else:
            self.attention_u = None
        self.attention_out = nn.Linear(attention_dim, branches)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized attention scores.

        Args:
            features (torch.Tensor): Embedded features shaped ``[B, N, H]``.

        Returns:
            torch.Tensor: Raw scores shaped ``[B, branches, N]``.
        """
        attention = self.attention_v(features)
        if self.attention_u is not None:
            attention = attention * self.attention_u(features)
        return self.attention_out(attention).transpose(1, 2)


class _CLAMBase(nn.Module):
    """Shared implementation for canonical CLAM-SB and CLAM-MB."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        attention_dim: int,
        num_classes: int,
        gated: bool,
        dropout: float,
        k_sample: int,
        subtyping: bool,
        attention_branches: int,
    ) -> None:
        """Initialize common CLAM modules.

        Args:
            input_dim (int): Input tile feature dimension.
            hidden_dim (int): Shared embedding dimension.
            attention_dim (int): Attention hidden dimension.
            num_classes (int): Number of bag classes.
            gated (bool): Whether to use gated attention.
            dropout (float): Dropout probability.
            k_sample (int): Maximum positive and negative instances per class.
            subtyping (bool): Whether to supervise out-of-class branches.
            attention_branches (int): One for SB or ``num_classes`` for MB.

        Returns:
            None: The initialized model.
        """
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or attention_dim <= 0:
            raise ValueError("Model dimensions must be positive.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if k_sample <= 0:
            raise ValueError("k_sample must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.k_sample = k_sample
        self.subtyping = subtyping
        self.num_attention_branches = attention_branches

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = _AttentionNetwork(
            input_dim=hidden_dim,
            attention_dim=attention_dim,
            branches=attention_branches,
            gated=gated,
            dropout=dropout,
        )
        self.instance_classifiers = nn.ModuleList(
            nn.Linear(hidden_dim, 2) for _ in range(num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize linear layers with canonical Xavier weights.

        Args:
            None.

        Returns:
            None: Parameters are initialized in place.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_inputs(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Validate inputs and materialize a mask.

        Args:
            features (torch.Tensor): Tile features shaped ``[B, N, D]``.
            mask (Optional[torch.Tensor]): Boolean valid-tile mask shaped ``[B, N]``.
            labels (Optional[torch.Tensor]): Integer class labels shaped ``[B]``.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Validated mask and labels.
        """
        if features.ndim != 3:
            raise ValueError("features must have shape [B, N, D].")
        batch_size, tile_count, feature_dim = features.shape
        if batch_size == 0 or tile_count == 0:
            raise ValueError("features must contain at least one bag and tile.")
        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, got {feature_dim}."
            )

        if mask is None:
            validated_mask = torch.ones(
                (batch_size, tile_count),
                dtype=torch.bool,
                device=features.device,
            )
        else:
            if mask.shape != (batch_size, tile_count):
                raise ValueError("mask must have shape [B, N].")
            if mask.dtype != torch.bool:
                raise TypeError("mask must have boolean dtype.")
            if mask.device != features.device:
                raise ValueError("features and mask must be on the same device.")
            validated_mask = mask

        empty_bags = (~validated_mask.any(dim=1)).nonzero(as_tuple=False).flatten()
        if empty_bags.numel() > 0:
            indices = empty_bags.detach().cpu().tolist()
            raise ValueError(f"All-empty bags are not allowed; bag indices: {indices}.")

        if labels is not None:
            if labels.shape != (batch_size,):
                raise ValueError("labels must have shape [B].")
            if labels.device != features.device:
                raise ValueError("features and labels must be on the same device.")
            if labels.dtype not in (torch.int32, torch.int64):
                raise TypeError("labels must have an integer dtype.")
            labels = labels.to(dtype=torch.long)
            if bool(((labels < 0) | (labels >= self.num_classes)).any()):
                raise ValueError("labels contain a class index outside the valid range.")

        return validated_mask, labels

    def _classify_bags(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """Classify pooled bag representations.

        Args:
            pooled_features (torch.Tensor): Pooled features shaped ``[B, K, H]``.

        Returns:
            torch.Tensor: Bag logits shaped ``[B, C]``.
        """
        raise NotImplementedError

    def _instance_supervision(
        self,
        embedded: torch.Tensor,
        attention_scores: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute label-aware CLAM instance supervision.

        Args:
            embedded (torch.Tensor): Embedded tile features shaped ``[B, N, H]``.
            attention_scores (torch.Tensor): Raw attention shaped ``[B, K, N]``.
            mask (torch.Tensor): Boolean valid-tile mask shaped ``[B, N]``.
            labels (torch.Tensor): Bag labels shaped ``[B]``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Mean CE loss, flattened
            instance predictions, and flattened binary targets.
        """
        logits_parts: List[torch.Tensor] = []
        loss_parts: List[torch.Tensor] = []
        target_parts: List[torch.Tensor] = []

        for bag_index in range(embedded.shape[0]):
            valid_features = embedded[bag_index, mask[bag_index]]
            valid_count = valid_features.shape[0]
            label = int(labels[bag_index].item())

            for class_index, classifier in enumerate(self.instance_classifiers):
                branch_index = (
                    class_index if self.num_attention_branches > 1 else 0
                )
                valid_scores = attention_scores[
                    bag_index, branch_index, mask[bag_index]
                ]
                order = torch.argsort(valid_scores)

                if class_index == label:
                    k = min(self.k_sample, valid_count // 2)
                    if k == 0:
                        continue
                    selected = torch.cat((order[-k:], order[:k]))
                    targets = torch.cat(
                        (
                            torch.ones(k, dtype=torch.long, device=embedded.device),
                            torch.zeros(k, dtype=torch.long, device=embedded.device),
                        )
                    )
                elif self.subtyping:
                    k = min(self.k_sample, valid_count)
                    selected = order[-k:]
                    targets = torch.zeros(
                        k, dtype=torch.long, device=embedded.device
                    )
                else:
                    continue

                instance_logits = classifier(valid_features[selected])
                logits_parts.append(instance_logits)
                loss_parts.append(F.cross_entropy(instance_logits, targets))
                target_parts.append(targets)

        if not logits_parts:
            zero = embedded.sum() * 0.0
            empty = torch.empty(0, dtype=torch.long, device=embedded.device)
            return zero, empty, empty

        instance_logits = torch.cat(logits_parts, dim=0)
        instance_targets = torch.cat(target_parts, dim=0)
        instance_loss = torch.stack(loss_parts).mean()
        instance_predictions = instance_logits.argmax(dim=1)
        return instance_loss, instance_predictions, instance_targets

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        instance_eval: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run batched, masked CLAM inference and optional instance supervision.

        Args:
            features (torch.Tensor): Tile features shaped ``[B, N, D]``.
            mask (Optional[torch.Tensor]): Boolean valid-tile mask shaped ``[B, N]``.
            labels (Optional[torch.Tensor]): Integer bag labels shaped ``[B]``.
            instance_eval (bool): Whether to compute instance supervision when
                labels are supplied.

        Returns:
            Dict[str, torch.Tensor]: Stable output dictionary containing bag
            predictions, attention tensors, pooled features, and instance results.
        """
        mask, labels = self._validate_inputs(features, mask, labels)
        embedded = self.embedding(features)
        attention_scores = self.attention(embedded)
        attention_scores = attention_scores.masked_fill(
            ~mask.unsqueeze(1), float("-inf")
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.masked_fill(~mask.unsqueeze(1), 0.0)
        pooled_features = torch.bmm(attention_weights, embedded)
        logits = self._classify_bags(pooled_features)
        probabilities = F.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        if labels is not None and instance_eval:
            instance_loss, instance_predictions, instance_targets = (
                self._instance_supervision(
                    embedded, attention_scores, mask, labels
                )
            )
        else:
            instance_loss = embedded.sum() * 0.0
            instance_predictions = torch.empty(
                0, dtype=torch.long, device=features.device
            )
            instance_targets = torch.empty(
                0, dtype=torch.long, device=features.device
            )

        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "attention_scores": attention_scores,
            "attention_weights": attention_weights,
            "pooled_features": pooled_features,
            "instance_loss": instance_loss,
            "instance_predictions": instance_predictions,
            "instance_targets": instance_targets,
        }


class CLAM_SB(_CLAMBase):
    """Canonical CLAM single-attention-branch model."""

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        num_classes: int = 2,
        gated: bool = True,
        dropout: float = 0.25,
        k_sample: int = 8,
        subtyping: bool = False,
    ) -> None:
        """Initialize CLAM-SB.

        Args:
            input_dim (int): Input tile feature dimension.
            hidden_dim (int): Shared embedding dimension.
            attention_dim (int): Attention hidden dimension.
            num_classes (int): Number of bag classes.
            gated (bool): Whether to use gated attention.
            dropout (float): Dropout probability.
            k_sample (int): Maximum positive and negative instances per class.
            subtyping (bool): Whether to supervise out-of-class classifiers.

        Returns:
            None: The initialized CLAM-SB model.
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            num_classes=num_classes,
            gated=gated,
            dropout=dropout,
            k_sample=k_sample,
            subtyping=subtyping,
            attention_branches=1,
        )
        self.classifiers = nn.ModuleList(
            nn.Linear(hidden_dim, 1) for _ in range(num_classes)
        )
        self._initialize_weights()

    def _classify_bags(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """Classify one shared pooled representation.

        Args:
            pooled_features (torch.Tensor): Features shaped ``[B, 1, H]``.

        Returns:
            torch.Tensor: Bag logits shaped ``[B, C]``.
        """
        shared_bag = pooled_features[:, 0]
        return torch.cat(
            [classifier(shared_bag) for classifier in self.classifiers], dim=1
        )


class CLAM_MB(_CLAMBase):
    """Canonical CLAM multi-attention-branch model."""

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        num_classes: int = 2,
        gated: bool = True,
        dropout: float = 0.25,
        k_sample: int = 8,
        subtyping: bool = False,
    ) -> None:
        """Initialize CLAM-MB.

        Args:
            input_dim (int): Input tile feature dimension.
            hidden_dim (int): Shared embedding dimension.
            attention_dim (int): Attention hidden dimension.
            num_classes (int): Number of bag classes and attention branches.
            gated (bool): Whether to use gated attention.
            dropout (float): Dropout probability.
            k_sample (int): Maximum positive and negative instances per class.
            subtyping (bool): Whether to supervise out-of-class branches.

        Returns:
            None: The initialized CLAM-MB model.
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            num_classes=num_classes,
            gated=gated,
            dropout=dropout,
            k_sample=k_sample,
            subtyping=subtyping,
            attention_branches=num_classes,
        )
        self.classifiers = nn.ModuleList(
            nn.Linear(hidden_dim, 1) for _ in range(num_classes)
        )
        self._initialize_weights()

    def _classify_bags(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """Classify each class-specific pooled representation directly.

        Args:
            pooled_features (torch.Tensor): Features shaped ``[B, C, H]``.

        Returns:
            torch.Tensor: Direct class logits shaped ``[B, C]``.
        """
        return torch.cat(
            [
                classifier(pooled_features[:, class_index])
                for class_index, classifier in enumerate(self.classifiers)
            ],
            dim=1,
        )
