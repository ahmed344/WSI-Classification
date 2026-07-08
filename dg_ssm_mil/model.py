"""
DG-SSM-MIL model components for tissue-level WSI classification.
"""
from math import sqrt
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, knn_graph

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
    raise ImportError(
        "DGSSMMILModel requires mamba-ssm. Install mamba-ssm before importing model.py."
    ) from exc


class SpatialGATEncoder(nn.Module):
    """
    Encode local spatial neighborhoods with a PyG graph attention layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        spatial_knn_k: int = 8,
        gat_heads: int = 4,
        dropout: float = 0.15,
    ) -> None:
        """
        Initialize the spatial graph attention encoder.

        Args:
            hidden_dim (int): Node feature dimension.
            spatial_knn_k (int): Number of coordinate nearest neighbors.
            gat_heads (int): Number of attention heads in GATv2Conv.
            dropout (float): Dropout probability for graph attention.

        Returns:
            None: This constructor initializes module layers in-place.
        """
        super().__init__()
        self.spatial_knn_k = spatial_knn_k
        self.gat = GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=gat_heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply spatial graph attention to each tissue in a padded batch.

        Args:
            features (torch.Tensor): Padded feature tensor `[B, N, D]`.
            coords (torch.Tensor): Padded coordinate tensor `[B, N, 2]`.
            mask (torch.Tensor): Boolean valid-tile mask `[B, N]`.

        Returns:
            torch.Tensor: Graph-enhanced features with shape `[B, N, D]`.
        """
        batch_outputs = torch.zeros_like(features)
        for batch_idx in range(features.shape[0]):
            valid_count = int(mask[batch_idx].sum().item())
            if valid_count <= 0:
                continue
            node_features = features[batch_idx, :valid_count]
            if valid_count == 1:
                batch_outputs[batch_idx, :valid_count] = node_features
                continue
            node_coords = coords[batch_idx, :valid_count]
            k_neighbors = min(self.spatial_knn_k, valid_count - 1)
            edge_index = knn_graph(
                node_coords,
                k=k_neighbors,
                loop=False,
                flow="source_to_target",
            )
            graph_features = self.gat(node_features, edge_index)
            graph_features = self.norm(node_features + self.dropout(graph_features))
            batch_outputs[batch_idx, :valid_count] = graph_features
        return batch_outputs


class DynamicGraphFusion(nn.Module):
    """
    Fuse projected features with graph features using dynamic top-k neighbors.
    """

    def __init__(
        self,
        hidden_dim: int,
        top_k: int = 6,
        chunk_size: int = 512,
        dropout: float = 0.15,
    ) -> None:
        """
        Initialize the dynamic graph fusion module.

        Args:
            hidden_dim (int): Feature dimension for projected and graph streams.
            top_k (int): Number of dynamic cross-sequence neighbors per tile.
            chunk_size (int): Number of query tiles processed per dynamic graph
                chunk to avoid materializing an `N x N` attention matrix.
            dropout (float): Dropout probability in the fusion MLP.

        Returns:
            None: This constructor initializes module layers in-place.
        """
        super().__init__()
        self.top_k = top_k
        self.chunk_size = chunk_size
        edge_dim = hidden_dim * 3
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        projected_features: torch.Tensor,
        graph_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse streams through top-k cross attention and gated residual updates.

        Args:
            projected_features (torch.Tensor): Projected features `[B, N, D]`.
            graph_features (torch.Tensor): Spatial graph features `[B, N, D]`.
            mask (torch.Tensor): Boolean valid-tile mask `[B, N]`.

        Returns:
            torch.Tensor: Fused features of shape `[B, N, D]`.
        """
        hidden_dim = projected_features.shape[-1]
        aggregated = torch.zeros_like(projected_features)
        for batch_idx in range(projected_features.shape[0]):
            valid_count = int(mask[batch_idx].sum().item())
            if valid_count <= 0:
                continue
            projected_valid = projected_features[batch_idx, :valid_count]
            graph_valid = graph_features[batch_idx, :valid_count]
            top_k = min(self.top_k, valid_count)
            for start_idx in range(0, valid_count, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, valid_count)
                query_chunk = projected_valid[start_idx:end_idx]
                scores = torch.matmul(
                    query_chunk,
                    graph_valid.transpose(0, 1),
                ) / sqrt(hidden_dim)
                top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
                top_weights = F.softmax(top_scores, dim=-1)
                neighbor_features = graph_valid[top_indices]
                query_features = query_chunk.unsqueeze(1).expand_as(neighbor_features)
                edge_inputs = torch.cat(
                    [
                        query_features,
                        neighbor_features,
                        query_features * neighbor_features,
                    ],
                    dim=-1,
                )
                edge_messages = self.edge_mlp(edge_inputs)
                aggregated[batch_idx, start_idx:end_idx] = torch.sum(
                    edge_messages * top_weights.unsqueeze(-1),
                    dim=1,
                )
        gate_input = torch.cat(
            [projected_features, aggregated, projected_features * aggregated],
            dim=-1,
        )
        update_gate = self.gate(gate_input)
        fused = projected_features + self.dropout(update_gate * aggregated)
        fused = self.output_norm(fused)
        return fused.masked_fill(~mask.unsqueeze(-1), 0.0)


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional vision-style Mamba block with an optional local Conv1d branch.
    """

    def __init__(
        self,
        hidden_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_conv_branch: bool = True,
        conv_kernel_size: int = 3,
        dropout: float = 0.15,
    ) -> None:
        """
        Initialize the bidirectional Mamba block.

        Args:
            hidden_dim (int): Input and output feature dimension.
            d_state (int): Mamba state dimension.
            d_conv (int): Mamba internal convolution width.
            expand (int): Mamba expansion factor.
            use_conv_branch (bool): Whether to include a non-causal Conv1d branch.
            conv_kernel_size (int): Kernel size for the local Conv1d branch.
            dropout (float): Dropout probability after stream fusion.

        Returns:
            None: This constructor initializes module layers in-place.
        """
        super().__init__()
        self.use_conv_branch = use_conv_branch
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.forward_mamba = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_mamba = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        if use_conv_branch:
            padding = conv_kernel_size // 2
            self.conv_branch = nn.Sequential(
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=conv_kernel_size,
                    padding=padding,
                    groups=1,
                ),
                nn.SiLU(),
            )
        else:
            self.conv_branch = None
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply bidirectional Mamba processing to a padded tile sequence.

        Args:
            features (torch.Tensor): Padded feature tensor `[B, N, D]`.
            mask (torch.Tensor): Boolean valid-tile mask `[B, N]`.

        Returns:
            torch.Tensor: Sequence features after bidirectional SSM processing.
        """
        normalized = self.input_norm(features).masked_fill(~mask.unsqueeze(-1), 0.0)
        forward_out = self.forward_mamba(normalized)
        reversed_input = _reverse_valid_prefix(normalized, mask)
        backward_reversed = self.backward_mamba(reversed_input)
        backward_out = _reverse_valid_prefix(backward_reversed, mask)

        combined = forward_out + backward_out
        if self.conv_branch is not None:
            conv_input = normalized.transpose(1, 2)
            conv_out = self.conv_branch(conv_input).transpose(1, 2)
            combined = combined + conv_out

        combined = self.fusion(combined)
        output = self.output_norm(features + self.dropout(combined))
        return output.masked_fill(~mask.unsqueeze(-1), 0.0)


class ABMILPooling(nn.Module):
    """
    Attention-based MIL pooling and classification head.
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        attention_type: str = "standard",
    ) -> None:
        """
        Initialize ABMIL pooling.

        Args:
            hidden_dim (int): Tile feature dimension.
            attention_hidden_dim (int): Hidden dimension for attention scoring.
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability before classification.
            attention_type (str): Attention scorer type, either `standard` or `gated`.

        Returns:
            None: This constructor initializes module layers in-place.
        """
        super().__init__()
        if attention_type not in {"standard", "gated"}:
            raise ValueError("attention_type must be one of: standard, gated.")
        self.attention_type = attention_type
        if attention_type == "gated":
            self.attention_v = nn.Sequential(
                nn.Linear(hidden_dim, attention_hidden_dim),
                nn.Tanh(),
            )
            self.attention_u = nn.Sequential(
                nn.Linear(hidden_dim, attention_hidden_dim),
                nn.Sigmoid(),
            )
            self.attention = nn.Linear(attention_hidden_dim, 1)
        else:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, attention_hidden_dim),
                nn.Tanh(),
                nn.Linear(attention_hidden_dim, 1),
            )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pool tile features and classify the tissue bag.

        Args:
            features (torch.Tensor): Tile features of shape `[B, N, D]`.
            mask (torch.Tensor): Boolean valid-tile mask `[B, N]`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Logits `[B, C]`,
            attention weights `[B, N]`, and pooled embeddings `[B, D]`.
        """
        if self.attention_type == "gated":
            scores = self.attention(
                self.attention_v(features) * self.attention_u(features)
            ).squeeze(-1)
        else:
            scores = self.attention(features).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention_weights = F.softmax(scores, dim=1)
        attention_weights = attention_weights.masked_fill(~mask, 0.0)
        pooled = torch.bmm(attention_weights.unsqueeze(1), features).squeeze(1)
        logits = self.classifier(pooled)
        return logits, attention_weights, pooled


class DGSSMMILModel(nn.Module):
    """
    Dynamic Graph and State Space MIL model for tissue-level classification.
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        num_classes: int = 5,
        projection_dropout: float = 0.15,
        spatial_knn_k: int = 8,
        gat_heads: int = 4,
        gat_dropout: float = 0.15,
        dynamic_graph_top_k: int = 6,
        dynamic_graph_chunk_size: int = 512,
        dynamic_graph_dropout: float = 0.15,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_conv_branch: bool = True,
        conv_kernel_size: int = 3,
        block_dropout: float = 0.15,
        attention_hidden_dim: int = 256,
        classifier_dropout: float = 0.3,
        attention_type: str = "standard",
    ) -> None:
        """
        Initialize the DG-SSM-MIL model.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Shared hidden dimension.
            num_classes (int): Number of tissue classes.
            projection_dropout (float): Dropout in input projection.
            spatial_knn_k (int): Spatial k-NN value for graph construction.
            gat_heads (int): Number of PyG GAT attention heads.
            gat_dropout (float): Dropout in the GAT layer.
            dynamic_graph_top_k (int): Top-k value for dynamic graph fusion.
            dynamic_graph_chunk_size (int): Query chunk size for memory-efficient
                dynamic graph top-k attention.
            dynamic_graph_dropout (float): Dropout in dynamic graph fusion.
            mamba_d_state (int): Mamba state dimension.
            mamba_d_conv (int): Mamba internal convolution width.
            mamba_expand (int): Mamba expansion factor.
            use_conv_branch (bool): Whether to include a Conv1d branch in Bi-SSM.
            conv_kernel_size (int): Conv1d branch kernel size.
            block_dropout (float): Dropout after SSM branch fusion.
            attention_hidden_dim (int): Hidden dimension for ABMIL attention.
            classifier_dropout (float): Dropout before final classifier.
            attention_type (str): Bag-level MIL attention scorer type.

        Returns:
            None: This constructor initializes module layers in-place.
        """
        super().__init__()
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(projection_dropout),
        )
        self.spatial_encoder = SpatialGATEncoder(
            hidden_dim=hidden_dim,
            spatial_knn_k=spatial_knn_k,
            gat_heads=gat_heads,
            dropout=gat_dropout,
        )
        self.dynamic_fusion = DynamicGraphFusion(
            hidden_dim=hidden_dim,
            top_k=dynamic_graph_top_k,
            chunk_size=dynamic_graph_chunk_size,
            dropout=dynamic_graph_dropout,
        )
        self.sequence_block = BidirectionalMambaBlock(
            hidden_dim=hidden_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            use_conv_branch=use_conv_branch,
            conv_kernel_size=conv_kernel_size,
            dropout=block_dropout,
        )
        self.pooling = ABMILPooling(
            hidden_dim=hidden_dim,
            attention_hidden_dim=attention_hidden_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
            attention_type=attention_type,
        )

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run DG-SSM-MIL inference on a padded tissue batch.

        Args:
            features (torch.Tensor): Padded input features `[B, N, input_dim]`.
            coords (torch.Tensor): Padded coordinates `[B, N, 2]`.
            mask (Optional[torch.Tensor]): Boolean valid-tile mask `[B, N]`.

        Returns:
            Dict[str, torch.Tensor]: Outputs containing `logits`,
            `attention_weights`, `pooled_features`, and intermediate tensors.
        """
        if mask is None:
            mask = torch.ones(
                features.shape[:2],
                dtype=torch.bool,
                device=features.device,
            )
        projected = self.feature_projection(features)
        projected = projected.masked_fill(~mask.unsqueeze(-1), 0.0)
        graph_features = self.spatial_encoder(projected, coords, mask)
        fused_features = self.dynamic_fusion(projected, graph_features, mask)
        sequence_features = self.sequence_block(fused_features, mask)
        logits, attention_weights, pooled_features = self.pooling(sequence_features, mask)
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "pooled_features": pooled_features,
            "projected_features": projected,
            "graph_features": graph_features,
            "fused_features": fused_features,
            "sequence_features": sequence_features,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DGSSMMILModel":
        """
        Build a DGSSMMILModel from a configuration dictionary.

        Args:
            config (Dict[str, Any]): Parsed DG-SSM-MIL configuration.

        Returns:
            DGSSMMILModel: Model initialized from config values.
        """
        return cls(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            num_classes=int(config["num_classes"]),
            projection_dropout=float(config.get("projection_dropout", 0.15)),
            spatial_knn_k=int(config.get("spatial_knn_k", 8)),
            gat_heads=int(config.get("gat_heads", 4)),
            gat_dropout=float(config.get("gat_dropout", 0.15)),
            dynamic_graph_top_k=int(config.get("dynamic_graph_top_k", 6)),
            dynamic_graph_chunk_size=int(config.get("dynamic_graph_chunk_size", 512)),
            dynamic_graph_dropout=float(config.get("dynamic_graph_dropout", 0.15)),
            mamba_d_state=int(config.get("mamba_d_state", 16)),
            mamba_d_conv=int(config.get("mamba_d_conv", 4)),
            mamba_expand=int(config.get("mamba_expand", 2)),
            use_conv_branch=bool(config.get("use_conv_branch", True)),
            conv_kernel_size=int(config.get("conv_kernel_size", 3)),
            block_dropout=float(config.get("block_dropout", 0.15)),
            attention_hidden_dim=int(config.get("attention_hidden_dim", 256)),
            classifier_dropout=float(config.get("classifier_dropout", 0.3)),
            attention_type=str(config.get("attention_type", "standard")),
        )


def _batched_gather_nodes(
    nodes: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Gather node features for batched top-k indices.

    Args:
        nodes (torch.Tensor): Node tensor of shape `[B, N, D]`.
        indices (torch.Tensor): Index tensor of shape `[B, N, K]`.

    Returns:
        torch.Tensor: Gathered tensor of shape `[B, N, K, D]`.
    """
    feature_dim = nodes.shape[-1]
    expanded_nodes = nodes.unsqueeze(1).expand(-1, indices.shape[1], -1, -1)
    expanded_indices = indices.unsqueeze(-1).expand(-1, -1, -1, feature_dim)
    return torch.gather(expanded_nodes, dim=2, index=expanded_indices)


def _reverse_valid_prefix(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Reverse only valid tiles in each padded sequence.

    Args:
        features (torch.Tensor): Padded feature tensor `[B, N, D]`.
        mask (torch.Tensor): Boolean valid-tile mask `[B, N]`.

    Returns:
        torch.Tensor: Tensor where each valid prefix has been reversed.
    """
    reversed_features = torch.zeros_like(features)
    for batch_idx in range(features.shape[0]):
        valid_count = int(mask[batch_idx].sum().item())
        if valid_count <= 0:
            continue
        reversed_features[batch_idx, :valid_count] = torch.flip(
            features[batch_idx, :valid_count],
            dims=[0],
        )
    return reversed_features
