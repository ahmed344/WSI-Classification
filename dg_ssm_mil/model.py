"""
DG-SSM-MIL model components for tissue-level WSI classification.
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import GATConv, knn_graph

try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
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
        self.gat = GATConv(
            hidden_dim,
            hidden_dim,
            heads=gat_heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        tissue_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply spatial graph attention to each tissue in a padded batch.

        Args:
            features (torch.Tensor): Padded feature tensor `[B, N, D]`.
            coords (torch.Tensor): Padded coordinate tensor `[B, N, 2]`.
            mask (torch.Tensor): Boolean valid-tile mask `[B, N]`.
            tissue_indices (Optional[torch.Tensor]): Tissue membership per tile
                `[B, N]`. When supplied, k-NN edges never cross tissue boundaries.

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
            edge_parts = []
            if tissue_indices is None:
                groups = torch.zeros(valid_count, dtype=torch.long, device=features.device)
            else:
                groups = tissue_indices[batch_idx, :valid_count]
            for tissue_id in torch.unique(groups):
                group_nodes = torch.nonzero(groups == tissue_id, as_tuple=False).flatten()
                if group_nodes.numel() <= 1:
                    continue
                local_edges = knn_graph(
                    node_coords[group_nodes],
                    k=min(self.spatial_knn_k, int(group_nodes.numel()) - 1),
                    loop=False,
                    flow="source_to_target",
                )
                edge_parts.append(group_nodes[local_edges])
            if edge_parts:
                edge_index = torch.cat(edge_parts, dim=1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=features.device)
            graph_features = self.gat(node_features, edge_index)
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
        lambda_weight: float = 0.5,
        activation: str = "silu",
        dropout: float = 0.15,
    ) -> None:
        """
        Initialize the dynamic graph fusion module.

        Args:
            hidden_dim (int): Feature dimension for projected and graph streams.
            top_k (int): Number of dynamic cross-sequence neighbors per tile.
            chunk_size (int): Number of query tiles processed per dynamic graph
                chunk to avoid materializing an `N x N` attention matrix.
            lambda_weight (float): Blend coefficient lambda from Equation (10).
            activation (str): Activation used for alpha and beta in Equation (12).
            dropout (float): Dropout probability in the fusion MLP.

        Returns:
            None: This constructor initializes module layers in-place.
        """
        super().__init__()
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.lambda_weight = lambda_weight
        if not 0.0 <= lambda_weight <= 1.0:
            raise ValueError("lambda_weight must be in [0, 1].")
        activations = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}
        if activation not in activations:
            raise ValueError(f"Unsupported dynamic graph activation: {activation}.")
        self.edge_score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.additive_transform = nn.Linear(hidden_dim, hidden_dim)
        self.multiplicative_transform = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = activations[activation]()
        self.beta = activations[activation]()

    def forward(
        self,
        projected_features: torch.Tensor,
        graph_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse streams using dynamic graph Equations (6) through (12).

        Args:
            projected_features (torch.Tensor): Projected features `[B, N, D]`.
            graph_features (torch.Tensor): Spatial graph features `[B, N, D]`.
            mask (torch.Tensor): Boolean valid-tile mask `[B, N]`.

        Returns:
            torch.Tensor: Fused features of shape `[B, N, D]`.
        """
        batch_outputs = []
        for batch_idx in range(projected_features.shape[0]):
            valid_count = int(mask[batch_idx].sum().item())
            if valid_count <= 0:
                batch_outputs.append(torch.zeros_like(projected_features[batch_idx]))
                continue
            projected_valid = projected_features[batch_idx, :valid_count]
            graph_valid = graph_features[batch_idx, :valid_count]
            top_k = min(self.top_k, max(1, valid_count - 1))
            aggregated_chunks = []
            for start_idx in range(0, valid_count, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, valid_count)
                query_chunk = projected_valid[start_idx:end_idx]
                similarities = torch.matmul(query_chunk, graph_valid.transpose(0, 1))
                omega = F.softmax(similarities, dim=-1)
                if valid_count > 1:
                    row_indices = torch.arange(
                        start_idx, end_idx, device=projected_features.device
                    )
                    diagonal_mask = torch.zeros_like(omega, dtype=torch.bool)
                    diagonal_mask[
                        torch.arange(end_idx - start_idx, device=projected_features.device),
                        row_indices,
                    ] = True
                    omega = omega.masked_fill(diagonal_mask, -1.0)
                top_omega, top_indices = torch.topk(omega, k=top_k, dim=-1)
                neighbor_features = graph_valid[top_indices]
                query_features = query_chunk.unsqueeze(1).expand_as(neighbor_features)
                edge_features = (
                    top_omega.unsqueeze(-1) * neighbor_features
                    + (1.0 - top_omega.unsqueeze(-1)) * query_features
                )
                analytic_score = torch.sum(
                    neighbor_features * torch.tanh(query_features + edge_features),
                    dim=-1,
                )
                learned_score = self.edge_score_mlp(
                    torch.cat(
                        [query_features, edge_features, neighbor_features],
                        dim=-1,
                    )
                ).squeeze(-1)
                epsilon = (
                    (1.0 - self.lambda_weight) * analytic_score
                    + self.lambda_weight * learned_score
                )
                theta = F.softmax(epsilon, dim=-1)
                aggregated_chunks.append(
                    torch.sum(neighbor_features * theta.unsqueeze(-1), dim=1)
                )
            aggregated = torch.cat(aggregated_chunks, dim=0)
            valid_output = self.alpha(
                self.additive_transform(projected_valid + aggregated)
            ) + self.beta(
                self.multiplicative_transform(
                    projected_valid * aggregated
                )
            )
            batch_outputs.append(
                F.pad(
                    valid_output,
                    (0, 0, 0, projected_features.shape[1] - valid_count),
                )
            )
        output = torch.stack(batch_outputs, dim=0)
        return output.masked_fill(~mask.unsqueeze(-1), 0.0)


class NonCausalMamba(Mamba):
    """
    Mamba selective SSM with a same-length non-causal depthwise convolution.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Run a Mamba branch with symmetric convolution.

        Args:
            hidden_states (torch.Tensor): Input sequence `[B, N, D]`.

        Returns:
            torch.Tensor: Selective-SSM output `[B, N, D]`.
        """
        _, sequence_length, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        left_padding = (self.d_conv - 1) // 2
        right_padding = self.d_conv - 1 - left_padding
        x = F.conv1d(
            F.pad(x, (left_padding, right_padding)),
            self.conv1d.weight,
            self.conv1d.bias,
            groups=self.d_inner,
        )
        x = self.act(x)
        projected = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        delta_rank, b_state, c_state = torch.split(
            projected, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = self.dt_proj.weight @ delta_rank.transpose(0, 1)
        delta = rearrange(delta, "d (b l) -> b d l", l=sequence_length)
        b_state = rearrange(
            b_state, "(b l) n -> b n l", l=sequence_length
        ).contiguous()
        c_state = rearrange(
            c_state, "(b l) n -> b n l", l=sequence_length
        ).contiguous()
        state_matrix = -torch.exp(self.A_log.float())
        if x.is_cuda:
            y = selective_scan_fn(
                x,
                delta,
                state_matrix,
                b_state,
                c_state,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            y = _selective_scan_reference(
                x,
                delta,
                state_matrix,
                b_state,
                c_state,
                self.D.float(),
                z,
                self.dt_proj.bias.float(),
            )
        return self.out_proj(y.transpose(1, 2))


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
        use_residual: bool = False,
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
            use_residual (bool): Whether to add a project-specific outer residual.

        Returns:
            None: This constructor initializes module layers in-place.
        """
        super().__init__()
        self.use_conv_branch = use_conv_branch
        self.use_residual = use_residual
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)
        self.forward_mamba = NonCausalMamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_mamba = NonCausalMamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        if use_conv_branch:
            self.conv_branch = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=conv_kernel_size,
                padding="same",
                groups=hidden_dim,
            )
        else:
            self.conv_branch = None
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
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
        normalized = self.input_norm(features)
        projected = self.input_projection(normalized).masked_fill(~mask.unsqueeze(-1), 0.0)
        forward_out = self.forward_mamba(projected)
        reversed_input = _reverse_valid_prefix(projected, mask)
        backward_reversed = self.backward_mamba(reversed_input)
        backward_out = _reverse_valid_prefix(backward_reversed, mask)

        combined = forward_out + backward_out
        if self.conv_branch is not None:
            conv_out = F.silu(self.conv_branch(projected.transpose(1, 2))).transpose(1, 2)
            combined = combined + conv_out

        output = self.dropout(self.fusion(combined))
        if self.use_residual:
            output = features + output
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
        dynamic_graph_lambda: float = 0.5,
        dynamic_graph_activation: str = "silu",
        dynamic_graph_dropout: float = 0.15,
        use_mamba_block: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_conv_branch: bool = True,
        conv_kernel_size: int = 3,
        block_dropout: float = 0.15,
        use_ssm_residual: bool = False,
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
            dynamic_graph_lambda (float): Lambda blend from dynamic graph Equation (10).
            dynamic_graph_activation (str): Alpha/beta activation from Equation (12).
            dynamic_graph_dropout (float): Dropout in dynamic graph fusion.
            use_mamba_block (bool): Whether to process fused features with the
                bidirectional Mamba block before MIL pooling.
            mamba_d_state (int): Mamba state dimension.
            mamba_d_conv (int): Mamba internal convolution width.
            mamba_expand (int): Mamba expansion factor.
            use_conv_branch (bool): Whether to include a Conv1d branch in Bi-SSM.
            conv_kernel_size (int): Conv1d branch kernel size.
            block_dropout (float): Dropout after SSM branch fusion.
            use_ssm_residual (bool): Enable a non-paper outer SSM residual.
            attention_hidden_dim (int): Hidden dimension for ABMIL attention.
            classifier_dropout (float): Dropout before final classifier.
            attention_type (str): Bag-level MIL attention scorer type.

        Returns:
            None: This constructor initializes module layers in-place.
        """
        super().__init__()
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
            lambda_weight=dynamic_graph_lambda,
            activation=dynamic_graph_activation,
            dropout=dynamic_graph_dropout,
        )
        self.use_mamba_block = use_mamba_block
        if use_mamba_block:
            self.sequence_block: Optional[BidirectionalMambaBlock] = (
                BidirectionalMambaBlock(
                    hidden_dim=hidden_dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                    use_conv_branch=use_conv_branch,
                    conv_kernel_size=conv_kernel_size,
                    dropout=block_dropout,
                    use_residual=use_ssm_residual,
                )
            )
        else:
            self.sequence_block = None
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
        tissue_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run DG-SSM-MIL inference on a padded tissue batch.

        Args:
            features (torch.Tensor): Padded input features `[B, N, input_dim]`.
            coords (torch.Tensor): Padded coordinates `[B, N, 2]`.
            mask (Optional[torch.Tensor]): Boolean valid-tile mask `[B, N]`.
            tissue_indices (Optional[torch.Tensor]): Optional tissue membership
                indices `[B, N]` used to prevent cross-tissue spatial edges.

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
        _validate_model_inputs(features, coords, mask, tissue_indices)
        projected = self.feature_projection(features)
        projected = projected.masked_fill(~mask.unsqueeze(-1), 0.0)
        graph_features = self.spatial_encoder(
            projected, coords, mask, tissue_indices=tissue_indices
        )
        fused_features = self.dynamic_fusion(projected, graph_features, mask)
        if self.sequence_block is None:
            sequence_features = fused_features
        else:
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
            dynamic_graph_lambda=float(config.get("dynamic_graph_lambda", 0.5)),
            dynamic_graph_activation=str(
                config.get("dynamic_graph_activation", "silu")
            ),
            dynamic_graph_dropout=float(config.get("dynamic_graph_dropout", 0.15)),
            use_mamba_block=bool(config.get("use_mamba_block", True)),
            mamba_d_state=int(config.get("mamba_d_state", 16)),
            mamba_d_conv=int(config.get("mamba_d_conv", 4)),
            mamba_expand=int(config.get("mamba_expand", 2)),
            use_conv_branch=bool(config.get("use_conv_branch", True)),
            conv_kernel_size=int(config.get("conv_kernel_size", 3)),
            block_dropout=float(config.get("block_dropout", 0.15)),
            use_ssm_residual=bool(config.get("use_ssm_residual", False)),
            attention_hidden_dim=int(config.get("attention_hidden_dim", 256)),
            classifier_dropout=float(config.get("classifier_dropout", 0.3)),
            attention_type=str(config.get("attention_type", "standard")),
        )


def _validate_model_inputs(
    features: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    tissue_indices: Optional[torch.Tensor],
) -> None:
    """
    Validate padded DG-SSM-MIL model inputs.

    Args:
        features (torch.Tensor): Feature tensor `[B, N, D]`.
        coords (torch.Tensor): Coordinate tensor `[B, N, 2]`.
        mask (torch.Tensor): Boolean validity mask `[B, N]`.
        tissue_indices (Optional[torch.Tensor]): Optional tissue IDs `[B, N]`.

    Returns:
        None: Raises a descriptive exception for invalid inputs.
    """
    if features.ndim != 3:
        raise ValueError("features must have shape [B, N, D].")
    if coords.shape != (*features.shape[:2], 2):
        raise ValueError("coords must have shape [B, N, 2].")
    if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with shape [B, N].")
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("All-empty bags are not supported.")
    if tissue_indices is not None and tissue_indices.shape != mask.shape:
        raise ValueError("tissue_indices must have shape [B, N].")
    if not torch.isfinite(features).all():
        raise ValueError("features contain non-finite values.")
    if not torch.isfinite(coords.masked_select(mask.unsqueeze(-1))).all():
        raise ValueError("valid coordinates contain non-finite values.")


def _selective_scan_reference(
    x: torch.Tensor,
    delta: torch.Tensor,
    state_matrix: torch.Tensor,
    b_state: torch.Tensor,
    c_state: torch.Tensor,
    skip: torch.Tensor,
    gate: torch.Tensor,
    delta_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Run a differentiable PyTorch selective scan for CPU portability.

    Args:
        x (torch.Tensor): SSM input `[B, D, N]`.
        delta (torch.Tensor): Input-dependent step sizes `[B, D, N]`.
        state_matrix (torch.Tensor): Continuous state matrix `[D, S]`.
        b_state (torch.Tensor): Input-dependent B values `[B, S, N]`.
        c_state (torch.Tensor): Input-dependent C values `[B, S, N]`.
        skip (torch.Tensor): Skip parameter `[D]`.
        gate (torch.Tensor): Mamba gate values `[B, D, N]`.
        delta_bias (torch.Tensor): Learned delta bias `[D]`.

    Returns:
        torch.Tensor: Selective scan output `[B, D, N]`.
    """
    step = F.softplus(delta + delta_bias.view(1, -1, 1))
    state = torch.zeros(
        x.shape[0],
        x.shape[1],
        state_matrix.shape[1],
        dtype=x.dtype,
        device=x.device,
    )
    outputs = []
    state_matrix = state_matrix.to(dtype=x.dtype)
    for token_idx in range(x.shape[-1]):
        token_step = step[:, :, token_idx]
        transition = torch.exp(token_step.unsqueeze(-1) * state_matrix)
        input_update = (
            token_step.unsqueeze(-1)
            * b_state[:, :, token_idx].unsqueeze(1)
            * x[:, :, token_idx].unsqueeze(-1)
        )
        state = transition * state + input_update
        token_output = torch.sum(
            state * c_state[:, :, token_idx].unsqueeze(1), dim=-1
        )
        token_output = token_output + skip * x[:, :, token_idx]
        outputs.append(token_output * F.silu(gate[:, :, token_idx]))
    return torch.stack(outputs, dim=-1)


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
