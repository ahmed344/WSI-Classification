"""
CLAM-MB (Multi-Branch) model implementation.

This module implements the CLAM-MB architecture for weakly supervised learning
on whole slide images. The model uses multiple attention branches (one per class)
and a clustering branch for unsupervised learning constraints.
"""
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionBranch(nn.Module):
    """
    Single attention branch for CLAM-MB.
    
    Uses attention pooling to aggregate patch features into a single representation
    for classification. Each branch learns to attend to class-specific regions.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 512) -> None:
        """
        Initialize attention branch.
        
        Args:
            input_dim (int): Dimension of input features (typically 1536 for H-Optimus).
            hidden_dim (int): Dimension of hidden layers. Defaults to 512.
        """
        super(AttentionBranch, self).__init__()
        # Intermediate dimensions for attention computation
        self.L = 500  # Intermediate dimension for attention network
        self.D = 256  # Attention dimension
        
        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # Attention network: computes attention weights for each tile
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, self.L),
            nn.Tanh(),
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)  # Output single attention score per tile
        )
        
        # Classification head: maps aggregated features to logit
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention branch.
        
        Args:
            features (torch.Tensor): Input features of shape [batch_size, n_tiles, input_dim].
            mask (Optional[torch.Tensor]): Boolean mask of shape [batch_size, n_tiles].
                True indicates valid tiles, False indicates padding. Defaults to None.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - logits (torch.Tensor): Classification logits of shape [batch_size, 1].
                - attention_weights (torch.Tensor): Attention weights of shape [batch_size, n_tiles].
        """
        # Extract features through feature extractor
        H = self.feature_extractor(features)  # [batch_size, n_tiles, hidden_dim]
        
        # Compute attention scores for each tile
        A = self.attention(H)  # [batch_size, n_tiles, 1]
        A = A.squeeze(-1)  # [batch_size, n_tiles]
        
        # Apply mask to ignore padding tiles
        if mask is not None:
            A = A.masked_fill(~mask, float('-inf'))
        
        # Normalize attention weights with softmax
        A = F.softmax(A, dim=1)  # [batch_size, n_tiles]
        
        # Attention pooling: weighted sum of features
        M = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # [batch_size, hidden_dim]
        
        # Classification: map aggregated features to logit
        logits = self.classifier(M)  # [batch_size, 1]
        
        return logits, A


class ClusteringBranch(nn.Module):
    """
    Clustering branch for unsupervised learning constraint.
    
    Uses K-means-like clustering to separate patches into clusters, encouraging
    the model to learn discriminative features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        k_clusters: int = 2
    ) -> None:
        """
        Initialize clustering branch.
        
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of hidden layers. Defaults to 512.
            k_clusters (int): Number of clusters to assign tiles to. Defaults to 2.
        """
        super(ClusteringBranch, self).__init__()
        self.k_clusters = k_clusters
        
        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # Cluster assignment network: maps features to cluster logits
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, k_clusters)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through clustering branch.
        
        Args:
            features (torch.Tensor): Input features of shape [batch_size, n_tiles, input_dim].
            mask (Optional[torch.Tensor]): Boolean mask of shape [batch_size, n_tiles].
                True indicates valid tiles, False indicates padding. Defaults to None.
        
        Returns:
            torch.Tensor: Soft cluster assignments of shape [batch_size, n_tiles, k_clusters].
                Each row sums to 1 (softmax probabilities).
        """
        # Extract features
        H = self.feature_extractor(features)  # [batch_size, n_tiles, hidden_dim]
        
        # Get cluster assignment logits
        cluster_logits = self.cluster_head(H)  # [batch_size, n_tiles, k_clusters]
        
        # Apply mask to ignore padding tiles
        if mask is not None:
            # Set invalid tiles to uniform distribution (via -inf before softmax)
            mask_expanded = mask.unsqueeze(-1).expand_as(cluster_logits)
            cluster_logits = cluster_logits.masked_fill(~mask_expanded, float('-inf'))
        
        # Convert logits to soft cluster assignments (probabilities)
        cluster_assignments = F.softmax(cluster_logits, dim=-1)
        
        return cluster_assignments


class CLAM_MB(nn.Module):
    """
    CLAM-MB (Multi-Branch) model for WSI classification.
    
    Uses multiple attention branches (one per class) and a clustering branch.
    Each attention branch learns to attend to class-specific regions, and the
    clustering branch provides unsupervised learning constraints.
    """
    
    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        num_classes: int = 5,
        k_clusters: int = 2,
        dropout: float = 0.25
    ) -> None:
        """
        Initialize CLAM-MB model.
        
        Args:
            input_dim (int): Dimension of input features. Defaults to 1536.
            hidden_dim (int): Dimension of hidden layers. Defaults to 512.
            num_classes (int): Number of classification classes. Defaults to 5.
            k_clusters (int): Number of clusters for clustering branch. Defaults to 2.
            dropout (float): Dropout probability. Defaults to 0.25.
        """
        super(CLAM_MB, self).__init__()
        self.num_classes = num_classes
        self.k_clusters = k_clusters
        
        # Shared feature projection: projects input features to hidden dimension
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple attention branches: one per class
        # Each branch learns to attend to class-specific regions
        self.attention_branches = nn.ModuleList([
            AttentionBranch(hidden_dim, hidden_dim) for _ in range(num_classes)
        ])
        
        # Clustering branch: provides unsupervised learning constraint
        self.clustering_branch = ClusteringBranch(
            hidden_dim, hidden_dim, k_clusters
        )
        
        # Final classification head: combines branch outputs
        self.classifier = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CLAM-MB model.
        
        Args:
            features (torch.Tensor): Input features of shape [batch_size, n_tiles, input_dim].
            mask (Optional[torch.Tensor]): Boolean mask of shape [batch_size, n_tiles].
                True indicates valid tiles, False indicates padding. Defaults to None.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'logits' (torch.Tensor): Final classification logits of shape
                  [batch_size, num_classes].
                - 'attention_weights' (List[torch.Tensor]): List of attention weight tensors,
                  one per branch. Each has shape [batch_size, n_tiles].
                - 'cluster_assignments' (torch.Tensor): Cluster assignments of shape
                  [batch_size, n_tiles, k_clusters].
                - 'branch_logits' (torch.Tensor): Branch logits before final classifier,
                  shape [batch_size, num_classes].
        """
        # Project input features to hidden dimension
        H = self.feature_projection(features)  # [batch_size, n_tiles, hidden_dim]
        
        # Get outputs from each attention branch
        branch_outputs: List[torch.Tensor] = []
        attention_weights_list: List[torch.Tensor] = []
        
        for branch in self.attention_branches:
            logits_branch, attn_weights = branch(H, mask)
            branch_outputs.append(logits_branch)
            attention_weights_list.append(attn_weights)
        
        # Stack branch outputs: [batch_size, num_classes]
        branch_logits = torch.cat(branch_outputs, dim=1)
        
        # Get clustering assignments
        cluster_assignments = self.clustering_branch(H, mask)
        
        # Final classification: combine branch outputs
        final_logits = self.classifier(branch_logits)  # [batch_size, num_classes]
        
        return {
            'logits': final_logits,
            'attention_weights': attention_weights_list,
            'cluster_assignments': cluster_assignments,
            'branch_logits': branch_logits
        }


def compute_clustering_loss(
    cluster_assignments: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute clustering loss to encourage confident cluster assignments.
    
    Uses entropy minimization: encourages the model to make confident (low entropy)
    cluster assignments rather than uncertain (high entropy) ones.
    
    Args:
        cluster_assignments (torch.Tensor): Soft cluster assignments of shape
            [batch_size, n_tiles, k_clusters]. Each row should sum to 1.
        mask (Optional[torch.Tensor]): Boolean mask of shape [batch_size, n_tiles].
            True indicates valid tiles, False indicates padding. Defaults to None.
    
    Returns:
        torch.Tensor: Scalar tensor containing the clustering loss (entropy).
    """
    if mask is not None:
        # Only compute loss for valid tiles (ignore padding)
        valid_assignments = cluster_assignments[mask]  # [n_valid_tiles, k_clusters]
        if valid_assignments.shape[0] == 0:
            # Return zero loss if no valid tiles
            return torch.tensor(0.0, device=cluster_assignments.device)
        
        # Compute entropy: -sum(p * log(p)) for each tile
        entropy = -torch.sum(
            valid_assignments * torch.log(valid_assignments + 1e-8), dim=-1
        )
        clustering_loss = torch.mean(entropy)
    else:
        # Compute entropy for all tiles
        entropy = -torch.sum(
            cluster_assignments * torch.log(cluster_assignments + 1e-8), dim=-1
        )
        clustering_loss = torch.mean(entropy)
    
    # Return entropy as loss (we want to minimize entropy = maximize confidence)
    return clustering_loss
