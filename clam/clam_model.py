import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionBranch(nn.Module):
    """
    Single attention branch for CLAM.
    Uses attention pooling to aggregate patch features.
    """
    def __init__(self, input_dim, hidden_dim=512):
        super(AttentionBranch, self).__init__()
        self.L = 500  # Intermediate dimension
        self.D = 256  # Attention dimension
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, self.L),
            nn.Tanh(),
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features, mask=None):
        """
        Args:
            features: [batch_size, n_tiles, input_dim]
            mask: [batch_size, n_tiles] - True for valid tiles
        
        Returns:
            logits: [batch_size, 1]
            attention_weights: [batch_size, n_tiles]
        """
        # Extract features
        H = self.feature_extractor(features)  # [batch_size, n_tiles, hidden_dim]
        
        # Compute attention weights
        A = self.attention(H)  # [batch_size, n_tiles, 1]
        A = A.squeeze(-1)  # [batch_size, n_tiles]
        
        # Apply mask if provided
        if mask is not None:
            A = A.masked_fill(~mask, float('-inf'))
        
        # Softmax attention weights
        A = F.softmax(A, dim=1)  # [batch_size, n_tiles]
        
        # Attention pooling
        M = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # [batch_size, hidden_dim]
        
        # Classification
        logits = self.classifier(M)  # [batch_size, 1]
        
        return logits, A


class ClusteringBranch(nn.Module):
    """
    Clustering branch for unsupervised learning constraint.
    Uses K-means-like clustering (K=2) to separate patches.
    """
    def __init__(self, input_dim, hidden_dim=512, k_clusters=2):
        super(ClusteringBranch, self).__init__()
        self.k_clusters = k_clusters
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # Cluster assignment network
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, k_clusters)
        )
    
    def forward(self, features, mask=None):
        """
        Args:
            features: [batch_size, n_tiles, input_dim]
            mask: [batch_size, n_tiles] - True for valid tiles
        
        Returns:
            cluster_assignments: [batch_size, n_tiles, k_clusters] - soft assignments
        """
        H = self.feature_extractor(features)  # [batch_size, n_tiles, hidden_dim]
        
        # Get cluster assignments
        cluster_logits = self.cluster_head(H)  # [batch_size, n_tiles, k_clusters]
        
        # Apply mask if provided
        if mask is not None:
            # Set invalid tiles to uniform distribution
            mask_expanded = mask.unsqueeze(-1).expand_as(cluster_logits)
            cluster_logits = cluster_logits.masked_fill(~mask_expanded, float('-inf'))
        
        # Softmax to get soft cluster assignments
        cluster_assignments = F.softmax(cluster_logits, dim=-1)
        
        return cluster_assignments


class CLAM_MB(nn.Module):
    """
    CLAM-MB (Multi-Branch) model for WSI classification.
    Uses multiple attention branches (one per class) and a clustering branch.
    """
    def __init__(self, input_dim=1536, hidden_dim=512, num_classes=5, k_clusters=2, dropout=0.25):
        super(CLAM_MB, self).__init__()
        self.num_classes = num_classes
        self.k_clusters = k_clusters
        
        # Shared feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple attention branches (one per class)
        self.attention_branches = nn.ModuleList([
            AttentionBranch(hidden_dim, hidden_dim) for _ in range(num_classes)
        ])
        
        # Clustering branch
        self.clustering_branch = ClusteringBranch(hidden_dim, hidden_dim, k_clusters)
        
        # Final classification head (combines all branches)
        self.classifier = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, features, mask=None):
        """
        Args:
            features: [batch_size, n_tiles, input_dim]
            mask: [batch_size, n_tiles] - True for valid tiles
        
        Returns:
            logits: [batch_size, num_classes]
            attention_weights: list of [batch_size, n_tiles] tensors (one per branch)
            cluster_assignments: [batch_size, n_tiles, k_clusters]
        """
        # Project features
        H = self.feature_projection(features)  # [batch_size, n_tiles, hidden_dim]
        
        # Get outputs from each attention branch
        branch_outputs = []
        attention_weights_list = []
        
        for branch in self.attention_branches:
            logits_branch, attn_weights = branch(H, mask)
            branch_outputs.append(logits_branch)
            attention_weights_list.append(attn_weights)
        
        # Stack branch outputs
        branch_logits = torch.cat(branch_outputs, dim=1)  # [batch_size, num_classes]
        
        # Get clustering assignments
        cluster_assignments = self.clustering_branch(H, mask)
        
        # Final classification
        final_logits = self.classifier(branch_logits)  # [batch_size, num_classes]
        
        return {
            'logits': final_logits,
            'attention_weights': attention_weights_list,
            'cluster_assignments': cluster_assignments,
            'branch_logits': branch_logits
        }


def compute_clustering_loss(cluster_assignments, mask=None):
    """
    Compute clustering loss to encourage separation between clusters.
    Uses entropy minimization to encourage confident cluster assignments.
    
    Args:
        cluster_assignments: [batch_size, n_tiles, k_clusters]
        mask: [batch_size, n_tiles] - True for valid tiles
    
    Returns:
        clustering_loss: scalar tensor
    """
    if mask is not None:
        # Only compute loss for valid tiles
        valid_assignments = cluster_assignments[mask]  # [n_valid_tiles, k_clusters]
        if valid_assignments.shape[0] == 0:
            return torch.tensor(0.0, device=cluster_assignments.device)
        
        # Entropy minimization: encourage confident assignments
        entropy = -torch.sum(valid_assignments * torch.log(valid_assignments + 1e-8), dim=-1)
        clustering_loss = torch.mean(entropy)
    else:
        entropy = -torch.sum(cluster_assignments * torch.log(cluster_assignments + 1e-8), dim=-1)
        clustering_loss = torch.mean(entropy)
    
    # We want to minimize entropy (maximize confidence), so we return it directly
    return clustering_loss

