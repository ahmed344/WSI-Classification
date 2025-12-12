"""
Integration test for CLAM-MB pipeline.

This module provides end-to-end integration tests to verify that the CLAM-MB
pipeline works correctly, including data loading, model initialization, forward
pass, loss computation, and backward pass.
"""
import torch
import sys
from torch.utils.data import DataLoader

from clam_dataset import WSIFeatureDataset, collate_fn
from clam_model import CLAM_MB, compute_clustering_loss


def test_integration() -> bool:
    """
    Run integration tests for CLAM-MB pipeline.
    
    Tests the following components:
    1. Data loading from dataset
    2. DataLoader with collate function
    3. Model initialization
    4. Forward pass through model
    5. Loss computation (classification + clustering)
    6. Backward pass (gradient computation)
    
    Returns:
        bool: True if all tests pass, False otherwise.
    """
    print("Testing CLAM-MB Integration...")
    print("="*60)
    
    # Test 1: Data Loading
    print("\n1. Testing data loading...")
    try:
        data_root = '/workspaces/WSI-Classification/data/HE-MYO/Processed'
        train_dataset = WSIFeatureDataset(
            data_root,
            split='train',
            train_ratio=0.9,
            random_seed=42
        )
        
        # Check if dataset is empty
        if len(train_dataset) == 0:
            print("   WARNING: No training samples found. Check data path.")
            return False
        
        print(f"   ✓ Loaded {len(train_dataset)} training samples")
        print(f"   ✓ Classes: {train_dataset.class_folders}")
        
        # Get a sample and verify structure
        sample = train_dataset[0]
        print(f"   ✓ Sample features shape: {sample['features'].shape}")
        print(
            f"   ✓ Sample label: {sample['label']} "
            f"({train_dataset.idx_to_class[sample['label']]})"
        )
        
    except Exception as e:
        print(f"   ✗ Data loading failed: {e}")
        return False
    
    # Test 2: DataLoader with collate function
    print("\n2. Testing DataLoader...")
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Get a batch and verify structure
        batch = next(iter(train_loader))
        print(f"   ✓ Batch features shape: {batch['features'].shape}")
        print(f"   ✓ Batch labels shape: {batch['labels'].shape}")
        print(f"   ✓ Batch masks shape: {batch['masks'].shape}")
        print(f"   ✓ Number of valid tiles: {batch['masks'].sum().item()}")
        
    except Exception as e:
        print(f"   ✗ DataLoader failed: {e}")
        return False
    
    # Test 3: Model initialization
    print("\n3. Testing model initialization...")
    try:
        # Determine device (GPU if available, else CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {device}")
        
        # Create model with standard configuration
        model = CLAM_MB(
            input_dim=1536,
            hidden_dim=512,
            num_classes=5,
            k_clusters=2
        ).to(device)
        
        # Count and report model parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Model created with {num_params:,} parameters")
        
    except Exception as e:
        print(f"   ✗ Model initialization failed: {e}")
        return False
    
    # Test 4: Forward pass
    print("\n4. Testing forward pass...")
    try:
        # Move batch to device
        features = batch['features'].to(device)
        masks = batch['masks'].to(device)
        
        # Forward pass through model
        outputs = model(features, masks)
        
        print(f"   ✓ Logits shape: {outputs['logits'].shape}")
        print(
            f"   ✓ Number of attention branches: "
            f"{len(outputs['attention_weights'])}"
        )
        print(
            f"   ✓ Attention weights shape (per branch): "
            f"{outputs['attention_weights'][0].shape}"
        )
        print(
            f"   ✓ Cluster assignments shape: "
            f"{outputs['cluster_assignments'].shape}"
        )
        
        # Verify output shapes match expectations
        assert outputs['logits'].shape[0] == features.shape[0], \
            "Batch size mismatch"
        assert outputs['logits'].shape[1] == 5, "Number of classes mismatch"
        
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Loss computation
    print("\n5. Testing loss computation...")
    try:
        # Get labels and compute losses
        labels = batch['labels'].to(device)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Classification loss
        cls_loss = criterion(outputs['logits'], labels)
        
        # Clustering loss
        cluster_loss = compute_clustering_loss(
            outputs['cluster_assignments'], masks
        )
        
        # Combined loss (with typical clustering weight)
        total_loss = cls_loss + 0.1 * cluster_loss
        
        print(f"   ✓ Classification loss: {cls_loss.item():.4f}")
        print(f"   ✓ Clustering loss: {cluster_loss.item():.4f}")
        print(f"   ✓ Total loss: {total_loss.item():.4f}")
        
    except Exception as e:
        print(f"   ✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Backward pass
    print("\n6. Testing backward pass...")
    try:
        # Create optimizer and perform backward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Zero gradients, compute gradients, update weights
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"   ✓ Backward pass successful")
        
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # All tests passed
    print("\n" + "="*60)
    print("✓ All integration tests passed!")
    print("="*60)
    return True


if __name__ == '__main__':
    success = test_integration()
    sys.exit(0 if success else 1)
