"""
Basic tests for SAVOR: Skill Affordance Learning from Visuo-Haptic Perception
for Robot-Assisted Bite Acquisition

This module contains basic tests to ensure the core functionality works correctly.
"""

import torch
import numpy as np
import pytest
from model import SAVORNet


class TestSAVORNet:
    """Test SAVOR-Net architecture."""
    
    def test_savor_net_initialization(self):
        """Test SAVORNet initialization."""
        SEQ_LENGTH = 20
        model = SAVORNet(seq_length=SEQ_LENGTH, feature_dim=128, lstm_hidden_dim=512)
        
        # Check that all components are initialized
        assert hasattr(model, 'rgb_encoder')
        assert hasattr(model, 'depth_encoder')
        assert hasattr(model, 'force_encoder')
        assert hasattr(model, 'pose_encoder')
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'final_mlp')
        
        # Check parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        # print(f"   SAVOR-Net parameters: {total_params:,}")
    
    def test_savor_net_forward(self):
        """Test SAVORNet forward pass."""
        SEQ_LENGTH = 20
        model = SAVORNet(seq_length=SEQ_LENGTH, feature_dim=128, lstm_hidden_dim=512)
        
        # Create sample inputs
        batch_size = 2
        rgb_images = torch.randn(batch_size, SEQ_LENGTH, 3, 224, 224)
        depth_images = torch.randn(batch_size, SEQ_LENGTH, 1, 224, 224)
        force_data = torch.randn(batch_size, SEQ_LENGTH, 6)
        pose_data = torch.randn(batch_size, SEQ_LENGTH, 6)
        
        # Forward pass
        output = model(rgb_images, depth_images, force_data, pose_data)
        
        # Check output shape
        assert output.shape == (batch_size, SEQ_LENGTH, 3, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Check that output values are reasonable (logits can be any value)
        assert torch.all(torch.isfinite(output))
    
    def test_encoder_outputs(self):
        """Test individual encoder outputs."""
        SEQ_LENGTH = 20
        model = SAVORNet(seq_length=SEQ_LENGTH, feature_dim=128, lstm_hidden_dim=512)
        
        batch_size = 2
        seq_length = SEQ_LENGTH
        
        # Test RGB encoder
        rgb_input = torch.randn(batch_size * seq_length, 3, 224, 224)
        rgb_output = model.rgb_encoder(rgb_input)
        assert rgb_output.shape == (batch_size * seq_length, 128)
        
        # Test depth encoder
        depth_input = torch.randn(batch_size * seq_length, 1, 224, 224)
        depth_output = model.depth_encoder(depth_input)
        assert depth_output.shape == (batch_size * seq_length, 128)
        
        # Test force encoder
        force_input = torch.randn(batch_size, seq_length, 6)
        force_output = model.force_encoder(force_input)
        assert force_output.shape == (batch_size, seq_length, 128)
        
        # Test pose encoder
        pose_input = torch.randn(batch_size, seq_length, 6)
        pose_output = model.pose_encoder(pose_input)
        assert pose_output.shape == (batch_size, seq_length, 128)


class TestTrainingComponents:
    """Test training-related components."""
    
    def test_loss_computation(self):
        """Test loss computation with new architecture."""
        SEQ_LENGTH = 20
        model = SAVORNet(seq_length=SEQ_LENGTH)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Create sample data
        batch_size = 2
        rgb_images = torch.randn(batch_size, SEQ_LENGTH, 3, 224, 224)
        depth_images = torch.randn(batch_size, SEQ_LENGTH, 1, 224, 224)
        force_data = torch.randn(batch_size, SEQ_LENGTH, 6)
        pose_data = torch.randn(batch_size, SEQ_LENGTH, 6)
        scores = torch.randint(1, 6, (batch_size, 6, 1, 1)).float()  # 1-5 range
        
        # Forward pass
        outputs = model(rgb_images, depth_images, force_data, pose_data)
        
        # Prepare scores for classification
        scores_expanded = scores.squeeze(-1).squeeze(-1)[:, :3]  # [batch_size, 3]
        class_labels = (scores_expanded - 1).long()  # Convert to class labels (0-4)
        class_labels = class_labels.unsqueeze(1).repeat(1, SEQ_LENGTH, 1)  # [batch_size, seq_length, 3]
        
        # Compute loss for each attribute
        loss = 0.0
        for attr_idx in range(3):
            attr_output = outputs[:, :, attr_idx, :]  # [batch_size, seq_length, 5]
            attr_labels = class_labels[:, :, attr_idx]  # [batch_size, seq_length]
            # Reshape for cross-entropy: [batch_size * seq_length, 5] and [batch_size * seq_length]
            attr_output_flat = attr_output.view(-1, 5)  # [batch_size * seq_length, 5]
            attr_labels_flat = attr_labels.view(-1)  # [batch_size * seq_length]
            
            loss += criterion(attr_output_flat, attr_labels_flat)
        loss = loss / 3
        
        # Check loss properties
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_gradient_computation(self):
        """Test gradient computation with new architecture."""
        SEQ_LENGTH = 20
        model = SAVORNet(seq_length=SEQ_LENGTH)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create sample data
        batch_size = 2
        rgb_images = torch.randn(batch_size, SEQ_LENGTH, 3, 224, 224)
        depth_images = torch.randn(batch_size, SEQ_LENGTH, 1, 224, 224)
        force_data = torch.randn(batch_size, SEQ_LENGTH, 6)
        pose_data = torch.randn(batch_size, SEQ_LENGTH, 6)
        scores = torch.randint(1, 6, (batch_size, 6, 1, 1)).float()  # 1-5 range
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(rgb_images, depth_images, force_data, pose_data)
        
        # Prepare scores for classification
        scores_expanded = scores.squeeze(-1).squeeze(-1)[:, :3]  # [batch_size, 3]
        class_labels = (scores_expanded - 1).long()  # Convert to class labels (0-4)
        class_labels = class_labels.unsqueeze(1).repeat(1, SEQ_LENGTH, 1)  # [batch_size, seq_length, 3]
        
        # Compute loss for each attribute
        loss = 0.0
        for attr_idx in range(3):
            attr_output = outputs[:, :, attr_idx, :]  # [batch_size, seq_length, 5]
            attr_labels = class_labels[:, :, attr_idx]  # [batch_size, seq_length]
            # Reshape for cross-entropy: [batch_size * seq_length, 5] and [batch_size * seq_length]
            attr_output_flat = attr_output.view(-1, 5)  # [batch_size * seq_length, 5]
            attr_labels_flat = attr_labels.view(-1)  # [batch_size * seq_length]
            
            loss += criterion(attr_output_flat, attr_labels_flat)
        loss = loss / 3
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()


def test_model_consistency():
    """Test model consistency across different inputs."""
    SEQ_LENGTH = 20
    model = SAVORNet(seq_length=SEQ_LENGTH)
    model.eval()
    
    # Create sample data
    batch_size = 2
    rgb_images = torch.randn(batch_size, SEQ_LENGTH, 3, 224, 224)
    depth_images = torch.randn(batch_size, SEQ_LENGTH, 1, 224, 224)
    force_data = torch.randn(batch_size, SEQ_LENGTH, 6)
    pose_data = torch.randn(batch_size, SEQ_LENGTH, 6)
    
    with torch.no_grad():
        # First forward pass
        output1 = model(rgb_images, depth_images, force_data, pose_data)
        
        # Second forward pass (should be identical)
        output2 = model(rgb_images, depth_images, force_data, pose_data)
        
        # Check consistency
        assert torch.allclose(output1, output2, atol=1e-6)


def test_device_compatibility():
    """Test model compatibility with different devices."""
    SEQ_LENGTH = 20
    model = SAVORNet(seq_length=SEQ_LENGTH)
    
    # Test CPU
    model_cpu = model.cpu()
    rgb_images_cpu = torch.randn(1, SEQ_LENGTH, 3, 224, 224)
    depth_images_cpu = torch.randn(1, SEQ_LENGTH, 1, 224, 224)
    force_data_cpu = torch.randn(1, SEQ_LENGTH, 6)
    pose_data_cpu = torch.randn(1, SEQ_LENGTH, 6)
    
    with torch.no_grad():
        output_cpu = model_cpu(rgb_images_cpu, depth_images_cpu, force_data_cpu, pose_data_cpu)
        assert output_cpu.device.type == 'cpu'
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        rgb_images_cuda = rgb_images_cpu.cuda()
        depth_images_cuda = depth_images_cpu.cuda()
        force_data_cuda = force_data_cpu.cuda()
        pose_data_cuda = pose_data_cpu.cuda()
        
        with torch.no_grad():
            output_cuda = model_cuda(rgb_images_cuda, depth_images_cuda, force_data_cuda, pose_data_cuda)
            assert output_cuda.device.type == 'cuda'


def test_data_shapes():
    """Test that data shapes are correct for the new architecture."""
    # Test input shapes
    batch_size = 2
    SEQ_LENGTH = 20
    
    rgb_images = torch.randn(batch_size, SEQ_LENGTH, 3, 224, 224)
    depth_images = torch.randn(batch_size, SEQ_LENGTH, 1, 224, 224)
    force_data = torch.randn(batch_size, SEQ_LENGTH, 6)
    pose_data = torch.randn(batch_size, SEQ_LENGTH, 6)
    
    # Verify shapes
    assert rgb_images.shape == (batch_size, SEQ_LENGTH, 3, 224, 224)
    assert depth_images.shape == (batch_size, SEQ_LENGTH, 1, 224, 224)
    assert force_data.shape == (batch_size, SEQ_LENGTH, 6)
    assert pose_data.shape == (batch_size, SEQ_LENGTH, 6)
    
    # Test model with these shapes
    model = SAVORNet(seq_length=SEQ_LENGTH)
    output = model(rgb_images, depth_images, force_data, pose_data)
    assert output.shape == (batch_size, SEQ_LENGTH, 3, 5)


if __name__ == "__main__":
    # Run basic tests
    print("Running SAVOR basic tests...")
    
    # Test SAVOR-Net
    test_savor = TestSAVORNet()
    test_savor.test_savor_net_initialization()
    test_savor.test_savor_net_forward()
    test_savor.test_encoder_outputs()
    print("[Pass] SAVOR-Net tests")
    
    # Test training components
    test_training = TestTrainingComponents()
    test_training.test_loss_computation()
    test_training.test_gradient_computation()
    print("[Pass] Training component tests")
    
    # Test additional functionality
    test_model_consistency()
    test_device_compatibility()
    test_data_shapes()
    print("[Pass] Additional tests")
    
    print("All tests passed successfully!")