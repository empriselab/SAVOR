"""
Example usage script for SAVOR: Skill Affordance Learning from Visuo-Haptic Perception
for Robot-Assisted Bite Acquisition

This script demonstrates how to use the SAVOR framework for food physical property prediction.
"""

import os
import torch
import numpy as np
from model import SAVORNet
from torch.utils.data import DataLoader


def create_sample_data(batch_size=1, seq_length=10):
    """
    Create sample data for demonstration purposes.
    
    Args:
        batch_size: Number of samples in the batch
        seq_length: Length of the sequence
        
    Returns:
        Tuple of (rgb_images, depth_images, force_data, pose_data, scores)
    """
    # Create sample RGB images
    rgb_images = torch.randn(batch_size, seq_length, 3, 224, 224)
    
    # Create sample depth images
    depth_images = torch.randn(batch_size, seq_length, 1, 224, 224)
    
    # Create sample force data (6D: force + torque)
    force_data = torch.randn(batch_size, seq_length, 6)
    
    # Create sample pose data (6D: position + orientation)
    pose_data = torch.randn(batch_size, seq_length, 6)
    
    # Create sample scores (softness, moisture, viscosity + initial values)
    # Food physical property is a discrete value from 1 to 5 (Likert scale)
    scores = torch.randint(1, 6, (batch_size, 6, 1, 1)).float()
    
    return rgb_images, depth_images, force_data, pose_data, scores


def demonstrate_model_usage():
    """Demonstrate basic model usage."""
    print("SAVOR Model Usage Demonstration")
    print("=" * 50)
    
    # Create sample data
    SEQ_LENGTH = 20
    print("Creating sample data...")
    rgb_images, depth_images, force_data, pose_data, scores = create_sample_data(seq_length=SEQ_LENGTH)
    print(f"   RGB Images shape: {rgb_images.shape}")
    print(f"   Depth Images shape: {depth_images.shape}")
    print(f"   Force data shape: {force_data.shape}")
    print(f"   Pose data shape: {pose_data.shape}")
    print(f"   Scores (food physical property) shape: {scores.shape}")
    
    # Initialize model
    print("\nInitializing model...")
    model = SAVORNet(seq_length=SEQ_LENGTH)
    # print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(rgb_images, depth_images, force_data, pose_data)
    print(f"   Output shape: {outputs.shape}")
    print(f"   Sample predictions: {outputs[0, 0, :].numpy()}")
    
    # Calculate loss
    print("\nCalculating loss (food physical property)...")
    criterion = torch.nn.CrossEntropyLoss()
    scores_expanded = scores.squeeze(-1).squeeze(-1)[:, :3]  # [batch_size, 3]
    # Convert to class labels (0-4) for 5 discrete Likert values
    class_labels = (scores_expanded - 1).long()  # 1-5 -> 0-4
    class_labels = class_labels.unsqueeze(1).repeat(1, SEQ_LENGTH, 1)  # [batch_size, seq_length, 3]
    
    # Calculate loss for each food physical property
    loss = 0.0
    for attr_idx in range(3):
        attr_output = outputs[:, :, attr_idx, :]  # [batch_size, seq_length, 5]
        attr_labels = class_labels[:, :, attr_idx]  # [batch_size, seq_length]

        # Reshape for cross-entropy: [batch_size * seq_length, 5] and [batch_size * seq_length]
        attr_output_flat = attr_output.view(-1, 5)  # [batch_size * seq_length, 5]
        attr_labels_flat = attr_labels.view(-1)  # [batch_size * seq_length]
        
        loss += criterion(attr_output_flat, attr_labels_flat)
    loss = loss / 3
    print(f"   Cross-Entropy Loss: {loss.item():.4f}")
    
    print("\nDemonstration completed successfully!")


def demonstrate_training_loop():
    """Demonstrate a simple training loop."""
    print("\nTraining Loop Demonstration")
    print("=" * 50)
    
    # Create sample data with smaller batch size for memory efficiency
    SEQ_LENGTH = 10
    rgb_images, depth_images, force_data, pose_data, scores = create_sample_data(batch_size=1, seq_length=SEQ_LENGTH)

    # Initialize model and optimizer
    print("Initializing model and optimizer...")
    model = SAVORNet(seq_length=SEQ_LENGTH)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(2):  # Just 2 epochs for demo
        optimizer.zero_grad()
        
        # Prepare data
        scores_expanded = scores.squeeze(-1).squeeze(-1)[:, :3]  # [batch_size, 3]
        # Convert to class labels (0-4) for 5 discrete Likert values
        class_labels = (scores_expanded - 1).long()  # 1-5 -> 0-4
        class_labels = class_labels.unsqueeze(1).repeat(1, SEQ_LENGTH, 1)  # [batch_size, seq_length, 3]
        
        # Forward pass
        outputs = model(rgb_images, depth_images, force_data, pose_data)
        
        # Calculate loss for each food physical property
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
        optimizer.step()
        
        print(f"   Epoch {epoch+1}/2, Loss: {loss.item():.4f}")
        
        # Clear cache to free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("Training demonstration completed!")


def demonstrate_data_loading():
    """Demonstrate data loading (requires actual data)."""
    print("\nRLSA Data Loading Demonstration")
    print("=" * 50)
    
    # Note: This requires actual data files
    print("Note: This demonstration requires actual data files or RLDS dataset.")
    print("   Please ensure you have the following structure:")
    print("   data/")
    print("   └── savor_rlds/")
    print("       ├── 1.0.0/")
    print("       ├── dataset_info.json")
    print("       ├── features.json")
    print("       └── savor_rlds-train.tfrecord-00000-of-00001")
    
    try:
        # Try to create dataset (will fail if data doesn't exist)
        try:
            from dataset_processor import SavorDataProcessor
        except:
            print("   SAVOR SavorDataProcessor not found. Skipping data loading demo.")
            return

        processor = SavorDataProcessor(
            data_dir="./data",
            batch_size=2,
            sequence_length=5,
            max_episodes=3,  # Only process 3 episodes for testing
            augment=False
        )
        print(f"   Dataset size: {len(processor.get_data())}")
        
        if len(processor.get_data()) > 0:
        # Create dataloader
            dataloader = DataLoader(processor.get_data(), batch_size=2, shuffle=True)
            print(f"   Number of batches: {len(dataloader)}")
            
            # Load one batch
            for images, depths, forces, poses, scores in dataloader:
                print(f"   Batch shapes:")
                print(f"     Images: {images.shape}")
                print(f"     Depths: {depths.shape}")
                print(f"     Forces: {forces.shape}")
                print(f"     Poses: {poses.shape}")
                print(f"     Scores (food physical property): {scores.shape}")
                break
        else:
            print("   No data available. Skipping data loading demo.")
    except FileNotFoundError:
        print("   Data files not found. Skipping data loading demo.")
        print("   Please add your data files to the 'data/' directory.")

def demonstrate_rawdata_loading():
    """Demonstrate raw data loading."""
    print("\nRaw Data Loading Demonstration")
    print("=" * 50)
    
    print("Checking for raw data folders...")
    print("   Expected structure:")
    print("   data/")
    print("   ├── subject1_food_rgb/")
    print("   ├── subject1_food_depth/")
    print("   ├── subject1_food_force/")
    print("   └── subject1_food_pose/")
    
    try:
        # Check if data directory exists
        if not os.path.exists("./data"):
            print("   Data directory not found.")
            print("   Please create the data directory and add your raw data folders.")
            return
        
        # Check for raw data folders
        required_folders = [
            "subject1_food_rgb",
            "subject1_food_depth", 
            "subject1_food_force",
            "subject1_food_pose"
        ]
        
        print("\n[RAW DATA] Checking raw data folders:")
        all_folders_exist = True
        
        for folder in required_folders:
            folder_path = os.path.join("./data", folder)
            if os.path.exists(folder_path):
                # Count files in folder
                file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
                print(f"{folder}/ - Found ({file_count} files)")
            else:
                print(f"{folder}/ - Missing")
                all_folders_exist = False
        
        if not all_folders_exist:
            print("\nRaw data folders are missing.")
            print("You can create the RLDS dataset from raw data using or write your own dataprocessor in self._load_raw_data().")
            return
        
        print("\nAll raw data folders found!")
        
    except Exception as e:
        print(f"[Error]Unexpected error: {e}")
        print("Please check your data setup.")

def main():
    """Main demonstration function."""
    print("SAVOR: Skill Affordance Learning from Visuo-Haptic Perception")
    print("   for Robot-Assisted Bite Acquisition")
    print("=" * 70)
    
    # Run demonstrations
    demonstrate_model_usage()
    demonstrate_training_loop()
    demonstrate_data_loading()
    demonstrate_rawdata_loading()
    
    print("\nAll demonstrations completed!")

if __name__ == "__main__":
    main()

