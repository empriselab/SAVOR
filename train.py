#!/usr/bin/env python3
"""
Training script for SavorNet
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataset_processor import rlds_dataset_processor
from model import SAVORNet, save_checkpoint, load_checkpoint


class SavorDataset(Dataset):
    """PyTorch Dataset wrapper for SAVOR data loaded directly."""
    
    def __init__(self, data_list, sequence_length=40):
        self.data_list = data_list
        self.sequence_length = sequence_length
        self.data = []
        self._process_data()
    
    def _process_data(self):
        """Process data from the loaded data list."""
        # print("Processing data for PyTorch...")
        for item in self.data_list:
            # Convert to PyTorch format: [seq, channels, height, width]
            rgb_seq = torch.from_numpy(item['rgb_sequence']).permute(0, 3, 1, 2).float()  # [T, C, H, W]
            
            # Handle depth sequence - check if it has channel dimension
            depth_seq = torch.from_numpy(item['depth_sequence']).float()
            if depth_seq.dim() == 3:  # [T, H, W] - no channel dimension
                depth_seq = depth_seq.unsqueeze(1)  # Add channel dimension: [T, 1, H, W]
            else:  # [T, H, W, C] - has channel dimension
                depth_seq = depth_seq.permute(0, 3, 1, 2)  # [T, C, H, W]
            
            pose_seq = torch.from_numpy(item['pose_sequence']).float()  # [T, 6]
            force_seq = torch.from_numpy(item['force_sequence']).float()  # [T, 6]
            physical_props = torch.from_numpy(item['physical_properties']).float()  # [T, 3]
            
            # Create labels for classification (convert 1-5 scores to 0-4 class labels)
            # Handle case where physical properties might be zeros (fallback to class 2 = score 3)
            if torch.any(physical_props == 0):
                print(f"[WARNING]: Some physical properties are zeros, using fallback values (score 3 = class 2)")
                physical_props = torch.where(physical_props == 0, torch.tensor(3.0), physical_props)  # Replace zeros with 3
            
            labels = (physical_props - 1).long()  # [T, 3] - convert 1-5 to 0-4 for cross-entropy
            
            self.data.append({
                'rgb': rgb_seq,  # [T, C, H, W]
                'depth': depth_seq,  # [T, C, H, W]
                'pose': pose_seq,  # [T, 6]
                'force': force_seq,  # [T, 6]
                'labels': labels  # [T, 3]
            })
        
        # print(f"Processed {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def train_model(data_dir, batch_size=32, epochs=50, save_dir="./checkpoints", 
                sequence_length=40, learning_rate=1e-4, device='cuda', max_episodes=10, augment=False):
    """Train SavorNet model with PyTorch."""
    print("Starting SavorNet training with PyTorch...")
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    # Clear GPU memory if using CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Create data processor
    processor = rlds_dataset_processor(
        data_dir=data_dir,
        batch_size=batch_size,
        sequence_length=sequence_length,
        image_size=(224, 224),
        max_episodes=max_episodes,  # Limit episodes for memory efficiency
        augment=augment,  # Enable data augmentation
        val_split=0.2,  # 20% for validation
        random_seed=42  # Reproducible splits
    )
    
    # Get training and validation data with proper split
    train_data = processor.get_data(split='train')
    val_data = processor.get_data(split='val')
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Convert to PyTorch datasets
    train_dataset = SavorDataset(train_data, sequence_length)
    val_dataset = SavorDataset(val_data, sequence_length)
    
    # Create PyTorch data loaders with small batch size to avoid memory issues
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Create model
    model = SAVORNet(seq_length=sequence_length, feature_dim=128, lstm_hidden_dim=512)
    model = model.to(device)
    
    # Loss and optimizer - use CrossEntropyLoss for classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        train_correct = 0
        train_total = 0
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            rgb = batch['rgb'].to(device)  # [1, T, C, H, W]
            depth = batch['depth'].to(device)  # [1, T, C, H, W]
            pose = batch['pose'].to(device)  # [1, T, 6]
            force = batch['force'].to(device)  # [1, T, 6]
            labels = batch['labels'].to(device)  # [1, T, 3] - class labels 0-4
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(rgb, depth, force, pose)  # [1, T, 3, 5]
            
            # Calculate loss for each attribute separately
            total_loss = 0.0
            for attr_idx in range(3):  # softness, moisture, viscosity
                attr_predictions = predictions[:, :, attr_idx, :]  # [1, T, 5]
                attr_labels = labels[:, :, attr_idx]  # [1, T]
                
                # Reshape for cross-entropy: [1*T, 5] and [1*T]
                attr_predictions_flat = attr_predictions.view(-1, 5)  # [T, 5]
                attr_labels_flat = attr_labels.view(-1)  # [T]
                
                loss = criterion(attr_predictions_flat, attr_labels_flat)
                total_loss += loss
            
            # Average loss across attributes
            loss = total_loss / 3
            
            # Calculate accuracy
            with torch.no_grad():
                # Get predicted classes
                predicted_classes = torch.argmax(predictions, dim=-1)  # [1, T, 3]
                correct = (predicted_classes == labels).sum().item()
                total = labels.numel()
                train_correct += correct
                train_total += total
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar with current metrics
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Acc': f'{correct/total:.4f}',
                'Avg_Loss': f'{train_loss/(batch_idx+1):.6f}'
            })
            
            if batch_idx % 10 == 0:
                # Calculate confidence scores for logging
                with torch.no_grad():
                    confidence_results = model.predict_with_confidence(rgb, depth, force, pose)
                    avg_confidence = confidence_results['confidence_scores'].mean().item()
                    # print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}, Acc: {correct/total:.4f}, Conf: {avg_confidence:.4f}')
        
        avg_train_loss = train_loss / train_batches
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_correct = 0
        val_total = 0
        
        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                rgb = batch['rgb'].to(device)
                depth = batch['depth'].to(device)
                pose = batch['pose'].to(device)
                force = batch['force'].to(device)
                labels = batch['labels'].to(device)
                
                predictions = model(rgb, depth, force, pose)
                
                # Calculate loss for each attribute separately
                total_loss = 0.0
                for attr_idx in range(3):  # softness, moisture, viscosity
                    attr_predictions = predictions[:, :, attr_idx, :]  # [1, T, 5]
                    attr_labels = labels[:, :, attr_idx]  # [1, T]
                    
                    # Reshape for cross-entropy: [1*T, 5] and [1*T]
                    attr_predictions_flat = attr_predictions.view(-1, 5)  # [T, 5]
                    attr_labels_flat = attr_labels.view(-1)  # [T]
                    
                    loss = criterion(attr_predictions_flat, attr_labels_flat)
                    total_loss += loss
                
                # Average loss across attributes
                loss = total_loss / 3
                
                # Calculate accuracy
                predicted_classes = torch.argmax(predictions, dim=-1)  # [1, T, 3]
                correct = (predicted_classes == labels).sum().item()
                total = labels.numel()
                val_correct += correct
                val_total += total
                
                val_loss += loss.item()
                val_batches += 1
                
                # Update validation progress bar
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Acc': f'{correct/total:.4f}',
                    'Avg_Loss': f'{val_loss/(val_batches):.6f}'
                })
        
        avg_val_loss = val_loss / val_batches
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:',f'Train Loss: {avg_train_loss:.6f}',f'Train Acc: {train_accuracy:.4f}',f'Val Loss: {avg_val_loss:.6f}',f'Val Acc: {val_accuracy:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(epoch, model, optimizer, avg_val_loss, 
                          os.path.join(save_dir, 'best_model.pth'))

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(epoch, model, optimizer, avg_val_loss,
                          os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    save_checkpoint(epochs-1, model, optimizer, avg_val_loss,
                   os.path.join(save_dir, 'final_model.pth'))
    
    print(f"[Completed]Training completed! Models saved to {save_dir}")
    return model, train_losses, val_losses, train_accuracies, val_accuracies


def test_dataloader(data_dir):
    """Test the data loader."""
    print("Testing data loader...")
    
    processor = rlds_dataset_processor(
        data_dir=data_dir,
        batch_size=2,
        sequence_length=5,
        max_episodes=3
    )
    
    rlds_data = processor.get_data()
    print(f"Loaded {len(rlds_data)} training samples")
    
    if len(rlds_data) > 0:
        sample = rlds_data[0]
        print(f"Sample data:")
        print(f"  RGB shape: {sample['rgb_sequence'].shape}")
        print(f"  Depth shape: {sample['depth_sequence'].shape}")
        print(f"  Pose shape: {sample['pose_sequence'].shape}")
        print(f"  Force/Torque shape: {sample['force_sequence'].shape}")
        print(f"  Physical properties shape: {sample['physical_properties'].shape}")
        print(f"  Physical properties values (softness, moisture, viscosity):")
        print(f"    First timestep: {sample['physical_properties'][0]}")
        print(f"    Last timestep: {sample['physical_properties'][-1]}")
        print(f"    Range: {sample['physical_properties'].min(axis=0)} to {sample['physical_properties'].max(axis=0)}")
        
        # Check if all zeros
        if np.all(sample['physical_properties'] == 0):
            print(f"[WARNING]: All physical properties are zeros!")
            print(f"[WARNING]: This will cause training issues. Please fix the score_template.csv issue.")
    else:
        print("No data available")
    
    print("Data loader test completed!")


def main():
    parser = argparse.ArgumentParser(description="Train SavorNet with PyTorch")
    parser.add_argument("--data_dir", required=True, help="Directory containing the RLDS dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--sequence_length", type=int, default=40, help="Sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--max_episodes", type=int, default=10, help="Maximum number of episodes to process")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio (0.0 to 1.0)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducible splits")
    parser.add_argument("--test_only", action="store_true", help="Only test the data loader")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_dataloader(args.data_dir)
    else:
        train_model(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            sequence_length=args.sequence_length,
            learning_rate=args.learning_rate,
            save_dir=args.save_dir,
            device=args.device,
            max_episodes=args.max_episodes,
            augment=args.augment
        )


if __name__ == "__main__":
    main()