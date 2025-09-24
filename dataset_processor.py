#!/usr/bin/env python3
"""
SAVOR Dataset for loading and preprocessing data from RLDS structure.
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, Tuple, List
import cv2


class SavorDataProcessor:
    """Data processor that loads and preprocesses data from RLDS structure."""
    
    def __init__(self, 
                 data_dir: str,
                 batch_size: int = 32,
                 image_size: Tuple[int, int] = (256, 256),
                 sequence_length: int = 10,
                 max_episodes: int = 10,
                 augment: bool = False,
                 val_split: float = 0.2,
                 random_seed: int = 42):
        """
        Initialize SavorDataProcessor.
        
        Args:
            val_split: Fraction of data to use for validation (0.0 to 1.0)
            random_seed: Random seed for reproducible splits
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.max_episodes = max_episodes
        self.augment = augment
        self.val_split = val_split
        self.random_seed = random_seed
        
        # Load and preprocess data directly
        self.data = self._load_and_preprocess_data()
        
        # Create train/val split indices
        self._create_split_indices()
    
    def _create_split_indices(self):
        """Create train/val split indices for reproducible splits."""
        import random
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        # Create list of indices
        indices = list(range(len(self.data)))
        random.shuffle(indices)
        
        # Calculate split point - ensure at least 1 sample for validation
        val_size = max(1, int(len(indices) * self.val_split)) if len(indices) > 1 else 0
        
        # Split indices
        self.val_indices = set(indices[:val_size])
        self.train_indices = set(indices[val_size:])
        
        print(f"Created train/val split: {len(self.train_indices)} train, {len(self.val_indices)} val")
    
    def _load_and_preprocess_data(self) -> List[Dict]:
        """Load and preprocess data directly into memory."""
        # print(f"Loading SAVOR dataset from {self.data_dir}")
        
        # Try to load as RLDS dataset first
        try:
            ds = tfds.load('savor_rlds', data_dir=self.data_dir, split='train')
            print("Loaded as RLDS dataset")
        except:
            # If RLDS fails, try to load the raw data directly
            print("RLDS dataset not found, trying to load raw data...")
            return self._load_raw_data()
        
        # print("Processing episodes...")
        processed_data = []
        episode_count = 0
        
        for episode in ds:
            if episode_count >= self.max_episodes:
                break
                
            try:
                # Extract steps from episode
                steps = episode['steps']
                steps_list = list(steps)
                
                if len(steps_list) < self.sequence_length:
                    # Skip episodes that are too short
                    print(f"Skipping episode {episode_count} - too short ({len(steps_list)} steps)")
                    episode_count += 1
                    continue
                
                # Process the episode
                episode_data = self._process_episode_direct(episode, steps_list)
                if episode_data is not None:
                    processed_data.append(episode_data)
                    # print(f"Processed episode {episode_count} with {len(steps_list)} steps")
                
                episode_count += 1
                
            except Exception as e:
                print(f"Error processing episode {episode_count}: {e}")
                episode_count += 1
                continue
        
        # print(f"Successfully processed {len(processed_data)} episodes")
        return processed_data
    
    def _load_raw_data(self) -> List[Dict]:
        """Load raw data directly from the data directory structure."""
        print("Loading raw data from directory structure...")
        # This would need to be implemented based on your actual data structure
        # For now, return empty list
        return []
    
    def _process_episode_direct(self, episode: Dict, steps_list: List) -> Dict:
        """Process episode data directly without TensorFlow operations."""
        try:
            # Extract data from steps
            rgb_images = []
            depth_images = []
            poses = []
            force_torques = []
            physical_properties = []
            
            for step in steps_list[:self.sequence_length]:  # Take only first sequence_length steps
                if step is None:
                    continue
                    
                # Extract observation data
                obs = step['observation']
                
                # Process RGB image
                rgb_img = obs['rgb'].numpy()
                if rgb_img.dtype != np.uint8:
                    rgb_img = rgb_img.astype(np.uint8)
                rgb_img = cv2.resize(rgb_img, self.image_size)
                rgb_images.append(rgb_img)
                
                # Process depth image
                depth_img = obs['depth'].numpy()
                if depth_img.dtype != np.uint8:
                    depth_img = depth_img.astype(np.uint8)
                depth_img = cv2.resize(depth_img, self.image_size)
                depth_images.append(depth_img)
                
                # Extract pose and force data
                poses.append(obs['pose'].numpy())
                force_torques.append(obs['force_torque'].numpy())
                physical_properties.append(step['physical_properties'].numpy())
            
            # Convert to numpy arrays
            rgb_sequence = np.stack(rgb_images)
            depth_sequence = np.stack(depth_images)
            pose_sequence = np.stack(poses)
            force_sequence = np.stack(force_torques)
            physical_sequence = np.stack(physical_properties)
            
            # Check if physical properties are all zeros and warn
            if np.all(physical_sequence == 0):
                print(f"[WARNING]: Physical properties are all zeros!")
            
            return {
                'rgb_sequence': rgb_sequence,
                'depth_sequence': depth_sequence,
                'pose_sequence': pose_sequence,
                'force_sequence': force_sequence,
                'physical_properties': physical_sequence,
                'episode_metadata': episode['episode_metadata']
            }
            
        except Exception as e:
            print(f"Error processing episode: {e}")
            return None
    
    def _apply_augmentation(self, data: Dict) -> Dict:
        """Apply data augmentation to the input data."""
        if not self.augment:
            return data
        
        rgb_sequence = data['rgb_sequence']
        depth_sequence = data['depth_sequence']
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            rgb_sequence = np.flip(rgb_sequence, axis=2)  # Flip width dimension
            depth_sequence = np.flip(depth_sequence, axis=2)
        
        # Random brightness adjustment for RGB
        if np.random.random() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            rgb_sequence = np.clip(rgb_sequence * brightness_factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment for RGB
        if np.random.random() > 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean = np.mean(rgb_sequence)
            rgb_sequence = np.clip((rgb_sequence - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        return {
            'rgb_sequence': rgb_sequence,
            'depth_sequence': depth_sequence,
            'pose_sequence': data['pose_sequence'],
            'force_sequence': data['force_sequence'],
            'physical_properties': data['physical_properties'],
            'episode_metadata': data['episode_metadata']
        }
    
    def get_data(self, split: str = 'all') -> List[Dict]:
        """Get data with optional train/val split.
        
        Args:
            split: 'all', 'train', or 'val'
        """
        if split == 'all':
            data = self.data
        elif split == 'train':
            # Use train indices
            data = [self.data[i] for i in self.train_indices]
        elif split == 'val':
            # Use val indices
            data = [self.data[i] for i in self.val_indices]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'all', 'train', or 'val'")
        
        if self.augment and split == 'train':
            print("Applying data augmentation...")
            augmented_data = []
            for item in data:
                # Apply augmentation
                augmented_item = self._apply_augmentation(item)
                augmented_data.append(augmented_item)
            return augmented_data
        else:
            return data
    
    def get_sample_batch(self, num_samples: int = 1) -> Dict:
        """Get a sample batch for testing."""
        if len(self.data) == 0:
            return None
            
        # Take first num_samples
        sample_data = self.data[:num_samples]
        
        # Convert to batch format
        batch = {
            'rgb_sequence': np.stack([d['rgb_sequence'] for d in sample_data]),
            'depth_sequence': np.stack([d['depth_sequence'] for d in sample_data]),
            'pose_sequence': np.stack([d['pose_sequence'] for d in sample_data]),
            'force_sequence': np.stack([d['force_sequence'] for d in sample_data]),
            'physical_properties': np.stack([d['physical_properties'] for d in sample_data]),
            'episode_metadata': [d['episode_metadata'] for d in sample_data]
        }
        
        return batch


def rlds_dataset_processor(data_dir: str, 
                          batch_size: int = 32,
                          image_size: Tuple[int, int] = (256, 256),
                          sequence_length: int = 10,
                          max_episodes: int = 10,
                          augment: bool = False,
                          val_split: float = 0.2,
                          random_seed: int = 42) -> SavorDataProcessor:
    """
    Factory function to create the SavorDataProcessor.
    """
    return SavorDataProcessor(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        sequence_length=sequence_length,
        max_episodes=max_episodes,
        augment=augment,
        val_split=val_split,
        random_seed=random_seed
    )


def test_processor(data_dir: str):
    """Test function to verify the SavorDataProcessor works correctly."""
    print("Testing SavorDataProcessor...")
    
    # Create processor with limited episodes
    processor = SavorDataProcessor(
        data_dir=data_dir,
        batch_size=2,
        sequence_length=5,
        max_episodes=3,  # Only process 3 episodes for testing
        augment=False
    )
    
    # Get training data
    train_data = processor.get_data()
    print(f"Loaded {len(train_data)} training samples")
    
    # Test sample batch
    sample_batch = processor.get_sample_batch(1)
    if sample_batch is not None:
        print(f"Sample batch:")
        print(f"  RGB sequence shape: {sample_batch['rgb_sequence'].shape}")
        print(f"  Depth sequence shape: {sample_batch['depth_sequence'].shape}")
        print(f"  Pose sequence shape: {sample_batch['pose_sequence'].shape}")
        print(f"  Force sequence shape: {sample_batch['force_sequence'].shape}")
        print(f"  Physical properties shape: {sample_batch['physical_properties'].shape}")
        print(f"  Episode metadata: {sample_batch['episode_metadata']}")
    else:
        print("No data available")
    
    print("Data processor test completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SavorDataProcessor")
    parser.add_argument("--data_dir", required=True, help="Directory containing the RLDS dataset")
    
    args = parser.parse_args()
    test_processor(args.data_dir)
