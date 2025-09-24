"""
Neural network model (SAVOR-Net) for SAVOR: Skill Affordance Learning from Visuo-Haptic Perception 
for Robot-Assisted Bite Acquisition
"""

import torch
import torch.nn as nn
import torchvision


class SAVORNet(nn.Module):    
    def __init__(self, seq_length=40, feature_dim=128, lstm_hidden_dim=512):
        super(SAVORNet, self).__init__()
        
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # RGB encoder: Pre-trained ResNet50 + 2-layer MLP
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.rgb_encoder = nn.Sequential(
            *list(resnet50.children())[:-1],  # Remove final FC layer
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Depth encoder: 4-layer CNN + 2-layer MLP
        self.depth_encoder = nn.Sequential(
            # 4-layer CNN with 3x3 kernels and LeakyReLU
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # 2-layer MLP
            nn.Linear(256, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Force encoder: 2-layer MLP (6D force data)
        self.force_encoder = nn.Sequential(
            nn.Linear(6, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Pose encoder: 2-layer MLP (6D pose data)
        self.pose_encoder = nn.Sequential(
            nn.Linear(6, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # LSTM for temporal fusion: 2 layers, hidden size 512
        self.lstm = nn.LSTM(
            input_size=feature_dim * 4,  # 128 features each
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Final 3-layer MLP for prediction
        self.final_mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3 * 5)  # 3 attributes * 5 classes each (scores 1-5)
        )

    def forward(self, rgb_images, depth_images, force_data, pose_data):
        """
        Args:
            rgb_images: RGB images [batch_size, seq_length, 3, H, W]
            depth_images: Depth images [batch_size, seq_length, 1, H, W]
            force_data: Force data [batch_size, seq_length, 6]
            pose_data: Pose data [batch_size, seq_length, 6]
            
        Returns:
            predictions: [batch_size, seq_length, 3, 5] - for each attribute and class
        """
        batch_size, seq_length = rgb_images.size(0), rgb_images.size(1)
        
        # Process RGB images
        rgb_flat = rgb_images.view(-1, 3, rgb_images.size(-2), rgb_images.size(-1))
        rgb_features = self.rgb_encoder(rgb_flat)  # [batch*seq, 128]
        rgb_features = rgb_features.view(batch_size, seq_length, -1)  # [batch, seq, 128]
        
        # Process depth images
        depth_flat = depth_images.view(-1, 1, depth_images.size(-2), depth_images.size(-1))
        depth_features = self.depth_encoder(depth_flat)  # [batch*seq, 128]
        depth_features = depth_features.view(batch_size, seq_length, -1)  # [batch, seq, 128]
        
        # Process force data
        force_features = self.force_encoder(force_data)  # [batch, seq, 128]
        
        # Process pose data
        pose_features = self.pose_encoder(pose_data)  # [batch, seq, 128]
        
        # Concatenate all features
        combined_features = torch.cat([
            rgb_features, depth_features, force_features, pose_features
        ], dim=-1)  # [batch, seq, 512]
        
        # Pass through LSTM
        lstm_output, _ = self.lstm(combined_features)  # [batch, seq, 512]
        
        # Final prediction
        predictions = self.final_mlp(lstm_output)  # [batch, seq, 27]
        predictions = predictions.view(batch_size, seq_length, 3, 5)  # [batch, seq, 3, 5]
        return predictions
    
    def predict_with_confidence(self, rgb_images, depth_images, force_data, pose_data):
        """
        Args:
            rgb_images: RGB images [batch_size, seq_length, 3, H, W]
            depth_images: Depth images [batch_size, seq_length, 1, H, W]
            force_data: Force data [batch_size, seq_length, 6]
            pose_data: Pose data [batch_size, seq_length, 6]
            
        Returns:
            Dictionary containing:
            - 'predictions': Raw logits [batch_size, seq_length, 3, 5]
            - 'probabilities': Softmax probabilities [batch_size, seq_length, 3, 5]
            - 'confidence_scores': Max confidence for each prediction [batch_size, seq_length, 3]
            - 'predicted_classes': Predicted class indices [batch_size, seq_length, 3]
            - 'entropy': Prediction entropy [batch_size, seq_length, 3]
        """
        import torch.nn.functional as F
        
        # Get raw predictions
        predictions = self.forward(rgb_images, depth_images, force_data, pose_data)
        
        # Calculate probabilities
        probabilities = F.softmax(predictions, dim=-1)
        
        # Get confidence scores and predicted classes
        confidence_scores, predicted_classes = torch.max(probabilities, dim=-1)
        
        # Calculate entropy (uncertainty measure)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_scores': confidence_scores,
            'predicted_classes': predicted_classes,
            'entropy': entropy
        }


def save_checkpoint(epoch, model, optimizer, loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_checkpoint(path, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('loss', 0.0)