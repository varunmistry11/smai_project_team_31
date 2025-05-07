# transformer/model/mvit_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion_strategies import SimpleAveraging, MaxPoolFusion, AttentionFusion

class MultiViewTransformer(nn.Module):
    """Multi-View Transformer model for foul classification"""
    
    def __init__(self, num_frames=16, num_classes=10, use_replay_speed=True):
        super(MultiViewTransformer, self).__init__()
        
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.use_replay_speed = use_replay_speed
        self.feature_dim = 768  # MViT-v2-S output dimension
        
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3)),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            # Additional layers would go here
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Fusion strategies
        self.fusion_strategies = {
            'simple_avg': SimpleAveraging(),
            'max_pooling': MaxPoolFusion(),
            'attention': AttentionFusion(self.feature_dim)
        }
        
        # Default fusion strategy
        self.default_fusion = 'attention'
        
        # Lifting network (optional MLP for feature transformation)
        self.lifting_network = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification heads
        self.body_part_classifier = nn.Linear(self.feature_dim, 2)
        
        # Add replay speed feature dimension if used
        upper_input_dim = self.feature_dim + 1 if self.use_replay_speed else self.feature_dim
        lower_input_dim = self.feature_dim + 1 if self.use_replay_speed else self.feature_dim
        severity_input_dim = self.feature_dim + 1 if self.use_replay_speed else self.feature_dim
        
        # Upper body classifier (5 classes)
        self.upper_body_classifier = nn.Sequential(
            nn.Linear(upper_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )
        
        # Lower body classifier (5 classes)
        self.lower_body_classifier = nn.Sequential(
            nn.Linear(lower_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )
        
        # Severity classifier (4 classes: no card, foul, yellow, red)
        self.severity_classifier = nn.Sequential(
            nn.Linear(severity_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
    
    def forward(self, x, replay_speed=None, fusion_strategy=None):
        """
        Forward pass of the model
        
        Args:
            x: Input tensor of shape [batch_size, num_views, channels, frames, height, width]
            replay_speed: Optional tensor of shape [batch_size, 1] containing replay speed
            fusion_strategy: Which fusion strategy to use (None = use default)
        
        Returns:
            Tuple of (body_part_logits, upper_body_logits, lower_body_logits, severity_logits)
        """
        batch_size, num_views = x.shape[:2]
        
        # Process each view independently
        # Reshape to [batch_size * num_views, channels, frames, height, width]
        x_flat = x.view(-1, *x.shape[2:])
        
        # Apply MViT backbone to extract features
        features = self.backbone(x_flat)
        
        # Reshape features to [batch_size, num_views, feature_dim]
        features = features.view(batch_size, num_views, -1)
        
        # Apply fusion strategy
        if fusion_strategy is None:
            fusion_strategy = self.default_fusion
        
        # Use selected fusion strategy
        if fusion_strategy == 'attention':
            fused_features, attention_weights = self.fusion_strategies[fusion_strategy](features)
            # Store attention weights for visualization
            self.last_attention_weights = attention_weights
        else:
            fused_features = self.fusion_strategies[fusion_strategy](features)
        
        # Apply lifting network
        lifted_features = self.lifting_network(fused_features)
        
        # Body part classification
        body_part_logits = self.body_part_classifier(lifted_features)
        
        # Concatenate replay speed if used
        if self.use_replay_speed and replay_speed is not None:
            features_with_speed = torch.cat([lifted_features, replay_speed], dim=1)
        else:
            features_with_speed = lifted_features
        
        # Foul type classification (specialized for upper/lower body)
        upper_body_logits = self.upper_body_classifier(features_with_speed)
        lower_body_logits = self.lower_body_classifier(features_with_speed)
        
        # Severity classification
        severity_logits = self.severity_classifier(features_with_speed)
        
        return body_part_logits, upper_body_logits, lower_body_logits, severity_logits