import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAveraging(nn.Module):
    """Simple averaging fusion strategy"""
    
    def __init__(self):
        super(SimpleAveraging, self).__init__()
    
    def forward(self, view_features):
        """
        Args:
            view_features: Tensor of shape [batch_size, num_views, feature_dim]
        
        Returns:
            Tensor of shape [batch_size, feature_dim]
        """
        # Simple averaging of features across views
        return torch.mean(view_features, dim=1)

class MaxPoolFusion(nn.Module):
    """Max pooling fusion strategy"""
    
    def __init__(self):
        super(MaxPoolFusion, self).__init__()
    
    def forward(self, view_features):
        """
        Args:
            view_features: Tensor of shape [batch_size, num_views, feature_dim]
        
        Returns:
            Tensor of shape [batch_size, feature_dim]
        """
        # Max pooling of features across views
        return torch.max(view_features, dim=1)[0]

class AttentionFusion(nn.Module):
    """Attention-based fusion strategy"""
    
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def forward(self, view_features):
        """
        Args:
            view_features: Tensor of shape [batch_size, num_views, feature_dim]
        
        Returns:
            Tensor of shape [batch_size, feature_dim]
        """
        # Calculate attention weights for each view
        batch_size, num_views, feature_dim = view_features.shape
        
        # Reshape for attention computation
        flat_views = view_features.reshape(-1, feature_dim)
        
        # Calculate attention scores
        attention_scores = self.attention(flat_views).view(batch_size, num_views)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights to features
        # Reshape weights for broadcasting: [batch_size, num_views, 1]
        attention_weights = attention_weights.unsqueeze(-1)
        
        # Weighted sum of features
        fused_features = torch.sum(view_features * attention_weights, dim=1)
        
        return fused_features, attention_weights.squeeze(-1)