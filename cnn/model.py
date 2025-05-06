"""
Soccer Foul Classification Model
This module implements the MVFoulClassifier model for soccer foul classification
using the SoccerNet-MVFouls dataset.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Configuration
class Config:
    def __init__(self):
        self.base_path = Path("../soccernet/fouls/mvfouls")
        #self.batch_size = 8
        self.batch_size = 2
        self.num_epochs = 5
        self.learning_rate = 3e-4
        self.weight_decay = 1e-5
        self.num_frames = 16  # Number of frames to sample from each video
        #self.frame_height = 224
        #self.frame_width = 224
        self.frame_height = 160
        self.frame_width = 160
        self.max_clips = 4  # Maximum number of clips per action
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define the classes based on dataset
        self.action_classes = [
            "Challenge",      # Upper body
            "Push",           # Upper body 
            "Hold",           # Upper body
            "Elbow",          # Upper body
            "Charge",         # Upper body
            "Tackle",         # Lower body
            "Standing tackle", # Lower body
            "High leg",       # Lower body
            "Kick",           # Lower body
            "Step"            # Lower body
        ]
        self.severity_classes = ["No foul", "Foul", "Yellow card", "Red card"]  # 0.0, 1.0, 2.0, 3.0
        self.body_part_classes = ["Upper body", "Lower body"]
        
        # Video extension - update based on your dataset
        self.video_extension = ".mp4"  # Changed from .avi to .mp4

# Data augmentation and preprocessing
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class SoccerNetMVFoulsDataset(Dataset):
    def __init__(self, base_path, split='train', transform=None, config=None):
        """
        Dataset for SoccerNet-MVFouls
        
        Args:
            base_path: Path to the dataset directory
            split: 'train', 'valid', or 'test'
            transform: Optional transforms to apply to frames
            config: Configuration object
        """
        if config is None:
            config = Config()
            
        self.base_path = Path(base_path) / split
        self.split = split
        self.transform = transform if transform is not None else get_transforms(split == 'train')
        self.config = config
        
        # Load annotations
        with open(self.base_path / "annotations.json", "r") as f:
            annotations = json.load(f)
        
        self.actions = annotations["Actions"]
        self.action_ids = list(self.actions.keys())
        
        # Create mapping from text labels to indices
        self.action_class_to_idx = {cls: i for i, cls in enumerate(config.action_classes)}
        self.severity_to_idx = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3}  # Map severity scores to indices
        self.body_part_to_idx = {cls: i for i, cls in enumerate(config.body_part_classes)}
        
        print(f"Loaded {len(self.action_ids)} actions for {split}")
    
    def __len__(self):
        return len(self.action_ids)
    
    def __getitem__(self, idx):
        action_id = self.action_ids[idx]
        action = self.actions[action_id]
        
        # Extract labels
        action_class = action["Action class"]
        # Handle empty or invalid severity values
        try:
            severity = float(action["Severity"]) if action["Severity"] else 0.0
        except (ValueError, TypeError):
            severity = 0.0  # Default to "No foul" if severity is invalid
            
        body_part = action["Bodypart"]
        
        if body_part == "Under body":  # Correct the label to match the class name
            body_part = "Lower body"
        
        # Convert labels to indices (with error handling)
        action_label = self.action_class_to_idx.get(action_class, 0)  # Default to 0 if not found
        severity_label = self.severity_to_idx.get(severity, 0)  # Default to 0 if not found
        body_part_label = self.body_part_to_idx.get(body_part, 0)  # Default to 0 if not found
        
        # Get all available clips for this action
        clips_data = action["Clips"]
        num_clips = len(clips_data)
        
        # Load frames from each clip (up to max_clips)
        all_clip_frames = []
        all_replay_speeds = []
        
        for i in range(min(num_clips, self.config.max_clips)):
            clip_info = clips_data[i]
            # Create proper path and handle URL paths
            clip_path = clip_info["Url"]
            # Clean up URL if needed
            if "Dataset/Train/" in clip_path:
                clip_path = clip_path.replace("Dataset/Train/", "")
            elif "Dataset/Valid/" in clip_path:
                clip_path = clip_path.replace("Dataset/Valid/", "")
            elif "Dataset/Test/" in clip_path:
                clip_path = clip_path.replace("Dataset/Test/", "")
            
            clip_path = self.base_path / clip_path
            
            # Handle replay speed (with error handling)
            try:
                replay_speed = float(clip_info["Replay speed"]) if clip_info["Replay speed"] else 1.0
            except (ValueError, TypeError):
                replay_speed = 1.0  # Default to normal speed if invalid
            
            # Try different video extensions if needed
            frames = self._sample_frames_from_video(clip_path)
            
            if frames is not None:
                all_clip_frames.append(frames)
                all_replay_speeds.append(replay_speed)
        
        # If we couldn't load any clips, create dummy data
        if len(all_clip_frames) == 0:
            # Generate random dummy data for debugging
            dummy_frames = torch.randn(self.config.num_frames, 3, self.config.frame_height, self.config.frame_width)
            dummy_frames = dummy_frames / dummy_frames.norm(dim=(1, 2, 3), keepdim=True)  # Normalize
            all_clip_frames.append(dummy_frames)
            all_replay_speeds.append(1.0)
            print(f"Warning: Using dummy data for action {action_id} because no clips could be loaded")
            
        # Pad if we have fewer than max_clips
        while len(all_clip_frames) < self.config.max_clips:
            all_clip_frames.append(all_clip_frames[0])  # Duplicate the first clip
            all_replay_speeds.append(all_replay_speeds[0])
        
        # Stack all clips
        clips_tensor = torch.stack(all_clip_frames)  # Shape: [max_clips, num_frames, C, H, W]
        replay_speeds = torch.tensor(all_replay_speeds, dtype=torch.float32)
        
        # Return the stacked clips and labels
        return {
            'clips': clips_tensor,
            'replay_speeds': replay_speeds,
            'action_label': torch.tensor(action_label, dtype=torch.long),
            'severity_label': torch.tensor(severity_label, dtype=torch.long),
            'body_part_label': torch.tensor(body_part_label, dtype=torch.long),
            'num_clips': torch.tensor(min(num_clips, self.config.max_clips), dtype=torch.long)
        }
    
    def _sample_frames_from_video(self, video_path):
        """Sample frames from a video, focusing on areas with high motion intensity."""
        # Try multiple extensions if the file doesn't exist
        video_extensions = [self.config.video_extension, ".avi", ".mp4", ".mkv"]
        
        for ext in video_extensions:
            full_path = str(video_path) + ext
            if os.path.exists(full_path):
                try:
                    cap = cv2.VideoCapture(full_path)
                    if not cap.isOpened():
                        continue  # Try next extension
                    
                    # Get video properties
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if frame_count <= 0:
                        cap.release()
                        continue  # Try next extension
                    
                    # Strategy: Sample frames focusing on the first 4 seconds (peak action from your analysis)
                    # Assuming 25 fps, 4 seconds = 100 frames
                    target_frames = min(100, frame_count)
                    
                    # Calculate the frame indices to sample
                    if target_frames <= self.config.num_frames:
                        # If we have fewer frames than needed, sample all and repeat
                        frame_indices = list(range(target_frames))
                        while len(frame_indices) < self.config.num_frames:
                            frame_indices.append(frame_indices[-1] if frame_indices else 0)
                    else:
                        # Sample evenly from the target range
                        frame_indices = np.linspace(0, target_frames-1, self.config.num_frames, dtype=int)
                    
                    # Extract the frames
                    frames = []
                    for idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Convert from BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Apply transform if available
                        if self.transform:
                            frame = self.transform(frame)
                        else:
                            frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
                        
                        frames.append(frame)
                    
                    cap.release()
                    
                    # If we couldn't get all frames, pad with the last frame
                    if frames:
                        while len(frames) < self.config.num_frames:
                            frames.append(frames[-1])
                        
                        return torch.stack(frames)
                    
                except Exception as e:
                    print(f"Error processing video {full_path}: {str(e)}")
        
        # If we reach here, we couldn't load the video with any extension
        # Instead of printing an error which would clutter the output, just return None
        return None


# Model Architecture - Hierarchical Multi-View Network
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, 
            stride=(1, stride, stride), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class MVFoulClassifier(nn.Module):
    def __init__(self, num_classes=10, num_severity_classes=4):
        super(MVFoulClassifier, self).__init__()
        
        # Base CNN for feature extraction (using ResNet-like blocks)
        self.base_model = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # ResNet-like blocks
            self._make_layer(64, 128, stride=2),
            self._make_layer(128, 256, stride=2),
            self._make_layer(256, 512, stride=2),
            
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        
        # Feature dimension after the base model
        self.feature_dim = 512
        
        # First stage: Body part classifier (upper vs lower body)
        self.body_part_classifier = nn.Linear(self.feature_dim + 1, 2)  # +1 for replay speed
        
        # Second stage: Specific classifiers for each body part
        self.upper_body_classifier = nn.Linear(self.feature_dim + 1, num_classes // 2)
        self.lower_body_classifier = nn.Linear(self.feature_dim + 1, num_classes // 2)
        
        # Severity classifier (shared)
        self.severity_classifier = nn.Linear(self.feature_dim + 1, num_severity_classes)
        
        # Multi-view fusion module
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def _make_layer(self, in_channels, out_channels, stride=1):
        """Create a ResNet-like block."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride=stride, downsample=downsample))
        layers.append(ResNetBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, replay_speeds, num_clips):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, max_clips, num_frames, C, H, W]
            replay_speeds: Tensor of shape [batch_size, max_clips]
            num_clips: Tensor of shape [batch_size] indicating the actual number of clips per sample
        
        Returns:
            Dictionary containing predictions for action class, severity, and body part
        """
        batch_size, max_clips, num_frames, C, H, W = x.size()
        
        # Process each clip independently
        all_features = []
        for clip_idx in range(max_clips):
            # Get current clip
            clip = x[:, clip_idx]  # [batch_size, num_frames, C, H, W]
            
            # Reshape for 3D CNN: [batch_size, C, num_frames, H, W]
            clip = clip.permute(0, 2, 1, 3, 4)
            
            # Extract features
            features = self.base_model(clip)  # [batch_size, feature_dim, 1, 1, 1]
            features = features.view(batch_size, self.feature_dim)  # [batch_size, feature_dim]
            
            all_features.append(features)
        
        # Stack features from all clips
        stacked_features = torch.stack(all_features, dim=1)  # [batch_size, max_clips, feature_dim]
        
        # Compute attention weights for multi-view fusion
        attention_weights = self.attention(stacked_features.view(-1, self.feature_dim))
        attention_weights = attention_weights.view(batch_size, max_clips)
        
        # Create a mask for valid clips
        clip_mask = torch.arange(max_clips, device=x.device).expand(batch_size, max_clips) < num_clips.unsqueeze(1)
        
        # Apply mask to attention weights (set weights for non-existent clips to -inf)
        attention_weights = attention_weights.masked_fill(~clip_mask, float('-inf'))
        
        # Softmax to get normalized weights
        attention_weights = F.softmax(attention_weights, dim=1).unsqueeze(2)  # [batch_size, max_clips, 1]
        
        # Weighted sum of features
        fused_features = torch.sum(stacked_features * attention_weights, dim=1)  # [batch_size, feature_dim]
        
        # Average the replay speeds for valid clips
        valid_replay_speeds = replay_speeds * clip_mask.float()
        avg_replay_speed = torch.sum(valid_replay_speeds, dim=1) / torch.clamp(num_clips.float(), min=1.0)
        avg_replay_speed = avg_replay_speed.unsqueeze(1)  # [batch_size, 1]
        
        # Concatenate features with replay speed
        features_with_speed = torch.cat([fused_features, avg_replay_speed], dim=1)
        
        # First stage: Body part classification
        body_part_logits = self.body_part_classifier(features_with_speed)
        
        # Second stage: Conditional action classification based on body part
        upper_body_logits = self.upper_body_classifier(features_with_speed)
        lower_body_logits = self.lower_body_classifier(features_with_speed)
        
        # Severity classification
        severity_logits = self.severity_classifier(features_with_speed)
        
        return {
            'body_part': body_part_logits,
            'upper_body': upper_body_logits,
            'lower_body': lower_body_logits,
            'severity': severity_logits
        }


# Loss function with class weighting to handle imbalance
class HierarchicalFoulLoss(nn.Module):
    def __init__(self, action_weights=None, severity_weights=None):
        super(HierarchicalFoulLoss, self).__init__()
        self.action_weights = action_weights
        self.severity_weights = severity_weights
        
        # Create individual loss functions
        self.body_part_loss = nn.CrossEntropyLoss()
        self.upper_body_loss = nn.CrossEntropyLoss(weight=action_weights[:5] if action_weights is not None else None)
        self.lower_body_loss = nn.CrossEntropyLoss(weight=action_weights[5:] if action_weights is not None else None)
        self.severity_loss = nn.CrossEntropyLoss(weight=severity_weights)
    
    def forward(self, outputs, targets):
        """
        Compute hierarchical loss
        
        Args:
            outputs: Dictionary containing model outputs
            targets: Dictionary containing ground truth labels
        
        Returns:
            Total loss value
        """
        # Extract predictions and targets
        body_part_pred = outputs['body_part']
        upper_body_pred = outputs['upper_body']
        lower_body_pred = outputs['lower_body']
        severity_pred = outputs['severity']
        
        body_part_target = targets['body_part_label']
        action_target = targets['action_label']
        severity_target = targets['severity_label']
        
        # Separate upper and lower body action targets
        # Assuming first half of classes are upper body and second half are lower body
        upper_body_mask = body_part_target == 0
        lower_body_mask = body_part_target == 1
        
        # Handle empty masks
        if upper_body_mask.sum() > 0:
            upper_body_action_target = action_target[upper_body_mask]
            # Ensure indices are in the right range (0-4)
            upper_body_action_target = torch.clamp(upper_body_action_target, 0, 4)
        else:
            # Create a dummy target if no upper body samples in this batch
            upper_body_action_target = torch.zeros(1, dtype=torch.long, device=body_part_target.device)
        
        if lower_body_mask.sum() > 0:
            lower_body_action_target = action_target[lower_body_mask]
            # Map to range 0-4 by subtracting 5 (since lower body classes start at index 5)
            lower_body_action_target = torch.clamp(lower_body_action_target - 5, 0, 4)
        else:
            # Create a dummy target if no lower body samples in this batch
            lower_body_action_target = torch.zeros(1, dtype=torch.long, device=body_part_target.device)
        
        # Body part loss
        bp_loss = self.body_part_loss(body_part_pred, body_part_target)
        
        # Upper body action loss (only for upper body samples)
        ub_loss = 0
        if upper_body_mask.sum() > 0:
            ub_loss = self.upper_body_loss(upper_body_pred[upper_body_mask], upper_body_action_target)
        
        # Lower body action loss (only for lower body samples)
        lb_loss = 0
        if lower_body_mask.sum() > 0:
            lb_loss = self.lower_body_loss(lower_body_pred[lower_body_mask], lower_body_action_target)
        
        # Severity loss
        sev_loss = self.severity_loss(severity_pred, severity_target)
        
        # Combine losses with weights
        total_loss = bp_loss + 0.5 * (ub_loss + lb_loss) + sev_loss
        
        return total_loss


# Training function
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the model and validate."""
    best_val_f1 = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Move data to device
            clips = batch['clips'].to(device)
            replay_speeds = batch['replay_speeds'].to(device)
            num_clips = batch['num_clips'].to(device)
            action_label = batch['action_label'].to(device)
            severity_label = batch['severity_label'].to(device)
            body_part_label = batch['body_part_label'].to(device)
            
            # Forward pass
            outputs = model(clips, replay_speeds, num_clips)
            
            # Compute loss
            loss = criterion(outputs, {
                'action_label': action_label,
                'severity_label': severity_label,
                'body_part_label': body_part_label
            })
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'train_loss': loss.item()})
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['action_f1'])
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Body Part F1: {val_metrics['body_part_f1']:.4f}")
        print(f"Val Action F1: {val_metrics['action_f1']:.4f}")
        print(f"Val Severity F1: {val_metrics['severity_f1']:.4f}")
        
        # Save best model
        if val_metrics['action_f1'] > best_val_f1:
            best_val_f1 = val_metrics['action_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
            }, 'best_mvfoul_model.pth')
            print(f"Saved new best model with F1: {best_val_f1:.4f}")
    
    return history

def validate(model, val_loader, criterion, device):
    """Validate the model and compute metrics."""
    model.eval()
    val_loss = 0.0
    
    # Lists to store predictions and targets
    body_part_preds = []
    body_part_targets = []
    action_preds = []
    action_targets = []
    severity_preds = []
    severity_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            clips = batch['clips'].to(device)
            replay_speeds = batch['replay_speeds'].to(device)
            num_clips = batch['num_clips'].to(device)
            action_label = batch['action_label'].to(device)
            severity_label = batch['severity_label'].to(device)
            body_part_label = batch['body_part_label'].to(device)
            
            # Forward pass
            outputs = model(clips, replay_speeds, num_clips)
            
            # Compute loss
            loss = criterion(outputs, {
                'action_label': action_label,
                'severity_label': severity_label,
                'body_part_label': body_part_label
            })
            
            val_loss += loss.item()
            
            # Predictions
            body_part_pred = torch.argmax(outputs['body_part'], dim=1)
            severity_pred = torch.argmax(outputs['severity'], dim=1)
            
            # For action prediction, use the appropriate classifier based on predicted body part
            upper_body_pred = torch.argmax(outputs['upper_body'], dim=1)
            lower_body_pred = torch.argmax(outputs['lower_body'], dim=1)
            
            # Combine predictions based on body part
            action_pred = torch.zeros_like(body_part_pred)
            action_pred[body_part_pred == 0] = upper_body_pred[body_part_pred == 0]
            action_pred[body_part_pred == 1] = lower_body_pred[body_part_pred == 1] + 5  # +5 because lower body classes start at index 5
            
            # Store predictions and targets
            body_part_preds.extend(body_part_pred.cpu().numpy())
            body_part_targets.extend(body_part_label.cpu().numpy())
            action_preds.extend(action_pred.cpu().numpy())
            action_targets.extend(action_label.cpu().numpy())
            severity_preds.extend(severity_pred.cpu().numpy())
            severity_targets.extend(severity_label.cpu().numpy())
    
    val_loss /= len(val_loader)
    
    # Compute metrics
    body_part_f1 = f1_score(body_part_targets, body_part_preds, average='weighted')
    action_f1 = f1_score(action_targets, action_preds, average='weighted')
    severity_f1 = f1_score(severity_targets, severity_preds, average='weighted')
    
    metrics = {
        'body_part_f1': body_part_f1,
        'action_f1': action_f1,
        'severity_f1': severity_f1
    }
    
    return val_loss, metrics

def test_model(model, test_loader, criterion, device, config):
    """Test the trained model on the test set."""
    model.eval()
    test_loss = 0.0
    
    # Lists to store predictions and targets
    body_part_preds = []
    body_part_targets = []
    action_preds = []
    action_targets = []
    severity_preds = []
    severity_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move data to device
            clips = batch['clips'].to(device)
            replay_speeds = batch['replay_speeds'].to(device)
            num_clips = batch['num_clips'].to(device)
            action_label = batch['action_label'].to(device)
            severity_label = batch['severity_label'].to(device)
            body_part_label = batch['body_part_label'].to(device)
            
            # Forward pass
            outputs = model(clips, replay_speeds, num_clips)
            
            # Compute loss
            loss = criterion(outputs, {
                'action_label': action_label,
                'severity_label': severity_label,
                'body_part_label': body_part_label
            })
            
            test_loss += loss.item()
            
            # Predictions
            body_part_pred = torch.argmax(outputs['body_part'], dim=1)
            severity_pred = torch.argmax(outputs['severity'], dim=1)
            
            # For action prediction, use the appropriate classifier based on predicted body part
            upper_body_pred = torch.argmax(outputs['upper_body'], dim=1)
            lower_body_pred = torch.argmax(outputs['lower_body'], dim=1)
            
            # Combine predictions based on body part
            action_pred = torch.zeros_like(body_part_pred)
            action_pred[body_part_pred == 0] = upper_body_pred[body_part_pred == 0]
            action_pred[body_part_pred == 1] = lower_body_pred[body_part_pred == 1] + 5
            
            # Store predictions and targets
            body_part_preds.extend(body_part_pred.cpu().numpy())
            body_part_targets.extend(body_part_label.cpu().numpy())
            action_preds.extend(action_pred.cpu().numpy())
            action_targets.extend(action_label.cpu().numpy())
            severity_preds.extend(severity_pred.cpu().numpy())
            severity_targets.extend(severity_label.cpu().numpy())
    
    test_loss /= len(test_loader)
    
    # Compute metrics
    body_part_f1 = f1_score(body_part_targets, body_part_preds, average='weighted')
    action_f1 = f1_score(action_targets, action_preds, average='weighted')
    severity_f1 = f1_score(severity_targets, severity_preds, average='weighted')
    
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Body Part F1: {body_part_f1:.4f}")
    print(f"Action F1: {action_f1:.4f}")
    print(f"Severity F1: {severity_f1:.4f}")
    
    # Generate classification reports
    print("\nBody Part Classification Report:")
    bp_report = classification_report(
        body_part_targets, 
        body_part_preds, 
        target_names=config.body_part_classes,
        digits=4
    )
    print(bp_report)
    
    print("\nAction Classification Report:")
    action_report = classification_report(
        action_targets, 
        action_preds, 
        target_names=config.action_classes,
        digits=4
    )
    print(action_report)
    
    print("\nSeverity Classification Report:")
    severity_report = classification_report(
        severity_targets, 
        severity_preds, 
        target_names=config.severity_classes,
        digits=4
    )
    print(severity_report)
    
    # Plot confusion matrices
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(body_part_targets, body_part_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.body_part_classes, 
                yticklabels=config.body_part_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Body Part Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig('body_part_confusion_matrix.png')
    plt.close()
    
    plt.figure(figsize=(15, 12))
    cm = confusion_matrix(action_targets, action_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.action_classes, 
                yticklabels=config.action_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Action Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig('action_confusion_matrix.png')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(severity_targets, severity_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.severity_classes, 
                yticklabels=config.severity_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Severity Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig('severity_confusion_matrix.png')
    plt.close()
    
    # Return metrics
    results = {
        'test_loss': test_loss,
        'body_part_f1': body_part_f1,
        'action_f1': action_f1,
        'severity_f1': severity_f1
    }
    
    return results

def calculate_class_weights(dataset):
    """Calculate class weights inversely proportional to class frequencies."""
    action_counts = np.zeros(len(dataset.action_class_to_idx))
    severity_counts = np.zeros(len(dataset.severity_to_idx))
    body_part_counts = np.zeros(len(dataset.body_part_to_idx))
    
    print("Calculating class weights...")
    
    valid_samples = 0
    for i in tqdm(range(len(dataset))):
        try:
            sample = dataset[i]
            action_label = sample['action_label'].item()
            severity_label = sample['severity_label'].item()
            body_part_label = sample['body_part_label'].item()
            
            # Check if labels are within valid ranges
            if 0 <= action_label < len(action_counts) and 0 <= severity_label < len(severity_counts) and 0 <= body_part_label < len(body_part_counts):
                action_counts[action_label] += 1
                severity_counts[severity_label] += 1
                body_part_counts[body_part_label] += 1
                valid_samples += 1
        except Exception as e:
            # Skip problematic samples
            print(f"Error processing sample {i}: {str(e)}")
            continue
    
    print(f"Processed {valid_samples} valid samples out of {len(dataset)}")
    
    # Check if we have valid counts
    if np.sum(action_counts) == 0 or np.sum(severity_counts) == 0:
        print("Warning: No valid labels found. Using uniform weights.")
        action_weights = np.ones(len(action_counts))
        severity_weights = np.ones(len(severity_counts))
    else:
        # Convert counts to weights (inverse frequency)
        action_weights = 1.0 / np.maximum(action_counts, 1)
        severity_weights = 1.0 / np.maximum(severity_counts, 1)
        
        # Normalize weights
        action_weights = action_weights / np.sum(action_weights) * len(action_weights)
        severity_weights = severity_weights / np.sum(severity_weights) * len(severity_weights)
    
    print("\nClass weights:")
    print("Action class weights:")
    for i, (cls, weight) in enumerate(zip(dataset.config.action_classes, action_weights)):
        print(f"  {cls}: {weight:.4f} (count: {int(action_counts[i])})")
    
    print("\nSeverity class weights:")
    for i, (cls, weight) in enumerate(zip(dataset.config.severity_classes, severity_weights)):
        print(f"  {cls}: {weight:.4f} (count: {int(severity_counts[i])})")
    
    return torch.tensor(action_weights, dtype=torch.float32), torch.tensor(severity_weights, dtype=torch.float32)


def plot_training_history(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot validation F1 score
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='Val F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Validation F1 Score')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def visualize_predictions(model, dataset, config, num_samples=5):
    """Visualize model predictions on random samples."""
    model.eval()
    device = config.device
    
    # Create output directory
    os.makedirs("predictions", exist_ok=True)
    
    # Sample indices randomly
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        try:
            # Get sample
            sample = dataset[idx]
            
            # Process the sample
            clips = sample['clips'].unsqueeze(0).to(device)
            replay_speeds = sample['replay_speeds'].unsqueeze(0).to(device)
            num_clips = sample['num_clips'].unsqueeze(0).to(device)
            
            true_action = sample['action_label'].item()
            true_body_part = sample['body_part_label'].item()
            true_severity = sample['severity_label'].item()
            
            # Get model predictions
            with torch.no_grad():
                outputs = model(clips, replay_speeds, num_clips)
            
            body_part_pred = torch.argmax(outputs['body_part'], dim=1).item()
            severity_pred = torch.argmax(outputs['severity'], dim=1).item()
            
            # For action prediction, use the appropriate classifier based on predicted body part
            if body_part_pred == 0:  # Upper body
                action_pred = torch.argmax(outputs['upper_body'], dim=1).item()
            else:  # Lower body
                action_pred = torch.argmax(outputs['lower_body'], dim=1).item() + 5
            
            # Convert to class names
            true_action_name = config.action_classes[true_action]
            pred_action_name = config.action_classes[action_pred]
            
            true_body_part_name = config.body_part_classes[true_body_part]
            pred_body_part_name = config.body_part_classes[body_part_pred]
            
            true_severity_name = config.severity_classes[true_severity]
            pred_severity_name = config.severity_classes[severity_pred]
            
            # Display a sample frame from the first clip
            first_clip = clips[0, 0]  # [num_frames, C, H, W]
            middle_frame_idx = config.num_frames // 2
            frame = first_clip[middle_frame_idx].permute(1, 2, 0).cpu().numpy()
            
            # Denormalize the frame
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = std * frame + mean
            frame = np.clip(frame, 0, 1)
            
            # Plot
            plt.figure(figsize=(12, 8))
            plt.imshow(frame)
            plt.axis('off')
            
            plt.title(f"Sample #{idx}\n"
                      f"True: {true_body_part_name} - {true_action_name} - {true_severity_name}\n"
                      f"Pred: {pred_body_part_name} - {pred_action_name} - {pred_severity_name}")
            
            plt.tight_layout()
            plt.savefig(f"predictions/sample_prediction_{idx}.png")
            plt.close()
            
        except Exception as e:
            print(f"Error visualizing sample {idx}: {str(e)}")
            continue


def main():
    """Main function to run the model training and evaluation."""
    # Configuration
    config = Config()
    
    # Use relative paths to make it work with your directory structure
    config.base_path = Path("../soccernet/fouls/mvfouls")
    
    print(f"Using device: {config.device}")
    
    # Data transformations
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = SoccerNetMVFoulsDataset(
        config.base_path, 
        split='train',
        transform=train_transform,
        config=config
    )
    
    val_dataset = SoccerNetMVFoulsDataset(
        config.base_path, 
        split='valid',
        transform=val_transform,
        config=config
    )
    
    test_dataset = SoccerNetMVFoulsDataset(
        config.base_path, 
        split='test',
        transform=val_transform,
        config=config
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Calculate class weights for handling imbalance
    action_weights, severity_weights = calculate_class_weights(train_dataset)
    
    # Create model
    model = MVFoulClassifier(
        num_classes=len(config.action_classes),
        num_severity_classes=len(config.severity_classes)
    ).to(config.device)
    
    # Print model summary
    print("\nModel structure:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function and optimizer
    criterion = HierarchicalFoulLoss(
        action_weights=action_weights.to(config.device), 
        severity_weights=severity_weights.to(config.device)
    )
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Check if we have a pretrained model
    model_path = 'best_mvfoul_model.pth'
    if os.path.exists(model_path):
        print(f"\nLoading pretrained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model with validation F1: {checkpoint['val_f1']:.4f}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        with torch.no_grad():
            test_results = test_model(model, test_loader, criterion, config.device, config)
        
        # Visualize some predictions
        print("\nGenerating prediction visualizations...")
        visualize_predictions(model, test_dataset, config)
        
    else:
        # Train and validate
        print("\nStarting training...")
        history = train(model, train_loader, val_loader, criterion, optimizer, config.num_epochs, config.device)
        
        # Plot training history
        plot_training_history(history)
        
        # Load best model for testing
        print("\nLoading best model for testing...")
        checkpoint = torch.load('best_mvfoul_model.pth', map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test the model
        print("\nEvaluating on test set...")
        test_results = test_model(model, test_loader, criterion, config.device, config)
        
        # Visualize some predictions
        print("\nGenerating prediction visualizations...")
        visualize_predictions(model, test_dataset, config)
    
    print("Done!")


if __name__ == "__main__":
    main()
