"""
Utility functions for Soccer Foul Classification
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

def compute_class_weights(dataset):
    """
    Compute class weights inversely proportional to class frequencies.
    
    Args:
        dataset: SoccerNetMVFoulsDataset instance
    
    Returns:
        Tuple of (action_weights, severity_weights, body_part_weights)
    """
    action_counts = np.zeros(len(dataset.action_class_to_idx))
    severity_counts = np.zeros(len(dataset.severity_to_idx))
    body_part_counts = np.zeros(len(dataset.body_part_to_idx))
    
    for i in tqdm(range(len(dataset)), desc="Computing class weights"):
        sample = dataset[i]
        action_label = sample['action_label'].item()
        severity_label = sample['severity_label'].item()
        body_part_label = sample['body_part_label'].item()
        
        action_counts[action_label] += 1
        severity_counts[severity_label] += 1
        body_part_counts[body_part_label] += 1
    
    # Convert counts to weights (inverse frequency)
    action_weights = 1.0 / np.maximum(action_counts, 1)
    severity_weights = 1.0 / np.maximum(severity_counts, 1)
    body_part_weights = 1.0 / np.maximum(body_part_counts, 1)
    
    # Normalize weights
    action_weights = action_weights / np.sum(action_weights) * len(action_weights)
    severity_weights = severity_weights / np.sum(severity_weights) * len(severity_weights)
    body_part_weights = body_part_weights / np.sum(body_part_weights) * len(body_part_weights)
    
    print("Class weights:")
    print(f"Action: {action_weights}")
    print(f"Severity: {severity_weights}")
    print(f"Body part: {body_part_weights}")
    
    return (
        torch.from_numpy(action_weights).float(),
        torch.from_numpy(severity_weights).float(),
        torch.from_numpy(body_part_weights).float()
    )

def plot_training_history(history, output_path=None):
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary containing training history
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot validation F1 scores
    plt.subplot(2, 2, 2)
    plt.plot(history['val_body_part_f1'], label='Body Part F1')
    plt.plot(history['val_action_f1'], label='Action F1')
    plt.plot(history['val_severity_f1'], label='Severity F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Validation F1 Scores')
    
    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    
    # Plot train vs val loss on log scale to better see convergence
    plt.subplot(2, 2, 4)
    plt.semilogy(history['train_loss'], label='Train Loss')
    plt.semilogy(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.title('Training and Validation Loss (Log Scale)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(targets, predictions, class_names, filename, title):
    """
    Plot and save confusion matrix.
    
    Args:
        targets: List of true labels
        predictions: List of predicted labels
        class_names: List of class names
        filename: Path to save the plot
        title: Title for the plot
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"{title} - Normalized Confusion Matrix")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(filename)
    plt.close()
    
    # Also plot raw counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"{title} - Counts")
    plt.tight_layout()
    
    # Save figure
    filename_base, filename_ext = os.path.splitext(filename)
    plt.savefig(f"{filename_base}_counts{filename_ext}")
    plt.close()

def visualize_sample_batch(batch, config, num_samples=2):
    """
    Visualize samples from a batch.
    
    Args:
        batch: Dictionary containing batch data
        config: Configuration object
        num_samples: Number of samples to visualize
    """
    # Get samples
    clips = batch['clips'][:num_samples]  # [num_samples, max_clips, num_frames, C, H, W]
    action_labels = batch['action_label'][:num_samples]
    severity_labels = batch['severity_label'][:num_samples]
    body_part_labels = batch['body_part_label'][:num_samples]
    num_clips = batch['num_clips'][:num_samples]
    
    for i in range(num_samples):
        action_name = config.action_classes[action_labels[i]]
        severity_name = config.severity_classes[severity_labels[i]]
        body_part_name = config.body_part_classes[body_part_labels[i]]
        n_clips = num_clips[i].item()
        
        print(f"Sample {i+1}:")
        print(f"  Action: {action_name}")
        print(f"  Severity: {severity_name}")
        print(f"  Body Part: {body_part_name}")
        print(f"  Number of clips: {n_clips}")
        
        # Create figure
        plt.figure(figsize=(15, 5 * min(n_clips, 2)))
        
        # Show one frame from each clip
        for clip_idx in range(min(n_clips, 2)):  # Show at most 2 clips
            clip = clips[i, clip_idx]  # [num_frames, C, H, W]
            
            # Get middle frame
            middle_frame_idx = config.num_frames // 2
            frame = clip[middle_frame_idx].permute(1, 2, 0).cpu().numpy()
            
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = std * frame + mean
            frame = np.clip(frame, 0, 1)
            
            # Show frame
            plt.subplot(min(n_clips, 2), 1, clip_idx + 1)
            plt.imshow(frame)
            plt.title(f"Clip {clip_idx + 1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def analyze_dataset_statistics(dataset, config):
    """
    Analyze dataset statistics and visualize class distributions.
    
    Args:
        dataset: SoccerNetMVFoulsDataset instance
        config: Configuration object
    
    Returns:
        Dictionary containing dataset statistics
    """
    # Count occurrences of each class
    action_counts = np.zeros(len(config.action_classes))
    severity_counts = np.zeros(len(config.severity_classes))
    body_part_counts = np.zeros(len(config.body_part_classes))
    clip_counts = np.zeros(config.max_clips + 1)  # +1 for 0 clips (should not occur)
    
    for i in tqdm(range(len(dataset)), desc="Analyzing dataset"):
        sample = dataset[i]
        action_label = sample['action_label'].item()
        severity_label = sample['severity_label'].item()
        body_part_label = sample['body_part_label'].item()
        num_clips = sample['num_clips'].item()
        
        action_counts[action_label] += 1
        severity_counts[severity_label] += 1
        body_part_counts[body_part_label] += 1
        clip_counts[num_clips] += 1
    
    # Create visualizations
    plt.figure(figsize=(15, 15))
    
    # Action distribution
    plt.subplot(3, 1, 1)
    bars = plt.bar(range(len(config.action_classes)), action_counts)
    plt.xticks(range(len(config.action_classes)), config.action_classes, rotation=45, ha='right')
    plt.xlabel('Action Class')
    plt.ylabel('Count')
    plt.title('Action Class Distribution')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', rotation=0)
    
    # Severity distribution
    plt.subplot(3, 1, 2)
    bars = plt.bar(range(len(config.severity_classes)), severity_counts)
    plt.xticks(range(len(config.severity_classes)), config.severity_classes)
    plt.xlabel('Severity Class')
    plt.ylabel('Count')
    plt.title('Severity Class Distribution')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', rotation=0)
    
    # Body part distribution
    plt.subplot(3, 1, 3)
    bars = plt.bar(range(len(config.body_part_classes)), body_part_counts)
    plt.xticks(range(len(config.body_part_classes)), config.body_part_classes)
    plt.xlabel('Body Part')
    plt.ylabel('Count')
    plt.title('Body Part Distribution')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig('dataset_statistics.png')
    plt.close()
    
    # Number of clips distribution
    plt.figure(figsize=(10, 6))
    valid_clip_counts = clip_counts[1:]  # Remove 0 clips
    bars = plt.bar(range(1, len(valid_clip_counts) + 1), valid_clip_counts)
    plt.xticks(range(1, len(valid_clip_counts) + 1))
    plt.xlabel('Number of Clips')
    plt.ylabel('Count')
    plt.title('Number of Clips Distribution')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig('clip_distribution.png')
    plt.close()
    
    # Return statistics
    stats = {
        'action_counts': action_counts.tolist(),
        'severity_counts': severity_counts.tolist(),
        'body_part_counts': body_part_counts.tolist(),
        'clip_counts': clip_counts.tolist(),
        'total_samples': len(dataset),
        'action_class_ratio': (np.max(action_counts) / np.min(action_counts[action_counts > 0])).item(),
        'severity_class_ratio': (np.max(severity_counts) / np.min(severity_counts[severity_counts > 0])).item(),
    }
    
    print("Dataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Action class ratio (max/min): {stats['action_class_ratio']:.2f}")
    print(f"Severity class ratio (max/min): {stats['severity_class_ratio']:.2f}")
    print(f"Average clips per action: {np.mean(np.arange(len(clip_counts)) * clip_counts / np.sum(clip_counts)):.2f}")
    
    return stats