"""
Visualization tool for Soccer Foul Classification model
This script implements Grad-CAM for visualizing what regions in the input images are important for the model's decisions.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import json

# Import our model definition (assuming model.py exists in the same directory)
from model import MVFoulClassifier, Config

class GradCAM:
    """
    Implements Grad-CAM for 3D CNN models
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.register_hooks()
    
    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks on the target layer
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_category=None):
        """
        Generate a class activation map for the given input
        
        Args:
            input_tensor: Input tensor for the model (expects [1, C, T, H, W])
            target_category: Target class index for backpropagation (if None, uses highest scoring class)
            
        Returns:
            cam: Class activation map
        """
        # Forward pass
        model_output = self.model(input_tensor)
        
        if target_category is None:
            target_category = model_output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(model_output)
        one_hot[0, target_category] = 1
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        # Weight the channels by the gradient
        weights = torch.mean(self.gradients, dim=(2, 3, 4))  # Global average pooling of gradients
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32).to(input_tensor.device)
        
        # Weight the activations by the gradients
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i]
        
        # Apply ReLU to focus on the positive contributions
        cam = F.relu(cam)
        
        # Normalize
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        # Convert to numpy
        cam = cam.cpu().detach().numpy()
        
        return cam

def preprocess_video(video_path, transform, num_frames=16):
    """
    Extract frames from video and preprocess them for the model
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None, None
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= 0:
            cap.release()
            return None, None
        
        # Sample frames evenly
        if frame_count <= num_frames:
            frame_indices = list(range(frame_count))
            while len(frame_indices) < num_frames:
                frame_indices.append(frame_indices[-1] if frame_indices else 0)
        else:
            frame_indices = np.linspace(0, frame_count-1, num_frames, dtype=int)
        
        # Extract frames
        frames = []
        original_frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save original frame
            original_frames.append(frame.copy())
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transform
            frame_processed = transform(frame_rgb)
            frames.append(frame_processed)
        
        cap.release()
        
        # Stack frames
        if len(frames) < num_frames:
            # Pad with the last frame
            while len(frames) < num_frames:
                frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
                original_frames.append(original_frames[-1] if original_frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        return torch.stack(frames), original_frames
    
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None, None

def overlay_cam_on_frame(frame, cam, alpha=0.5):
    """
    Overlay the class activation map on a frame
    """
    # Resize CAM to match frame
    cam_resized = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = alpha * heatmap + (1 - alpha) * frame
    
    return np.uint8(overlay)

def visualize_model_attention(model, video_path, output_dir, config):
    """
    Visualize what parts of the input frames the model is focusing on using Grad-CAM
    
    Args:
        model: Trained model
        video_path: Path to the video file
        output_dir: Directory to save visualizations
        config: Configuration object with model parameters
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create transform for preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.frame_height, config.frame_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Extract frames from video
    frames, original_frames = preprocess_video(video_path, transform, config.num_frames)
    if frames is None:
        print(f"Failed to process video: {video_path}")
        return
    
    # Prepare input for the model
    clips_tensor = frames.unsqueeze(0).unsqueeze(0).to(config.device)  # [1, 1, T, C, H, W]
    replay_speeds = torch.tensor([[1.0]], dtype=torch.float32).to(config.device)  # Assuming normal speed
    num_clips = torch.tensor([1], dtype=torch.long).to(config.device)
    
    # Create GradCAM instance
    # Target the last convolutional layer in the base model
    target_layer = model.base_model[4][1].conv2  # Last ResNet block's conv layer
    grad_cam = GradCAM(model, target_layer)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(clips_tensor, replay_speeds, num_clips)
    
    # Extract predictions
    body_part_pred = torch.argmax(outputs['body_part'], dim=1).item()
    severity_pred = torch.argmax(outputs['severity'], dim=1).item()
    
    if body_part_pred == 0:  # Upper body
        action_pred = torch.argmax(outputs['upper_body'], dim=1).item()
    else:  # Lower body
        action_pred = torch.argmax(outputs['lower_body'], dim=1).item() + 5
    
    # Convert to class names
    body_part_name = config.body_part_classes[body_part_pred]
    action_name = config.action_classes[action_pred]
    severity_name = config.severity_classes[severity_pred]
    
    print(f"Predictions:")
    print(f"  Body Part: {body_part_name}")
    print(f"  Action: {action_name}")
    print(f"  Severity: {severity_name}")
    
    # Generate class activation maps for key frames
    # Select 6 frames evenly spaced throughout the video
    key_frame_indices = np.linspace(0, config.num_frames-1, 6, dtype=int)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual frame visualizations and create figure
    plt.figure(figsize=(20, 10))
    
    for i, frame_idx in enumerate(key_frame_indices):
        # Create a placeholder tensor for the current frame, shaped like a clip
        frame_tensor = frames[frame_idx].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, C, H, W]
        frame_tensor = frame_tensor.to(config.device)
        
        # Get CAMs for each classification task
        cams = {}
        
        # Body part CAM
        model.zero_grad()
        body_part_logits = model.body_part_classifier(torch.cat([
            model.base_model(frame_tensor.permute(0, 2, 3, 1, 4, 5)).view(-1, model.feature_dim),
            torch.ones(1, 1).to(frame_tensor.device)  # Dummy replay speed
        ], dim=1))
        body_part_cam = grad_cam.generate_cam(
            frame_tensor.permute(0, 2, 3, 1, 4, 5),  # [1, C, T, 1, H, W]
            target_category=body_part_pred
        )
        cams['body_part'] = body_part_cam
        
        # Action CAM
        model.zero_grad()
        if body_part_pred == 0:  # Upper body
            action_logits = model.upper_body_classifier(torch.cat([
                model.base_model(frame_tensor.permute(0, 2, 3, 1, 4, 5)).view(-1, model.feature_dim),
                torch.ones(1, 1).to(frame_tensor.device)  # Dummy replay speed
            ], dim=1))
        else:  # Lower body
            action_logits = model.lower_body_classifier(torch.cat([
                model.base_model(frame_tensor.permute(0, 2, 3, 1, 4, 5)).view(-1, model.feature_dim),
                torch.ones(1, 1).to(frame_tensor.device)  # Dummy replay speed
            ], dim=1))
        action_cam = grad_cam.generate_cam(
            frame_tensor.permute(0, 2, 3, 1, 4, 5),
            target_category=action_pred % 5  # Use mod 5 since we have separate classifiers for upper/lower
        )
        cams['action'] = action_cam
        
        # Severity CAM
        model.zero_grad()
        severity_logits = model.severity_classifier(torch.cat([
            model.base_model(frame_tensor.permute(0, 2, 3, 1, 4, 5)).view(-1, model.feature_dim),
            torch.ones(1, 1).to(frame_tensor.device)  # Dummy replay speed
        ], dim=1))
        severity_cam = grad_cam.generate_cam(
            frame_tensor.permute(0, 2, 3, 1, 4, 5),
            target_category=severity_pred
        )
        cams['severity'] = severity_cam
        
        # Get original frame
        original_frame = cv2.cvtColor(original_frames[frame_idx], cv2.COLOR_BGR2RGB)
        
        # Create overlays
        overlays = {
            'body_part': overlay_cam_on_frame(original_frame, cams['body_part']),
            'action': overlay_cam_on_frame(original_frame, cams['action']),
            'severity': overlay_cam_on_frame(original_frame, cams['severity'])
        }
        
        # Plot in the figure
        plt.subplot(3, 6, i+1)
        plt.imshow(original_frame)
        plt.title(f"Frame {frame_idx}")
        plt.axis('off')
        
        plt.subplot(3, 6, i+7)
        plt.imshow(overlays['body_part'])
        if i == 0:
            plt.ylabel(f"Body Part\n({body_part_name})")
        plt.axis('off')
        
        plt.subplot(3, 6, i+13)
        plt.imshow(overlays['action'])
        if i == 0:
            plt.ylabel(f"Action\n({action_name})")
        plt.axis('off')
        
        # Save individual visualizations
        for task, overlay in overlays.items():
            task_dir = output_dir / task
            task_dir.mkdir(exist_ok=True)
            
            img = Image.fromarray(overlay)
            img.save(task_dir / f"frame_{frame_idx}.png")
    
    # Add a title
    plt.suptitle(
        f"Foul Classification: {body_part_name} - {action_name} - {severity_name}",
        fontsize=16
    )
    
    # Save the combined visualization
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_dir / "combined_visualization.png")
    plt.close()
    
    # Save predictions
    predictions = {
        'body_part': body_part_name,
        'action': action_name,
        'severity': severity_name
    }
    
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
    
    print(f"Visualizations saved to {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize model attention for soccer foul classification')
    parser.add_argument('--video_path', required=True, help='Path to input video')
    parser.add_argument('--output_dir', required=True, help='Directory to save visualizations')
    parser.add_argument('--model_path', default='best_mvfoul_model.pth', help='Path to trained model')
    args = parser.parse_args()
    
    # Configuration
    config = Config()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the model
    model = MVFoulClassifier(
        num_classes=len(config.action_classes),
        num_severity_classes=len(config.severity_classes)
    ).to(config.device)
    
    # Load the trained model
    try:
        checkpoint = torch.load(args.model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Run visualization
    visualize_model_attention(model, args.video_path, args.output_dir, config)

if __name__ == "__main__":
    main()