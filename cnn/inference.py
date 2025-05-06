"""
Inference script for Soccer Foul Classification model.
This script performs inference on new videos using a trained model.
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys

# Import the model and config classes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MVFoulClassifier, Config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run inference on soccer videos for foul classification')
    parser.add_argument('--input_path', required=True, help='Path to input video or directory')
    parser.add_argument('--output_path', required=True, help='Path to save results')
    parser.add_argument('--model_path', default='best_mvfoul_model.pth', help='Path to trained model')
    return parser.parse_args()

def preprocess_video(video_path, transform, config):
    """Extract and preprocess frames from a video"""
    video_extensions = [config.video_extension, ".avi", ".mp4", ".mkv"]
    frames = None
    original_frames = None
    
    # Try different extensions
    for ext in video_extensions:
        full_path = str(video_path) + ext
        if os.path.exists(full_path):
            try:
                cap = cv2.VideoCapture(full_path)
                if not cap.isOpened():
                    continue
                
                # Get video properties
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if frame_count <= 0:
                    cap.release()
                    continue
                
                # Sample frames
                frame_indices = np.linspace(0, min(100, frame_count) - 1, config.num_frames, dtype=int)
                
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
                    
                    # Preprocess frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_processed = transform(frame_rgb)
                    frames.append(frame_processed)
                
                cap.release()
                
                if len(frames) == config.num_frames:
                    return torch.stack(frames), original_frames
                
                # If we didn't get enough frames, pad with the last frame
                if frames:
                    while len(frames) < config.num_frames:
                        frames.append(frames[-1])
                        original_frames.append(original_frames[-1])
                    
                    return torch.stack(frames), original_frames
                
            except Exception as e:
                print(f"Error processing video {full_path}: {str(e)}")
    
    # If we reach here, we couldn't load the video with any extension
    print(f"Could not load video: {video_path}")
    
    # Create dummy data for testing
    dummy_frames = []
    dummy_originals = []
    
    # Generate random normalized frames
    for _ in range(config.num_frames):
        dummy_frame = np.random.rand(config.frame_height, config.frame_width, 3)
        dummy_originals.append((dummy_frame * 255).astype(np.uint8))
        
        # Apply transform
        processed = transform(dummy_frame)
        dummy_frames.append(processed)
    
    return torch.stack(dummy_frames), dummy_originals

def estimate_replay_speed(frames):
    """Estimate replay speed based on motion between frames"""
    if len(frames) < 2:
        return 1.0
    
    # Compute optical flow between consecutive frames
    flow_magnitudes = []
    for i in range(len(frames) - 1):
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        try:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = np.mean(mag)
            flow_magnitudes.append(mean_mag)
        except:
            flow_magnitudes.append(0.0)
    
    if not flow_magnitudes:
        return 1.0
    
    avg_flow = np.mean(flow_magnitudes)
    
    # Map average flow to replay speed
    if avg_flow < 1.0:
        return 1.8  # Slow motion replay
    elif avg_flow < 3.0:
        return 1.4  # Medium slow motion
    else:
        return 1.0  # Regular speed

def run_inference(model, video_path, config):
    """Run inference on a single video"""
    # Create preprocessing transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.frame_height, config.frame_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Extract frames from video
    frames, original_frames = preprocess_video(video_path, transform, config)
    if frames is None:
        return None, None
    
    # Estimate replay speed
    replay_speed = estimate_replay_speed(original_frames)
    
    # Prepare input tensors
    clips_tensor = frames.unsqueeze(0).unsqueeze(0)  # Add batch and clip dimensions
    replay_speeds = torch.tensor([[replay_speed]], dtype=torch.float32).to(config.device)
    num_clips = torch.tensor([1], dtype=torch.long).to(config.device)
    
    # Move data to device
    clips_tensor = clips_tensor.to(config.device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(clips_tensor, replay_speeds, num_clips)
    
    # Get predictions
    body_part_pred = torch.argmax(outputs['body_part'], dim=1).item()
    severity_pred = torch.argmax(outputs['severity'], dim=1).item()
    
    # For action prediction, use the appropriate classifier based on predicted body part
    if body_part_pred == 0:  # Upper body
        action_pred = torch.argmax(outputs['upper_body'], dim=1).item()
    else:  # Lower body
        action_pred = torch.argmax(outputs['lower_body'], dim=1).item() + 5
    
    # Get probabilities
    body_part_probs = torch.softmax(outputs['body_part'], dim=1)[0].cpu().numpy()
    severity_probs = torch.softmax(outputs['severity'], dim=1)[0].cpu().numpy()
    
    if body_part_pred == 0:  # Upper body
        action_probs = torch.softmax(outputs['upper_body'], dim=1)[0].cpu().numpy()
        # Pad with zeros for lower body actions
        action_probs = np.concatenate([action_probs, np.zeros(5)])
    else:  # Lower body
        action_probs = torch.softmax(outputs['lower_body'], dim=1)[0].cpu().numpy()
        # Pad with zeros for upper body actions
        action_probs = np.concatenate([np.zeros(5), action_probs])
    
    # Convert to class names
    body_part_name = config.body_part_classes[body_part_pred]
    action_name = config.action_classes[action_pred]
    severity_name = config.severity_classes[severity_pred]
    
    # Return predictions and original frames for visualization
    result = {
        'body_part': {
            'predicted': body_part_name,
            'index': body_part_pred,
            'probabilities': {cls: float(prob) for cls, prob in zip(config.body_part_classes, body_part_probs)}
        },
        'action': {
            'predicted': action_name,
            'index': action_pred,
            'probabilities': {cls: float(prob) for cls, prob in zip(config.action_classes, action_probs)}
        },
        'severity': {
            'predicted': severity_name,
            'index': severity_pred,
            'probabilities': {cls: float(prob) for cls, prob in zip(config.severity_classes, severity_probs)}
        },
        'estimated_replay_speed': float(replay_speed)
    }
    
    return result, original_frames

def visualize_result(result, frames, output_path=None):
    """Visualize inference result with selected frames"""
    if result is None or frames is None:
        return
    
    # Create a figure
    fig = plt.figure(figsize=(16, 10))
    
    # Add a title with the predictions
    plt.suptitle(
        f"Foul Classification\n"
        f"Body Part: {result['body_part']['predicted']} ({result['body_part']['probabilities'][result['body_part']['predicted']]:.2f})\n"
        f"Action: {result['action']['predicted']} ({result['action']['probabilities'][result['action']['predicted']]:.2f})\n"
        f"Severity: {result['severity']['predicted']} ({result['severity']['probabilities'][result['severity']['predicted']]:.2f})",
        fontsize=16
    )
    
    # Select 6 frames to display (evenly spaced)
    indices = np.linspace(0, len(frames)-1, 6, dtype=int)
    
    # Display frames
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(2, 3, i+1)
        frame = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        ax.imshow(frame)
        ax.set_title(f"Frame {idx}")
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def process_directory(model, input_dir, output_dir, config):
    """Process all video files in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all folders that might contain videos
    # For SoccerNet-MVFouls format, these would be action_X folders
    action_folders = [f for f in input_path.glob("action_*") if f.is_dir()]
    
    if not action_folders:
        print(f"No action folders found in {input_dir}")
        return {}
    
    results = {}
    
    for action_folder in tqdm(action_folders, desc="Processing actions"):
        action_id = action_folder.name
        
        # Look for clips in this action folder
        clip_paths = sorted(action_folder.glob("clip_*"))
        
        if not clip_paths:
            # Handle case where path doesn't include clip_ prefix
            clip_paths = sorted(list(action_folder.iterdir()))
        
        action_results = []
        
        for clip_path in clip_paths:
            # Remove extension if present
            clip_path_str = str(clip_path)
            if clip_path_str.endswith(('.avi', '.mp4', '.mkv')):
                clip_path_str = os.path.splitext(clip_path_str)[0]
                
            # Run inference
            result, frames = run_inference(model, clip_path_str, config)
            
            if result is not None:
                # Save result
                action_results.append(result)
                
                # Save visualization
                vis_path = output_path / f"{action_id}_{clip_path.stem}_visualization.png"
                visualize_result(result, frames, str(vis_path))
        
        if action_results:
            results[action_id] = action_results
    
    # Save all results to a JSON file
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Processed {len(results)} actions")
    return results

def process_single_video(model, video_path, output_dir, config):
    """Process a single video file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    result, frames = run_inference(model, video_path, config)
    
    if result is not None:
        # Save result
        video_name = Path(video_path).stem
        with open(output_path / f"{video_name}_result.json", "w") as f:
            json.dump(result, f, indent=4)
        
        # Save visualization
        vis_path = output_path / f"{video_name}_visualization.png"
        visualize_result(result, frames, str(vis_path))
    
    return result

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create the model
    config = Config()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {config.device}")
    
    model = MVFoulClassifier(
        num_classes=len(config.action_classes),
        num_severity_classes=len(config.severity_classes)
    ).to(config.device)
    
    # Load the trained model with correct settings for PyTorch 2.6+
    try:
        print(f"Loading model from {args.model_path}")
        # Use weights_only=False to handle older PyTorch checkpoints
        checkpoint = torch.load(args.model_path, map_location=config.device, weights_only=False)
        
        # Check if it's a state_dict or a full checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model checkpoint with validation F1: {checkpoint.get('val_f1', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dictionary")
        
    except Exception as e:
        # Add a fallback for older PyTorch versions
        try:
            print(f"First loading attempt failed, trying alternative method...")
            checkpoint = torch.load(args.model_path, map_location=config.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            print("Successfully loaded model with alternative method")
        except Exception as e2:
            print(f"Error loading model: {str(e)}")
            print(f"Alternative loading also failed: {str(e2)}")
            return
    
    # Check if input is a file or directory
    input_path = Path(args.input_path)
    if input_path.is_file():
        print(f"Processing single video: {input_path}")
        process_single_video(model, str(input_path), args.output_path, config)
    elif input_path.is_dir():
        print(f"Processing directory: {input_path}")
        process_directory(model, args.input_path, args.output_path, config)
    else:
        print(f"Input path {args.input_path} does not exist")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

""" 
Soccer Foul Classification - Inference Script
This script loads a trained model and runs inference on a single video or a directory of videos.

Usage:
    python inference.py --input_path <path_to_video_or_directory> --output_path <path_to_save_results>


import os
import sys
import argparse
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Add the project root to path so we can import from the model file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our model definition
from model import MVFoulClassifier, Config

# Configuration
config = Config()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing transform for inference
inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config.frame_height, config.frame_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_frames_from_video(video_path, num_frames=16):
    #Extract frames from a video file for inference.
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None, None
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        
        if frame_count <= 0:
            cap.release()
            return None, None
        
        # Sample frames focusing on the first 4 seconds (peak action)
        # If video shorter than 4 seconds, use full video
        target_seconds = min(4.0, duration)
        target_frames = min(int(target_seconds * fps), frame_count)
        
        # Calculate the frame indices to sample
        if target_frames <= num_frames:
            # If we have fewer frames than needed, sample all and repeat
            frame_indices = list(range(target_frames))
            while len(frame_indices) < num_frames:
                frame_indices.append(frame_indices[-1])
        else:
            # Sample evenly from the target range
            frame_indices = np.linspace(0, target_frames-1, num_frames, dtype=int)
        
        # Extract the frames
        frames = []
        original_frames = []  # Keep original frames for visualization
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save original frame for visualization
            original_frames.append(frame.copy())
            
            # Convert from BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transform
            frame_processed = inference_transform(frame_rgb)
            frames.append(frame_processed)
        
        cap.release()
        
        if len(frames) < num_frames:
            print(f"Warning: Only {len(frames)} frames extracted from {video_path}. Padding...")
            # Pad with the last frame if not enough frames
            while len(frames) < num_frames:
                frames.append(frames[-1] if frames else torch.zeros(3, config.frame_height, config.frame_width))
                original_frames.append(original_frames[-1] if original_frames else np.zeros((config.frame_height, config.frame_width, 3), dtype=np.uint8))
        
        return torch.stack(frames), original_frames
    
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None, None

def compute_replay_speed(frames):
    if len(frames) < 2:
        return 1.0
    
    # Compute optical flow between consecutive frames
    flow_magnitudes = []
    for i in range(len(frames) - 1):
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = np.mean(mag)
        flow_magnitudes.append(mean_mag)
    
    avg_flow = np.mean(flow_magnitudes)
    
    # Map average flow to replay speed
    # Lower flow typically means slower replay (higher replay speed value)
    # This mapping is empirical and should be calibrated based on your data
    if avg_flow < 1.0:
        return 1.8  # Slow motion replay
    elif avg_flow < 3.0:
        return 1.4  # Medium slow motion
    else:
        return 1.0  # Regular speed
    
def run_inference(model, video_path):
    # Extract frames from video
    frames, original_frames = extract_frames_from_video(video_path, config.num_frames)
    if frames is None:
        return None
    
    # Estimate replay speed
    replay_speed = compute_replay_speed(original_frames)
    
    # Prepare input tensors
    clips_tensor = frames.unsqueeze(0).unsqueeze(0)  # Add batch and clip dimensions
    replay_speeds = torch.tensor([[replay_speed]], dtype=torch.float32).to(config.device)
    num_clips = torch.tensor([1], dtype=torch.long).to(config.device)
    
    # Move data to device
    clips_tensor = clips_tensor.to(config.device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(clips_tensor, replay_speeds, num_clips)
    
    # Get predictions
    body_part_pred = torch.argmax(outputs['body_part'], dim=1).item()
    severity_pred = torch.argmax(outputs['severity'], dim=1).item()
    
    # For action prediction, use the appropriate classifier based on predicted body part
    if body_part_pred == 0:  # Upper body
        action_pred = torch.argmax(outputs['upper_body'], dim=1).item()
    else:  # Lower body
        action_pred = torch.argmax(outputs['lower_body'], dim=1).item() + 5
    
    # Get probabilities
    body_part_probs = torch.softmax(outputs['body_part'], dim=1)[0].cpu().numpy()
    severity_probs = torch.softmax(outputs['severity'], dim=1)[0].cpu().numpy()
    
    if body_part_pred == 0:  # Upper body
        action_probs = torch.softmax(outputs['upper_body'], dim=1)[0].cpu().numpy()
        # Pad with zeros for lower body actions
        action_probs = np.concatenate([action_probs, np.zeros(5)])
    else:  # Lower body
        action_probs = torch.softmax(outputs['lower_body'], dim=1)[0].cpu().numpy()
        # Pad with zeros for upper body actions
        action_probs = np.concatenate([np.zeros(5), action_probs])
    
    # Convert to class names
    body_part_name = config.body_part_classes[body_part_pred]
    action_name = config.action_classes[action_pred]
    severity_name = config.severity_classes[severity_pred]
    
    # Return predictions and original frames for visualization
    result = {
        'body_part': {
            'predicted': body_part_name,
            'index': body_part_pred,
            'probabilities': {cls: float(prob) for cls, prob in zip(config.body_part_classes, body_part_probs)}
        },
        'action': {
            'predicted': action_name,
            'index': action_pred,
            'probabilities': {cls: float(prob) for cls, prob in zip(config.action_classes, action_probs)}
        },
        'severity': {
            'predicted': severity_name,
            'index': severity_pred,
            'probabilities': {cls: float(prob) for cls, prob in zip(config.severity_classes, severity_probs)}
        },
        'estimated_replay_speed': float(replay_speed)
    }
    
    return result, original_frames

def visualize_result(result, frames, output_path=None):
    if result is None or frames is None:
        return
    
    # Create a figure
    fig = plt.figure(figsize=(16, 10))
    
    # Add a title with the predictions
    plt.suptitle(
        f"Foul Classification\n"
        f"Body Part: {result['body_part']['predicted']} ({result['body_part']['probabilities'][result['body_part']['predicted']]:.2f})\n"
        f"Action: {result['action']['predicted']} ({result['action']['probabilities'][result['action']['predicted']]:.2f})\n"
        f"Severity: {result['severity']['predicted']} ({result['severity']['probabilities'][result['severity']['predicted']]:.2f})",
        fontsize=16
    )
    
    # Select 6 frames to display (evenly spaced)
    indices = np.linspace(0, len(frames)-1, 6, dtype=int)
    
    # Display frames
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(2, 3, i+1)
        frame = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        ax.imshow(frame)
        ax.set_title(f"Frame {idx}")
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def process_directory(model, input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = list(input_path.glob('**/*.avi')) + list(input_path.glob('**/*.mp4'))
    
    results = {}
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        # Run inference
        result, frames = run_inference(model, str(video_file))
        
        if result is not None:
            # Save result
            relative_path = video_file.relative_to(input_path)
            results[str(relative_path)] = result
            
            # Save visualization
            vis_path = output_path / f"{relative_path.stem}_visualization.png"
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            visualize_result(result, frames, str(vis_path))
    
    # Save all results to a JSON file
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    return results

def process_single_video(model, video_path, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    result, frames = run_inference(model, video_path)
    
    if result is not None:
        # Save result
        video_name = Path(video_path).stem
        with open(output_path / f"{video_name}_result.json", "w") as f:
            json.dump(result, f, indent=4)
        
        # Save visualization
        vis_path = output_path / f"{video_name}_visualization.png"
        visualize_result(result, frames, str(vis_path))
    
    return result

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inference on soccer videos for foul classification')
    parser.add_argument('--input_path', required=True, help='Path to input video or directory')
    parser.add_argument('--output_path', required=True, help='Path to save results')
    parser.add_argument('--model_path', default='best_mvfoul_model.pth', help='Path to trained model')
    args = parser.parse_args()
    
    # Create the model
    model = MVFoulClassifier(
        num_classes=len(config.action_classes),
        num_severity_classes=len(config.severity_classes)
    ).to(config.device)
    
    # Load the trained model
    try:
        checkpoint = torch.load(args.model_path, map_location=config.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Check if input is a file or directory
    input_path = Path(args.input_path)
    if input_path.is_file():
        print(f"Processing single video: {input_path}")
        process_single_video(model, str(input_path), args.output_path)
    elif input_path.is_dir():
        print(f"Processing directory: {input_path}")
        process_directory(model, args.input_path, args.output_path)
    else:
        print(f"Input path {args.input_path} does not exist")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds") """