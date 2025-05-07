# transformer/evaluate_performance.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from model.mvit_model import MultiViewTransformer
import os
import json
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(model_path, config):
    """Load trained model from checkpoint"""
    model = MultiViewTransformer(
        num_frames=config['num_frames'],
        num_classes=config['num_classes'],
        use_replay_speed=config['use_replay_speed']
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model, data_loader, use_body_part_prediction=True, num_views=None, use_replay_speed=True, fusion_strategy=None):
    """
    Evaluate model performance
    
    Args:
        model: The trained model
        data_loader: DataLoader with test data
        use_body_part_prediction: Whether to use hierarchical approach with body part prediction
        num_views: Number of views to use (None = all available)
        use_replay_speed: Whether to use replay speed feature
        fusion_strategy: Which fusion strategy to use (None = use model's default)
    
    Returns:
        Dictionary with evaluation metrics
    """
    all_body_part_preds = []
    all_body_part_labels = []
    all_foul_preds = []
    all_foul_labels = []
    all_severity_preds = []
    all_severity_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # Unpack batch data
            video_clips = batch['video_clips'].to(device)  # Shape: [B, V, C, T, H, W]
            body_part_labels = batch['body_part'].to(device)
            foul_labels = batch['foul_type'].to(device)
            severity_labels = batch['severity'].to(device)
            
            if 'replay_speed' in batch and use_replay_speed:
                replay_speed = batch['replay_speed'].to(device)
            else:
                replay_speed = None
            
            # If num_views is specified, limit the number of views used
            if num_views is not None:
                # Ensure we don't exceed the available views
                actual_views = min(num_views, video_clips.shape[1])
                video_clips = video_clips[:, :actual_views]
            
            # Get model predictions
            body_part_preds, upper_body_preds, lower_body_preds, severity_preds = model(
                video_clips, replay_speed, fusion_strategy=fusion_strategy
            )
            
            # If using hierarchical approach, select foul predictions based on body part prediction
            if use_body_part_prediction:
                # Decide which classifier (upper or lower body) to use based on body part prediction
                body_part_pred_labels = torch.argmax(body_part_preds, dim=1)
                batch_size = body_part_pred_labels.size(0)
                
                # Initialize foul predictions
                foul_preds = torch.zeros((batch_size, 10), device=device)
                
                # For samples predicted as upper body (class 0), use upper body predictions
                upper_indices = (body_part_pred_labels == 0).nonzero(as_tuple=True)[0]
                if upper_indices.size(0) > 0:
                    foul_preds[upper_indices, :5] = upper_body_preds[upper_indices]
                
                # For samples predicted as lower body (class 1), use lower body predictions
                lower_indices = (body_part_pred_labels == 1).nonzero(as_tuple=True)[0]
                if lower_indices.size(0) > 0:
                    foul_preds[lower_indices, 5:] = lower_body_preds[lower_indices]
            else:
                # For flat classification, simply concatenate all predictions
                foul_preds = torch.cat([upper_body_preds, lower_body_preds], dim=1)
            
            # Collect predictions and labels
            all_body_part_preds.append(body_part_preds.cpu().numpy())
            all_body_part_labels.append(body_part_labels.cpu().numpy())
            all_foul_preds.append(foul_preds.cpu().numpy())
            all_foul_labels.append(foul_labels.cpu().numpy())
            all_severity_preds.append(severity_preds.cpu().numpy())
            all_severity_labels.append(severity_labels.cpu().numpy())
    
    # Concatenate all batches
    all_body_part_preds = np.concatenate(all_body_part_preds, axis=0)
    all_body_part_labels = np.concatenate(all_body_part_labels, axis=0)
    all_foul_preds = np.concatenate(all_foul_preds, axis=0)
    all_foul_labels = np.concatenate(all_foul_labels, axis=0)
    all_severity_preds = np.concatenate(all_severity_preds, axis=0)
    all_severity_labels = np.concatenate(all_severity_labels, axis=0)
    
    # Convert predictions to class indices
    body_part_pred_indices = np.argmax(all_body_part_preds, axis=1)
    foul_pred_indices = np.argmax(all_foul_preds, axis=1)
    severity_pred_indices = np.argmax(all_severity_preds, axis=1)
    
    # Calculate metrics
    body_part_accuracy = accuracy_score(all_body_part_labels, body_part_pred_indices)
    foul_accuracy = accuracy_score(all_foul_labels, foul_pred_indices)
    severity_accuracy = accuracy_score(all_severity_labels, severity_pred_indices)
    
    # Calculate weighted F1 scores
    foul_f1 = f1_score(all_foul_labels, foul_pred_indices, average='weighted')
    severity_f1 = f1_score(all_severity_labels, severity_pred_indices, average='weighted')
    
    results = {
        'body_part_accuracy': body_part_accuracy * 100,
        'foul_accuracy': foul_accuracy * 100,
        'severity_accuracy': severity_accuracy * 100,
        'foul_f1': foul_f1,
        'severity_f1': severity_f1
    }
    
    return results

def run_multiview_comparison(model, data_loaders):
    """
    Compare performance with different numbers of views
    
    Args:
        model: The trained model
        data_loaders: Dictionary with DataLoaders for different view configurations
    
    Returns:
        DataFrame with results
    """
    results = []
    
    view_configs = {
        'Single View (Main Camera)': {'loader': data_loaders['main_camera'], 'num_views': 1},
        'Single View (Close-up)': {'loader': data_loaders['closeup'], 'num_views': 1},
        'Two Views': {'loader': data_loaders['test'], 'num_views': 2},
        'Three+ Views (when available)': {'loader': data_loaders['test'], 'num_views': None}
    }
    
    for name, config in view_configs.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_model(
            model, 
            config['loader'], 
            use_body_part_prediction=True, 
            num_views=config['num_views']
        )
        
        results.append({
            'View Configuration': name,
            'Foul Type Accuracy': metrics['foul_accuracy'],
            'Weighted F1': metrics['foul_f1']
        })
    
    return pd.DataFrame(results)

def compare_hierarchical_vs_flat(model, data_loader):
    """
    Compare hierarchical vs flat classification
    
    Args:
        model: The trained model
        data_loader: DataLoader with test data
    
    Returns:
        DataFrame with results
    """
    results = []
    
    print("Evaluating hierarchical classification...")
    hierarchical_metrics = evaluate_model(
        model, 
        data_loader, 
        use_body_part_prediction=True
    )
    
    print("Evaluating flat classification...")
    flat_metrics = evaluate_model(
        model, 
        data_loader, 
        use_body_part_prediction=False
    )
    
    results = [
        {
            'Classification Approach': 'Flat (10-class)',
            'Foul Type Accuracy': flat_metrics['foul_accuracy'],
            'Weighted F1': flat_metrics['foul_f1']
        },
        {
            'Classification Approach': 'Hierarchical (2-class → 5+5-class)',
            'Foul Type Accuracy': hierarchical_metrics['foul_accuracy'],
            'Weighted F1': hierarchical_metrics['foul_f1']
        }
    ]
    
    return pd.DataFrame(results)

def run_ablation_study_replay_speed(model, data_loader):
    """
    Ablation study on the impact of replay speed
    
    Args:
        model: The trained model
        data_loader: DataLoader with test data
    
    Returns:
        DataFrame with results
    """
    results = []
    
    print("Evaluating with replay speed...")
    with_replay_metrics = evaluate_model(
        model, 
        data_loader, 
        use_replay_speed=True
    )
    
    print("Evaluating without replay speed...")
    without_replay_metrics = evaluate_model(
        model, 
        data_loader, 
        use_replay_speed=False
    )
    
    results = [
        {
            'Model Configuration': 'Without replay speed',
            'Severity Accuracy': without_replay_metrics['severity_accuracy'],
            'Severity F1': without_replay_metrics['severity_f1']
        },
        {
            'Model Configuration': 'With replay speed',
            'Severity Accuracy': with_replay_metrics['severity_accuracy'],
            'Severity F1': with_replay_metrics['severity_f1']
        }
    ]
    
    return pd.DataFrame(results)

def run_view_fusion_comparison(model, data_loader):
    """
    Compare different view fusion strategies
    
    Args:
        model: The model with different fusion strategies implemented
        data_loader: DataLoader with test data
    
    Returns:
        DataFrame with results
    """
    results = []
    
    fusion_strategies = ['simple_avg', 'max_pooling', 'attention']
    fusion_names = {
        'simple_avg': 'Simple Averaging',
        'max_pooling': 'Max Pooling',
        'attention': 'Learned Attention'
    }
    
    for strategy in fusion_strategies:
        print(f"Evaluating {fusion_names[strategy]} fusion strategy...")
        
        metrics = evaluate_model(
            model, 
            data_loader, 
            fusion_strategy=strategy
        )
        
        results.append({
            'Fusion Method': fusion_names[strategy],
            'Foul Type Accuracy': metrics['foul_accuracy'],
            'Weighted F1': metrics['foul_f1']
        })
    
    return pd.DataFrame(results)

def analyze_confusion_patterns(model, data_loader):
    """
    Analyze confusion patterns to identify common misclassifications
    
    Args:
        model: The trained model
        data_loader: DataLoader with test data
    
    Returns:
        Confusion matrix and analysis findings
    """
    all_foul_preds = []
    all_foul_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # Unpack batch data
            video_clips = batch['video_clips'].to(device)
            foul_labels = batch['foul_type'].to(device)
            
            if 'replay_speed' in batch:
                replay_speed = batch['replay_speed'].to(device)
            else:
                replay_speed = None
            
            # Get model predictions
            body_part_preds, upper_body_preds, lower_body_preds, _ = model(
                video_clips, replay_speed
            )
            
            # Decide which classifier (upper or lower body) to use based on body part prediction
            body_part_pred_labels = torch.argmax(body_part_preds, dim=1)
            batch_size = body_part_pred_labels.size(0)
            
            # Initialize foul predictions
            foul_preds = torch.zeros((batch_size, 10), device=device)
            
            # For samples predicted as upper body (class 0), use upper body predictions
            upper_indices = (body_part_pred_labels == 0).nonzero(as_tuple=True)[0]
            if upper_indices.size(0) > 0:
                foul_preds[upper_indices, :5] = upper_body_preds[upper_indices]
            
            # For samples predicted as lower body (class 1), use lower body predictions
            lower_indices = (body_part_pred_labels == 1).nonzero(as_tuple=True)[0]
            if lower_indices.size(0) > 0:
                foul_preds[lower_indices, 5:] = lower_body_preds[lower_indices]
            
            # Collect predictions and labels
            all_foul_preds.append(foul_preds.cpu().numpy())
            all_foul_labels.append(foul_labels.cpu().numpy())
    
    # Concatenate all batches
    all_foul_preds = np.concatenate(all_foul_preds, axis=0)
    all_foul_labels = np.concatenate(all_foul_labels, axis=0)
    
    # Convert predictions to class indices
    foul_pred_indices = np.argmax(all_foul_preds, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_foul_labels, foul_pred_indices)
    
    # Calculate prediction confidences
    pred_confidences = np.max(all_foul_preds, axis=1)
    
    # Calculate average confidence for correct and incorrect predictions
    correct_mask = (foul_pred_indices == all_foul_labels)
    avg_correct_confidence = np.mean(pred_confidences[correct_mask])
    avg_incorrect_confidence = np.mean(pred_confidences[~correct_mask])
    
    # Identify most common confusion patterns
    # Get indices of non-diagonal elements in the confusion matrix
    rows, cols = np.where(np.eye(cm.shape[0]) == 0)
    confusion_scores = [(cm[i, j], i, j) for i, j in zip(rows, cols)]
    confusion_scores.sort(reverse=True)
    
    # Map class indices to foul names
    foul_names = [
        "Push", "Hold", "Elbow", "Challenge", "Charge",  # Upper body fouls
        "Tackle", "Standing Tackle", "High Leg", "Kick", "Step"  # Lower body fouls
    ]
    
    top_confusions = []
    for score, true_idx, pred_idx in confusion_scores[:5]:
        if score > 0:
            top_confusions.append({
                'True Foul': foul_names[true_idx],
                'Predicted Foul': foul_names[pred_idx],
                'Count': score
            })
    
    results = {
        'confusion_matrix': cm,
        'top_confusions': top_confusions,
        'avg_correct_confidence': avg_correct_confidence,
        'avg_incorrect_confidence': avg_incorrect_confidence
    }
    
    return results

if __name__ == '__main__':
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Load test data
    from data_loader import get_dataloaders
    data_loaders = get_dataloaders(config)
    
    # Load model
    model = load_model('checkpoints/best_model.pth', config)
    
    # Multi-view vs single-view comparison
    multiview_results = run_multiview_comparison(model, data_loaders)
    print("\nMulti-View vs. Single-View Performance:")
    print(multiview_results)
    multiview_results.to_csv('results/multiview_comparison.csv', index=False)
    
    # Hierarchical vs flat classification
    hierarchy_results = compare_hierarchical_vs_flat(model, data_loaders['test'])
    print("\nHierarchical vs. Flat Classification:")
    print(hierarchy_results)
    hierarchy_results.to_csv('results/hierarchy_comparison.csv', index=False)
    
    # Ablation study: Impact of replay speed
    replay_speed_results = run_ablation_study_replay_speed(model, data_loaders['test'])
    print("\nImpact of Replay Speed Feature:")
    print(replay_speed_results)
    replay_speed_results.to_csv('results/replay_speed_ablation.csv', index=False)
    
    # Fusion strategy comparison (MViT model can switch between fusion strategies at inference)
    fusion_results = run_view_fusion_comparison(model, data_loaders['test'])
    print("\nComparison of Multi-view Fusion Strategies:")
    print(fusion_results)
    fusion_results.to_csv('results/fusion_strategy_comparison.csv', index=False)
    
    # Analyze confusion patterns
    confusion_analysis = analyze_confusion_patterns(model, data_loaders['test'])
    print("\nTop confusion patterns:")
    for confusion in confusion_analysis['top_confusions']:
        print(f"True: {confusion['True Foul']} → Predicted: {confusion['Predicted Foul']} (Count: {confusion['Count']})")
    
    print(f"\nAverage confidence for correct predictions: {confusion_analysis['avg_correct_confidence']:.3f}")
    print(f"Average confidence for incorrect predictions: {confusion_analysis['avg_incorrect_confidence']:.3f}")
    
    # Create visualization of results
    plt.figure(figsize=(10, 6))
    plt.bar(multiview_results['View Configuration'], multiview_results['Foul Type Accuracy'])
    plt.title('Impact of Multiple Views on Foul Classification Accuracy (MViT Model)')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/multiview_comparison_mvit.png')
    
    # Create visualization for fusion strategies
    plt.figure(figsize=(8, 5))
    plt.bar(fusion_results['Fusion Method'], fusion_results['Foul Type Accuracy'])
    plt.title('Comparison of Multi-view Fusion Strategies (MViT Model)')
    plt.ylabel('Foul Type Accuracy (%)')
    plt.ylim(40, 46)
    plt.tight_layout()
    plt.savefig('results/fusion_comparison_mvit.png')
    
    # Create combined visualization for comparison
    both_models = pd.DataFrame({
        'View Configuration': multiview_results['View Configuration'],
        'MViT Model': multiview_results['Foul Type Accuracy'],
        'CNN Model': [36.15, 38.42, 45.77, 47.33]  # Example values from CNN model
    })
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(both_models['View Configuration']))
    width = 0.35
    
    plt.bar(x - width/2, both_models['CNN Model'], width, label='CNN Model')
    plt.bar(x + width/2, both_models['MViT Model'], width, label='MViT Model')
    
    plt.xlabel('Number of Views')
    plt.ylabel('Foul Type Accuracy (%)')
    plt.title('CNN vs MViT Performance with Different View Counts')
    plt.xticks(x, both_models['View Configuration'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')