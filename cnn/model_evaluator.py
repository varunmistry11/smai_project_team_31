"""
Model performance evaluator for Soccer Foul Classification
This script performs detailed analysis of the model's performance on different types of fouls
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.utils.data import DataLoader

# Import our model and dataset
from model import MVFoulClassifier, SoccerNetMVFoulsDataset, Config, get_transforms

def load_model(model_path, config):
    """Load a trained model from checkpoint"""
    model = MVFoulClassifier(
        num_classes=len(config.action_classes),
        num_severity_classes=len(config.severity_classes)
    ).to(config.device)
    
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from {model_path}")
    print(f"Model validation F1 score: {checkpoint['val_f1']:.4f}")
    
    return model

def evaluate_model(model, test_loader, config):
    """Evaluate model and collect detailed predictions"""
    model.eval()
    
    # Lists to store detailed results
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating model"):
            # Move data to device
            clips = batch['clips'].to(config.device)
            replay_speeds = batch['replay_speeds'].to(config.device)
            num_clips = batch['num_clips'].to(config.device)
            action_label = batch['action_label'].to(config.device)
            severity_label = batch['severity_label'].to(config.device)
            body_part_label = batch['body_part_label'].to(config.device)
            
            # Forward pass
            outputs = model(clips, replay_speeds, num_clips)
            
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
            
            # Get probabilities
            body_part_probs = torch.softmax(outputs['body_part'], dim=1)
            upper_body_probs = torch.softmax(outputs['upper_body'], dim=1)
            lower_body_probs = torch.softmax(outputs['lower_body'], dim=1)
            severity_probs = torch.softmax(outputs['severity'], dim=1)
            
            # Store results for each item in the batch
            batch_size = clips.size(0)
            for i in range(batch_size):
                # Convert to Python types for JSON serialization
                true_action = action_label[i].item()
                true_body_part = body_part_label[i].item()
                true_severity = severity_label[i].item()
                
                pred_action = action_pred[i].item()
                pred_body_part = body_part_pred[i].item()
                pred_severity = severity_pred[i].item()
                
                bp_prob = body_part_probs[i, pred_body_part].item()
                
                if pred_body_part == 0:  # Upper body
                    action_prob = upper_body_probs[i, pred_action].item()
                else:  # Lower body
                    action_prob = lower_body_probs[i, pred_action - 5].item()
                
                severity_prob = severity_probs[i, pred_severity].item()
                
                # Store all information
                all_predictions.append({
                    'true_action': true_action,
                    'true_action_name': config.action_classes[true_action],
                    'true_body_part': true_body_part,
                    'true_body_part_name': config.body_part_classes[true_body_part],
                    'true_severity': true_severity,
                    'true_severity_name': config.severity_classes[true_severity],
                    
                    'pred_action': pred_action,
                    'pred_action_name': config.action_classes[pred_action],
                    'pred_body_part': pred_body_part,
                    'pred_body_part_name': config.body_part_classes[pred_body_part],
                    'pred_severity': pred_severity,
                    'pred_severity_name': config.severity_classes[pred_severity],
                    
                    'body_part_probability': bp_prob,
                    'action_probability': action_prob,
                    'severity_probability': severity_prob,
                    
                    'is_body_part_correct': true_body_part == pred_body_part,
                    'is_action_correct': true_action == pred_action,
                    'is_severity_correct': true_severity == pred_severity,
                    'is_fully_correct': (true_body_part == pred_body_part and 
                                        true_action == pred_action and 
                                        true_severity == pred_severity)
                })
    
    return all_predictions

def analyze_performance_by_class(predictions, config):
    """Analyze model performance by different classes and attributes"""
    # Convert predictions to DataFrame for easier analysis
    df = pd.DataFrame(predictions)
    
    # Overall accuracy metrics
    body_part_acc = df['is_body_part_correct'].mean()
    action_acc = df['is_action_correct'].mean()
    severity_acc = df['is_severity_correct'].mean()
    full_acc = df['is_fully_correct'].mean()
    
    print(f"Overall Body Part Accuracy: {body_part_acc:.4f}")
    print(f"Overall Action Accuracy: {action_acc:.4f}")
    print(f"Overall Severity Accuracy: {severity_acc:.4f}")
    print(f"Overall Full Accuracy: {full_acc:.4f}")
    
    # Analyze accuracy by action class
    action_accuracy = df.groupby('true_action_name')['is_action_correct'].agg(['mean', 'count'])
    action_accuracy.columns = ['Accuracy', 'Count']
    action_accuracy = action_accuracy.sort_values('Accuracy', ascending=False)
    
    print("\nAction Accuracy by Class:")
    print(action_accuracy)
    
    # Plot action accuracy
    plt.figure(figsize=(12, 6))
    bars = plt.bar(action_accuracy.index, action_accuracy['Accuracy'])
    
    # Add count labels
    for i, (_, row) in enumerate(action_accuracy.iterrows()):
        plt.text(i, row['Accuracy'] + 0.02, f"n={int(row['Count'])}", 
                 ha='center', va='bottom', fontsize=9)
    
    plt.axhline(y=action_acc, color='r', linestyle='--', label=f'Average: {action_acc:.2f}')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.title('Action Classification Accuracy by Class')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('action_accuracy_by_class.png')
    plt.close()
    
    # Analyze accuracy by severity class
    severity_accuracy = df.groupby('true_severity_name')['is_severity_correct'].agg(['mean', 'count'])
    severity_accuracy.columns = ['Accuracy', 'Count']
    severity_accuracy = severity_accuracy.sort_values('Accuracy', ascending=False)
    
    print("\nSeverity Accuracy by Class:")
    print(severity_accuracy)
    
    # Plot severity accuracy
    plt.figure(figsize=(10, 6))
    bars = plt.bar(severity_accuracy.index, severity_accuracy['Accuracy'])
    
    # Add count labels
    for i, (_, row) in enumerate(severity_accuracy.iterrows()):
        plt.text(i, row['Accuracy'] + 0.02, f"n={int(row['Count'])}", 
                 ha='center', va='bottom', fontsize=9)
    
    plt.axhline(y=severity_acc, color='r', linestyle='--', label=f'Average: {severity_acc:.2f}')
    plt.ylim(0, 1.1)
    plt.title('Severity Classification Accuracy by Class')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('severity_accuracy_by_class.png')
    plt.close()
    
    # Analyze effect of prediction confidence on accuracy
    plt.figure(figsize=(12, 5))
    
    # Body part confidence vs accuracy
    plt.subplot(1, 3, 1)
    confidence_bins = np.linspace(0, 1, 11)
    df['body_part_conf_bin'] = pd.cut(df['body_part_probability'], confidence_bins)
    bp_conf_acc = df.groupby('body_part_conf_bin')['is_body_part_correct'].mean()
    plt.plot(confidence_bins[:-1] + 0.05, bp_conf_acc.values, 'o-')
    plt.xlabel('Body Part Confidence')
    plt.ylabel('Accuracy')
    plt.title('Body Part Confidence vs Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Action confidence vs accuracy
    plt.subplot(1, 3, 2)
    df['action_conf_bin'] = pd.cut(df['action_probability'], confidence_bins)
    action_conf_acc = df.groupby('action_conf_bin')['is_action_correct'].mean()
    plt.plot(confidence_bins[:-1] + 0.05, action_conf_acc.values, 'o-')
    plt.xlabel('Action Confidence')
    plt.ylabel('Accuracy')
    plt.title('Action Confidence vs Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Severity confidence vs accuracy
    plt.subplot(1, 3, 3)
    df['severity_conf_bin'] = pd.cut(df['severity_probability'], confidence_bins)
    severity_conf_acc = df.groupby('severity_conf_bin')['is_severity_correct'].mean()
    plt.plot(confidence_bins[:-1] + 0.05, severity_conf_acc.values, 'o-')
    plt.xlabel('Severity Confidence')
    plt.ylabel('Accuracy')
    plt.title('Severity Confidence vs Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('confidence_vs_accuracy.png')
    plt.close()
    
    # Analyze confusion patterns
    # Action confusion heatmap
    action_cm = confusion_matrix(
        df['true_action'], 
        df['pred_action'], 
        labels=list(range(len(config.action_classes)))
    )
    action_cm_norm = action_cm.astype('float') / action_cm.sum(axis=1)[:, np.newaxis]
    action_cm_norm = np.nan_to_num(action_cm_norm)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        action_cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=config.action_classes,
        yticklabels=config.action_classes
    )
    plt.xlabel('Predicted Action')
    plt.ylabel('True Action')
    plt.title('Action Confusion Matrix (Normalized)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('action_confusion_matrix_detailed.png')
    plt.close()
    
    # Severity confusion heatmap
    severity_cm = confusion_matrix(
        df['true_severity'], 
        df['pred_severity'], 
        labels=list(range(len(config.severity_classes)))
    )
    severity_cm_norm = severity_cm.astype('float') / severity_cm.sum(axis=1)[:, np.newaxis]
    severity_cm_norm = np.nan_to_num(severity_cm_norm)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        severity_cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=config.severity_classes,
        yticklabels=config.severity_classes
    )
    plt.xlabel('Predicted Severity')
    plt.ylabel('True Severity')
    plt.title('Severity Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig('severity_confusion_matrix_detailed.png')
    plt.close()
    
    # Analyze misclassified examples
    misclassified = df[~df['is_action_correct']]
    print(f"\nNumber of misclassified action examples: {len(misclassified)}")
    
    # Top confusion pairs
    confusion_pairs = misclassified.groupby(['true_action_name', 'pred_action_name']).size().reset_index()
    confusion_pairs.columns = ['True Action', 'Predicted Action', 'Count']
    confusion_pairs = confusion_pairs.sort_values('Count', ascending=False)
    
    print("\nTop confusion pairs (True -> Predicted):")
    print(confusion_pairs.head(10))
    
    # Find difficult examples (consistently misclassified with high confidence)
    difficult_examples = misclassified[misclassified['action_probability'] > 0.8]
    print(f"\nNumber of high-confidence misclassifications: {len(difficult_examples)}")
    
    if len(difficult_examples) > 0:
        print("\nSample of high-confidence misclassifications:")
        sample_difficult = difficult_examples.sample(min(5, len(difficult_examples)))
        for _, row in sample_difficult.iterrows():
            print(f"True: {row['true_action_name']} | Predicted: {row['pred_action_name']} (Conf: {row['action_probability']:.2f})")
    
    return {
        'body_part_accuracy': body_part_acc,
        'action_accuracy': action_acc,
        'severity_accuracy': severity_acc,
        'full_accuracy': full_acc,
        'action_accuracy_by_class': action_accuracy.to_dict(),
        'severity_accuracy_by_class': severity_accuracy.to_dict(),
        'top_confusion_pairs': confusion_pairs.head(10).to_dict()
    }

def analyze_errors_by_attribute(predictions, config):
    """Analyze error patterns by different attributes"""
    df = pd.DataFrame(predictions)
    
    # Error analysis by body part
    body_part_errors = df.groupby(['true_body_part_name', 'is_action_correct']).size().unstack()
    body_part_errors.columns = ['Incorrect', 'Correct']
    body_part_error_rate = body_part_errors['Incorrect'] / (body_part_errors['Incorrect'] + body_part_errors['Correct'])
    
    print("\nAction error rate by body part:")
    print(body_part_error_rate)
    
    # Error analysis by severity
    severity_errors = df.groupby(['true_severity_name', 'is_action_correct']).size().unstack()
    severity_errors.columns = ['Incorrect', 'Correct']
    severity_error_rate = severity_errors['Incorrect'] / (severity_errors['Incorrect'] + severity_errors['Correct'])
    
    print("\nAction error rate by severity:")
    print(severity_error_rate)
    
    # Plot error rates
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    body_part_error_rate.plot(kind='bar')
    plt.title('Action Error Rate by Body Part')
    plt.ylabel('Error Rate')
    plt.ylim(0, 1)
    
    plt.subplot(1, 2, 2)
    severity_error_rate.plot(kind='bar')
    plt.title('Action Error Rate by Severity')
    plt.ylabel('Error Rate')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('error_rates_by_attribute.png')
    plt.close()
    
    # Analyze which classes are most confused with each other
    action_pairs = []
    for action in config.action_classes:
        # Get samples where this was the true class
        true_samples = df[df['true_action_name'] == action]
        if len(true_samples) == 0:
            continue
        
        # Get most common misclassifications
        if len(true_samples) > 1:  # Check if we have multiple samples
            misclassified = true_samples[true_samples['true_action_name'] != true_samples['pred_action_name']]
            if len(misclassified) > 0:
                most_common = misclassified['pred_action_name'].value_counts().index[0]
                count = misclassified['pred_action_name'].value_counts().iloc[0]
                
                action_pairs.append({
                    'true_action': action,
                    'confused_with': most_common,
                    'count': count,
                    'error_rate': count / len(true_samples)
                })
    
    # Sort by error rate
    action_pairs = sorted(action_pairs, key=lambda x: x['error_rate'], reverse=True)
    
    print("\nMost confused action pairs:")
    for pair in action_pairs[:5]:
        print(f"{pair['true_action']} confused with {pair['confused_with']} "
              f"({pair['count']} times, {pair['error_rate']:.2f} error rate)")
    
    # Plot confusion pairs
    if len(action_pairs) > 0:
        plt.figure(figsize=(12, 6))
        actions = [f"{p['true_action']} → {p['confused_with']}" for p in action_pairs[:10]]
        error_rates = [p['error_rate'] for p in action_pairs[:10]]
        
        bars = plt.barh(actions, error_rates)
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                     f"{action_pairs[i]['count']}", va='center')
        
        plt.xlim(0, 1)
        plt.xlabel('Error Rate')
        plt.title('Top Confused Action Pairs')
        plt.tight_layout()
        plt.savefig('top_confused_pairs.png')
        plt.close()
    
    return {
        'body_part_error_rate': body_part_error_rate.to_dict(),
        'severity_error_rate': severity_error_rate.to_dict(),
        'most_confused_pairs': action_pairs[:5]
    }

def main():
    """Main function to evaluate the model"""
    parser = argparse.ArgumentParser(description='Evaluate Soccer Foul Classification model')
    parser.add_argument('--model_path', default='best_mvfoul_model.pth', help='Path to trained model')
    parser.add_argument('--data_path', default='./soccernet/fouls/mvfouls', help='Path to dataset')
    parser.add_argument('--output_dir', default='model_evaluation', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Configuration
    config = Config()
    config.base_path = Path(args.data_path)
    config.batch_size = args.batch_size
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    
    # Load model
    model = load_model(args.model_path, config)
    
    # Create test dataset
    test_transform = get_transforms(train=False)
    test_dataset = SoccerNetMVFoulsDataset(
        config.base_path, 
        split='test',
        transform=test_transform,
        config=config
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Evaluate model
    print("Evaluating model...")
    predictions = evaluate_model(model, test_loader, config)
    
    # Analyze performance
    print("\nAnalyzing performance by class...")
    performance_stats = analyze_performance_by_class(predictions, config)
    
    print("\nAnalyzing error patterns...")
    error_stats = analyze_errors_by_attribute(predictions, config)
    
    # Save predictions and statistics
    with open('model_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    
    with open('performance_statistics.json', 'w') as f:
        json.dump({
            'performance_stats': performance_stats,
            'error_stats': error_stats
        }, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    import argparse
    main()