"""
Soccer Foul Classification Results Analyzer
This script analyzes the inference results JSON file and calculates performance metrics
by comparing predictions against ground truth data from annotations.json.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze Soccer Foul Classification results')
    parser.add_argument('--results_file', required=True, help='Path to inference results JSON file')
    parser.add_argument('--annotations_file', required=True, help='Path to ground truth annotations JSON file')
    parser.add_argument('--output_dir', default='analysis_results', help='Directory to save analysis results')
    return parser.parse_args()

def load_data(results_file, annotations_file):
    """Load results and ground truth data"""
    # Load inference results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load ground truth annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    return results, annotations

def prepare_data_for_analysis(results, annotations):
    """Prepare data for analysis by pairing predictions with ground truth"""
    data = []
    
    # Get annotations actions dictionary
    actions = annotations.get('Actions', {})
    
    # Process each action in the results
    for action_id, action_results in results.items():
        # Extract ID number (remove 'action_' prefix)
        action_num = action_id.replace('action_', '')
        
        # Check if this action exists in the ground truth
        if action_num in actions:
            ground_truth = actions[action_num]
            
            # Get ground truth labels
            true_action_class = ground_truth.get('Action class', '')
            true_severity = ground_truth.get('Severity', '')
            try:
                true_severity = float(true_severity) if true_severity else 0.0
            except (ValueError, TypeError):
                true_severity = 0.0
                
            true_body_part = ground_truth.get('Bodypart', '')
            if true_body_part == 'Under body':
                true_body_part = 'Lower body'
            
            # Process each clip result
            for clip_idx, result in enumerate(action_results):
                pred_body_part = result.get('body_part', {}).get('predicted', '')
                pred_action = result.get('action', {}).get('predicted', '')
                pred_severity = result.get('severity', {}).get('predicted', '')
                
                # Map severity name to numeric value
                severity_map = {'No foul': 0.0, 'Foul': 1.0, 'Yellow card': 2.0, 'Red card': 3.0}
                pred_severity_value = severity_map.get(pred_severity, 0.0)
                
                # Add to data list
                data.append({
                    'action_id': action_id,
                    'clip_idx': clip_idx,
                    'true_body_part': true_body_part,
                    'pred_body_part': pred_body_part,
                    'true_action': true_action_class,
                    'pred_action': pred_action,
                    'true_severity': true_severity,
                    'pred_severity_value': pred_severity_value,
                    'pred_severity': pred_severity,
                    'body_part_correct': true_body_part == pred_body_part,
                    'action_correct': true_action_class == pred_action,
                    'severity_correct': true_severity == pred_severity_value
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def calculate_metrics(df, output_dir):
    """Calculate performance metrics for each task"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store metrics
    metrics = {}
    
    # Overall accuracy
    metrics['body_part_accuracy'] = accuracy_score(df['true_body_part'], df['pred_body_part'])
    metrics['action_accuracy'] = accuracy_score(df['true_action'], df['pred_action'])
    metrics['severity_accuracy'] = accuracy_score(df['true_severity'], df['pred_severity_value'])
    
    # Weighted precision, recall, F1 (weighted to handle class imbalance)
    metrics['body_part_precision'] = precision_score(df['true_body_part'], df['pred_body_part'], average='weighted', zero_division=0)
    metrics['body_part_recall'] = recall_score(df['true_body_part'], df['pred_body_part'], average='weighted', zero_division=0)
    metrics['body_part_f1'] = f1_score(df['true_body_part'], df['pred_body_part'], average='weighted', zero_division=0)
    
    metrics['action_precision'] = precision_score(df['true_action'], df['pred_action'], average='weighted', zero_division=0)
    metrics['action_recall'] = recall_score(df['true_action'], df['pred_action'], average='weighted', zero_division=0)
    metrics['action_f1'] = f1_score(df['true_action'], df['pred_action'], average='weighted', zero_division=0)
    
    metrics['severity_precision'] = precision_score(df['true_severity'], df['pred_severity_value'], average='weighted', zero_division=0)
    metrics['severity_recall'] = recall_score(df['true_severity'], df['pred_severity_value'], average='weighted', zero_division=0)
    metrics['severity_f1'] = f1_score(df['true_severity'], df['pred_severity_value'], average='weighted', zero_division=0)
    
    # Print overall metrics
    print("\nOverall Performance Metrics:")
    print("===========================")
    print("\nBody Part Classification:")
    print(f"Accuracy: {metrics['body_part_accuracy']:.4f}")
    print(f"Precision: {metrics['body_part_precision']:.4f}")
    print(f"Recall: {metrics['body_part_recall']:.4f}")
    print(f"F1 Score: {metrics['body_part_f1']:.4f}")
    
    print("\nAction Classification:")
    print(f"Accuracy: {metrics['action_accuracy']:.4f}")
    print(f"Precision: {metrics['action_precision']:.4f}")
    print(f"Recall: {metrics['action_recall']:.4f}")
    print(f"F1 Score: {metrics['action_f1']:.4f}")
    
    print("\nSeverity Classification:")
    print(f"Accuracy: {metrics['severity_accuracy']:.4f}")
    print(f"Precision: {metrics['severity_precision']:.4f}")
    print(f"Recall: {metrics['severity_recall']:.4f}")
    print(f"F1 Score: {metrics['severity_f1']:.4f}")
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def generate_classification_reports(df, output_dir):
    """Generate detailed classification reports for each task"""
    # Body part classification report
    bp_report = classification_report(
        df['true_body_part'], 
        df['pred_body_part'],
        output_dict=True
    )
    
    # Action classification report
    action_report = classification_report(
        df['true_action'], 
        df['pred_action'],
        output_dict=True
    )
    
    # Severity classification report
    severity_report = classification_report(
        df['true_severity'], 
        df['pred_severity_value'],
        output_dict=True
    )
    
    # Save reports to file
    with open(os.path.join(output_dir, 'classification_reports.json'), 'w') as f:
        json.dump({
            'body_part': bp_report,
            'action': action_report,
            'severity': severity_report
        }, f, indent=4)
    
    # Print detailed reports
    print("\nDetailed Classification Reports:")
    print("===============================")
    
    print("\nBody Part Classification Report:")
    print(classification_report(df['true_body_part'], df['pred_body_part']))
    
    print("\nAction Classification Report:")
    print(classification_report(df['true_action'], df['pred_action']))
    
    print("\nSeverity Classification Report:")
    print(classification_report(df['true_severity'], df['pred_severity_value']))
    
    return bp_report, action_report, severity_report

def generate_confusion_matrices(df, output_dir):
    """Generate and save confusion matrices for each task"""
    # Body part confusion matrix
    plt.figure(figsize=(8, 6))
    body_part_cm = confusion_matrix(
        df['true_body_part'], 
        df['pred_body_part']
    )
    sns.heatmap(body_part_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=df['pred_body_part'].unique(),
                yticklabels=df['true_body_part'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Body Part Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'body_part_confusion_matrix.png'))
    plt.close()
    
    # Action confusion matrix
    plt.figure(figsize=(12, 10))
    action_cm = confusion_matrix(
        df['true_action'], 
        df['pred_action']
    )
    sns.heatmap(action_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=df['pred_action'].unique(),
                yticklabels=df['true_action'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Action Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_confusion_matrix.png'))
    plt.close()
    
    # Severity confusion matrix
    plt.figure(figsize=(8, 6))
    severity_cm = confusion_matrix(
        df['true_severity'], 
        df['pred_severity_value']
    )
    severity_labels = ['No foul (0.0)', 'Foul (1.0)', 'Yellow card (2.0)', 'Red card (3.0)']
    sns.heatmap(severity_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=severity_labels,
                yticklabels=severity_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Severity Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'severity_confusion_matrix.png'))
    plt.close()

def analyze_error_patterns(df, output_dir):
    """Analyze error patterns in the predictions"""
    # Create output directory
    os.makedirs(os.path.join(output_dir, 'error_analysis'), exist_ok=True)
    
    # Analyze action misclassifications
    action_errors = df[df['action_correct'] == False]
    action_error_counts = action_errors.groupby(['true_action', 'pred_action']).size().reset_index()
    action_error_counts.columns = ['True Action', 'Predicted Action', 'Count']
    action_error_counts = action_error_counts.sort_values('Count', ascending=False)
    
    # Save action error patterns
    action_error_counts.to_csv(os.path.join(output_dir, 'error_analysis', 'action_error_patterns.csv'), index=False)
    
    # Visualize top action misclassifications
    if len(action_error_counts) > 0:
        plt.figure(figsize=(12, 8))
        top_errors = action_error_counts.head(10)
        sns.barplot(x='Count', y=top_errors['True Action'] + ' → ' + top_errors['Predicted Action'], data=top_errors)
        plt.title('Top 10 Action Misclassifications')
        plt.xlabel('Count')
        plt.ylabel('True → Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_analysis', 'top_action_errors.png'))
        plt.close()
    
    # Analyze severity misclassifications
    severity_errors = df[df['severity_correct'] == False]
    severity_error_counts = severity_errors.groupby(['true_severity', 'pred_severity_value']).size().reset_index()
    severity_error_counts.columns = ['True Severity', 'Predicted Severity', 'Count']
    severity_error_counts = severity_error_counts.sort_values('Count', ascending=False)
    
    # Save severity error patterns
    severity_error_counts.to_csv(os.path.join(output_dir, 'error_analysis', 'severity_error_patterns.csv'), index=False)
    
    # Calculate performance by action class
    action_performance = df.groupby('true_action').agg({
        'action_correct': 'mean',
        'body_part_correct': 'mean',
        'severity_correct': 'mean'
    }).reset_index()
    
    action_performance.columns = ['Action Class', 'Action Accuracy', 'Body Part Accuracy', 'Severity Accuracy']
    action_performance = action_performance.sort_values('Action Accuracy')
    
    # Save action performance
    action_performance.to_csv(os.path.join(output_dir, 'error_analysis', 'action_performance.csv'), index=False)
    
    # Visualize action performance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Action Accuracy', y='Action Class', data=action_performance)
    plt.title('Action Classification Accuracy by Class')
    plt.xlabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis', 'action_accuracy_by_class.png'))
    plt.close()
    
    # Return summary
    return {
        'action_error_patterns': action_error_counts.to_dict('records'),
        'severity_error_patterns': severity_error_counts.to_dict('records'),
        'action_performance': action_performance.to_dict('records')
    }

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load data
    print(f"Loading results from {args.results_file}")
    print(f"Loading ground truth from {args.annotations_file}")
    results, annotations = load_data(args.results_file, args.annotations_file)
    
    # Prepare data for analysis
    print("Preparing data for analysis...")
    df = prepare_data_for_analysis(results, annotations)
    
    # Save processed dataframe
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, 'processed_results.csv'), index=False)
    
    print(f"Analyzed {len(df)} predictions")
    
    # Calculate metrics
    print("Calculating performance metrics...")
    metrics = calculate_metrics(df, args.output_dir)
    
    # Generate classification reports
    print("Generating classification reports...")
    reports = generate_classification_reports(df, args.output_dir)
    
    # Generate confusion matrices
    print("Generating confusion matrices...")
    generate_confusion_matrices(df, args.output_dir)
    
    # Analyze error patterns
    print("Analyzing error patterns...")
    error_analysis = analyze_error_patterns(df, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()