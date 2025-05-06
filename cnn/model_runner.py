"""
Training script for the Soccer Foul Classification model.
This script handles training, validation, and testing of the model.
"""

import os
import argparse
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report

from model import MVFoulClassifier, SoccerNetMVFoulsDataset, HierarchicalFoulLoss, Config
from utils import compute_class_weights, plot_training_history, plot_confusion_matrix

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Soccer Foul Classification model')
    
    # Data paths
    parser.add_argument('--base_path', type=str, help='Path to SoccerNet-MVFouls dataset')
    parser.add_argument('--model_save_path', type=str, help='Path to save trained models')
    parser.add_argument('--results_path', type=str, help='Path to save results')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for optimizer')
    parser.add_argument('--early_stopping_patience', type=int, help='Patience for early stopping')
    
    # Model parameters
    parser.add_argument('--num_frames', type=int, help='Number of frames to sample from each video')
    parser.add_argument('--frame_height', type=int, help='Frame height for preprocessing')
    parser.add_argument('--frame_width', type=int, help='Frame width for preprocessing')
    parser.add_argument('--max_clips', type=int, help='Maximum number of clips per action')
    
    # Execution settings
    parser.add_argument('--num_workers', type=int, help='Number of workers for data loading')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training if available')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on test set')
    parser.add_argument('--config_path', type=str, help='Path to config file')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    if args.config_path and os.path.exists(args.config_path):
        config = Config.load(args.config_path)
        config.update_from_args(args)
    else:
        config = Config()
        config.update_from_args(args)
    
    print(f"Using device: {config.device}")
    
    # Create datasets with appropriate transforms
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=(config.frame_height, config.frame_width), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.frame_height, config.frame_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SoccerNetMVFoulsDataset(
        config.base_path, 
        split='train',
        transform=train_transform
    )
    
    val_dataset = SoccerNetMVFoulsDataset(
        config.base_path, 
        split='valid',
        transform=val_transform
    )
    
    test_dataset = SoccerNetMVFoulsDataset(
        config.base_path, 
        split='test',
        transform=val_transform
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
    
    # Calculate class weights if not provided
    if args.config_path is None or not os.path.exists(args.config_path):
        print("Computing class weights from dataset...")
        action_weights, severity_weights, body_part_weights = compute_class_weights(train_dataset)
        config.action_weights = action_weights
        config.severity_weights = severity_weights
        config.body_part_weights = body_part_weights
    
    # Create model
    model = MVFoulClassifier(
        num_classes=len(config.action_classes),
        num_severity_classes=len(config.severity_classes)
    ).to(config.device)
    
    # Create loss function and optimizer
    criterion = HierarchicalFoulLoss(
        action_weights=config.action_weights.to(config.device), 
        severity_weights=config.severity_weights.to(config.device)
    )
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience, 
        verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_f1 = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=config.device)
            start_epoch = checkpoint['epoch'] + 1
            best_val_f1 = checkpoint['val_f1']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Save configuration
    os.makedirs(config.results_path, exist_ok=True)
    config.save(os.path.join(config.results_path, 'config.json'))
    
    # Evaluate mode - only run on test set
    if args.evaluate:
        print("Evaluating model on test set...")
        if args.resume is None:
            print("Warning: No checkpoint specified for evaluation. Using randomly initialized model.")
        
        test_results = evaluate(model, test_loader, criterion, config)
        
        # Save test results
        with open(os.path.join(config.results_path, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
        
        return
    
    # Training mode
    print("Starting training...")
    history = train(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler,
        config.num_epochs,
        config.device,
        config.model_save_path,
        best_val_f1,
        config.early_stopping_patience,
        start_epoch
    )
    
    # Save training history
    with open(os.path.join(config.results_path, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot training history
    plot_training_history(history, os.path.join(config.results_path, 'training_history.png'))
    
    # Evaluate on test set
    print("Evaluating on test set...")
    model_path = os.path.join(config.model_save_path, 'best_model.pth')
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate(model, test_loader, criterion, config)
    
    # Save test results
    with open(os.path.join(config.results_path, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, 
          save_path, best_val_f1=0.0, early_stopping_patience=5, start_epoch=0):
    """Train the model with early stopping"""
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_body_part_f1': [],
        'val_action_f1': [],
        'val_severity_f1': [],
        'lr': []
    }
    
    # Create directory for saving models
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize early stopping variables
    patience_counter = 0
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
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
        
        train_loss /= len(train_loader)
        
        # Validation phase
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['action_f1'])
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_body_part_f1'].append(val_metrics['body_part_f1'])
        history['val_action_f1'].append(val_metrics['action_f1'])
        history['val_severity_f1'].append(val_metrics['severity_f1'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Body Part F1: {val_metrics['body_part_f1']:.4f}")
        print(f"Val Action F1: {val_metrics['action_f1']:.4f}")
        print(f"Val Severity F1: {val_metrics['severity_f1']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check if this is the best model
        if val_metrics['action_f1'] > best_val_f1:
            best_val_f1 = val_metrics['action_f1']
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_metrics': val_metrics,
            }, os.path.join(save_path, 'best_model.pth'))
            
            print(f"Saved new best model with F1: {best_val_f1:.4f}")
            
            # Reset patience counter
            patience_counter = 0
        else:
            # Increment patience counter
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs. Best F1: {best_val_f1:.4f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_metrics['action_f1'],
            'val_metrics': val_metrics,
        }, os.path.join(save_path, 'last_model.pth'))
    
    return history

def validate(model, val_loader, criterion, device):
    """Validate the model and compute metrics"""
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
            action_pred[body_part_pred == 1] = lower_body_pred[body_part_pred == 1] + 5
            
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
        'body_part_f1': float(body_part_f1),
        'action_f1': float(action_f1),
        'severity_f1': float(severity_f1)
    }
    
    return val_loss, metrics

def evaluate(model, test_loader, criterion, config):
    """Evaluate model on test set"""
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
        for batch in test_loader:
            # Move data to device
            clips = batch['clips'].to(config.device)
            replay_speeds = batch['replay_speeds'].to(config.device)
            num_clips = batch['num_clips'].to(config.device)
            action_label = batch['action_label'].to(config.device)
            severity_label = batch['severity_label'].to(config.device)
            body_part_label = batch['body_part_label'].to(config.device)
            
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
    
    # Generate detailed classification reports
    body_part_report = classification_report(
        body_part_targets, body_part_preds, 
        target_names=config.body_part_classes, 
        output_dict=True
    )
    
    action_report = classification_report(
        action_targets, action_preds, 
        target_names=config.action_classes, 
        output_dict=True
    )
    
    severity_report = classification_report(
        severity_targets, severity_preds, 
        target_names=config.severity_classes, 
        output_dict=True
    )
    
    # Generate confusion matrices
    plot_confusion_matrix(
        body_part_targets, body_part_preds, 
        config.body_part_classes,
        os.path.join(config.results_path, 'body_part_confusion_matrix.png'),
        "Body Part Classification"
    )
    
    plot_confusion_matrix(
        action_targets, action_preds, 
        config.action_classes,
        os.path.join(config.results_path, 'action_confusion_matrix.png'),
        "Action Classification"
    )
    
    plot_confusion_matrix(
        severity_targets, severity_preds, 
        config.severity_classes,
        os.path.join(config.results_path, 'severity_confusion_matrix.png'),
        "Severity Classification"
    )
    
    # Return metrics and reports
    results = {
        'test_loss': float(test_loss),
        'body_part_f1': float(body_part_f1),
        'action_f1': float(action_f1),
        'severity_f1': float(severity_f1),
        'body_part_report': body_part_report,
        'action_report': action_report,
        'severity_report': severity_report
    }
    
    return results

if __name__ == "__main__":
    main()