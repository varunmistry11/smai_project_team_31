import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def create_combined_visualizations():
    """Create combined visualizations for CNN and MViT model performance"""
    
    # Load CNN results
    cnn_multiview = pd.read_csv('cnn/results/multiview_comparison.csv')
    cnn_fusion = pd.read_csv('cnn/results/fusion_strategy_comparison.csv')
    cnn_hierarchy = pd.read_csv('cnn/results/hierarchy_comparison.csv')
    cnn_replay = pd.read_csv('cnn/results/replay_speed_ablation.csv')
    
    # Load MViT results
    mvit_multiview = pd.read_csv('transformer/results/multiview_comparison.csv')
    mvit_fusion = pd.read_csv('transformer/results/fusion_strategy_comparison.csv')
    mvit_hierarchy = pd.read_csv('transformer/results/hierarchy_comparison.csv')
    mvit_replay = pd.read_csv('transformer/results/replay_speed_ablation.csv')
    
    # Create directory for combined results
    os.makedirs('combined_results', exist_ok=True)
    
    # 1. Multi-view performance comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(cnn_multiview['View Configuration']))
    width = 0.35
    
    plt.bar(x - width/2, cnn_multiview['Foul Type Accuracy'], width, label='CNN Model')
    plt.bar(x + width/2, mvit_multiview['Foul Type Accuracy'], width, label='MViT Model')
    
    plt.xlabel('Camera Views')
    plt.ylabel('Foul Type Accuracy (%)')
    plt.title('Impact of Multiple Views on Foul Classification Accuracy')
    plt.xticks(x, cnn_multiview['View Configuration'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('combined_results/multiview_comparison.png', dpi=300)
    
    # 2. Hierarchical vs flat classification
    plt.figure(figsize=(10, 5))
    x = np.arange(len(cnn_hierarchy['Classification Approach']))
    width = 0.35
    
    plt.bar(x - width/2, cnn_hierarchy['Foul Type Accuracy'], width, label='CNN Model')
    plt.bar(x + width/2, mvit_hierarchy['Foul Type Accuracy'], width, label='MViT Model')
    
    plt.xlabel('Classification Approach')
    plt.ylabel('Foul Type Accuracy (%)')
    plt.title('Hierarchical vs. Flat Classification Performance')
    plt.xticks(x, cnn_hierarchy['Classification Approach'], rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig('combined_results/hierarchy_comparison.png', dpi=300)
    
    # 3. Fusion strategy comparison
    plt.figure(figsize=(10, 5))
    x = np.arange(len(cnn_fusion['Fusion Method']))
    width = 0.35
    
    plt.bar(x - width/2, cnn_fusion['Foul Type Accuracy'], width, label='CNN Model')
    plt.bar(x + width/2, mvit_fusion['Foul Type Accuracy'], width, label='MViT Model')
    
    plt.xlabel('Fusion Method')
    plt.ylabel('Foul Type Accuracy (%)')
    plt.title('Comparison of Multi-view Fusion Strategies')
    plt.xticks(x, cnn_fusion['Fusion Method'], rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig('combined_results/fusion_comparison.png', dpi=300)
    
    # 4. Replay speed impact
    plt.figure(figsize=(10, 5))
    x = np.arange(len(cnn_replay['Model Configuration']))
    width = 0.35
    
    plt.bar(x - width/2, cnn_replay['Severity Accuracy'], width, label='CNN Model')
    plt.bar(x + width/2, mvit_replay['Severity Accuracy'], width, label='MViT Model')
    
    plt.xlabel('Configuration')
    plt.ylabel('Severity Classification Accuracy (%)')
    plt.title('Impact of Replay Speed Feature on Severity Classification')
    plt.xticks(x, cnn_replay['Model Configuration'], rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig('combined_results/replay_speed_impact.png', dpi=300)
    
    # 5. Create summary table
    summary = {
        'Metric': [
            'Body Part Accuracy', 
            'Foul Type Accuracy', 
            'Severity Accuracy', 
            'Weighted F1 Score'
        ],
        'CNN': [79.23, 38.92, 67.41, 0.385],
        'MViT': [83.65, 45.77, 72.18, 0.462],
        'Improvement': ['+4.42%', '+6.85%', '+4.77%', '+0.077']
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('combined_results/model_comparison_summary.csv', index=False)
    
    print("Combined visualizations created in 'combined_results' directory")

if __name__ == '__main__':
    create_combined_visualizations()