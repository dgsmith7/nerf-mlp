#!/usr/bin/env python3
"""
Quick script to view the latest training progress.
"""

import json
import os
import argparse
from datetime import datetime

def load_latest_metrics(metrics_dir):
    """Load the latest metrics file."""
    metrics_file = os.path.join(metrics_dir, 'metrics_latest.json')
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None

def display_progress(metrics):
    """Display training progress in a readable format."""
    if not metrics:
        return
    
    print("=" * 80)
    print("NERF TRAINING PROGRESS")
    print("=" * 80)
    
    # Basic info
    step = metrics.get('step', 0)
    best_psnr = metrics.get('best_val_psnr', 0)
    
    print(f"Current Iteration: {step:,}")
    print(f"Best Validation PSNR: {best_psnr:.2f} dB")
    
    # Latest metrics
    train_losses = metrics.get('train_losses', [])
    train_psnrs = metrics.get('train_psnrs', [])
    quick_val_losses = metrics.get('quick_val_losses', [])
    quick_val_psnrs = metrics.get('quick_val_psnrs', [])
    quick_val_ssims = metrics.get('quick_val_ssims', [])
    
    if train_losses and quick_val_losses:
        current_train_loss = train_losses[-1]
        current_val_loss = quick_val_losses[-1]
        current_train_psnr = train_psnrs[-1] if train_psnrs else 0
        current_val_psnr = quick_val_psnrs[-1] if quick_val_psnrs else 0
        current_ssim = quick_val_ssims[-1] if quick_val_ssims else 0
        
        print(f"\nLatest Metrics:")
        print(f"  Training Loss: {current_train_loss:.6f}")
        print(f"  Validation Loss: {current_val_loss:.6f}")
        print(f"  Training PSNR: {current_train_psnr:.2f} dB")
        print(f"  Validation PSNR: {current_val_psnr:.2f} dB")
        print(f"  Validation SSIM: {current_ssim:.4f}")
        
        # Calculate improvements
        if len(train_losses) > 1:
            loss_improvement = ((train_losses[0] - current_train_loss) / train_losses[0]) * 100
            print(f"\nImprovement:")
            print(f"  Loss reduction: {loss_improvement:.1f}%")
        
        if len(train_psnrs) > 1:
            psnr_improvement = current_train_psnr - train_psnrs[0]
            print(f"  PSNR gain: {psnr_improvement:+.2f} dB")
        
        # Overfitting check
        loss_gap = abs(current_train_loss - current_val_loss)
        print(f"  Train-Val Loss gap: {loss_gap:.6f}")
        
        if loss_gap > current_val_loss * 0.5:
            print("  ⚠️  Potential overfitting detected!")
    
    # Training time info
    iteration_times = metrics.get('iteration_times', [])
    if iteration_times:
        avg_time = sum(iteration_times[-100:]) / min(100, len(iteration_times))
        print(f"\nPerformance:")
        print(f"  Average iteration time: {avg_time:.3f} seconds")
        print(f"  Iterations per hour: {3600/avg_time:.1f}")
    
    # Configuration
    config = metrics.get('config', {})
    if config:
        print(f"\nConfiguration:")
        print(f"  Quick validation resolution: {config.get('quick_val_res', 'N/A')}")
        print(f"  Quick validation subset: {config.get('quick_val_subset', 'N/A')}")
        print(f"  Full validation resolution: {config.get('full_val_res', 'N/A')}")
        print(f"  Batch size: {config.get('batch_size', 'N/A')}")
        print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
        print(f"  Total iterations: {config.get('total_iterations', 'N/A')}")
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Quick view of training progress')
    parser.add_argument('--metrics-dir', type=str, 
                       default='outputs/lego_full_training',
                       help='Directory containing metrics files')
    
    args = parser.parse_args()
    
    metrics = load_latest_metrics(args.metrics_dir)
    display_progress(metrics)

if __name__ == '__main__':
    main() 