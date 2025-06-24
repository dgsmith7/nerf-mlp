#!/usr/bin/env python3
"""
Real-time training progress plotting script.
This script can be run during training to monitor progress by reading the metrics files.
"""

import json
import os
import time
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def load_metrics(metrics_file):
    """Load metrics from JSON file."""
    if not os.path.exists(metrics_file):
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None

def create_progress_plot(metrics, save_dir, show_plot=True):
    """Create a comprehensive training progress plot."""
    if not metrics:
        print("No metrics data available.")
        return
    
    # Extract data
    train_losses = metrics.get('train_losses', [])
    train_psnrs = metrics.get('train_psnrs', [])
    quick_val_losses = metrics.get('quick_val_losses', [])
    quick_val_psnrs = metrics.get('quick_val_psnrs', [])
    quick_val_ssims = metrics.get('quick_val_ssims', [])
    val_steps = metrics.get('val_steps', [])
    iteration_times = metrics.get('iteration_times', [])
    
    if not val_steps:
        print("No validation steps found in metrics.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NeRF Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Loss convergence (top left, larger)
    ax1 = axes[0, 0]
    if train_losses and quick_val_losses:
        ax1.plot(val_steps, train_losses, label='Training Loss', marker='o', markersize=3, linewidth=1.5, color='blue', alpha=0.8)
        ax1.plot(val_steps, quick_val_losses, label='Validation Loss', marker='s', markersize=3, linewidth=1.5, color='red', alpha=0.8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
    
    # 2. PSNR convergence (top middle)
    ax2 = axes[0, 1]
    if train_psnrs and quick_val_psnrs:
        ax2.plot(val_steps, train_psnrs, label='Training PSNR', marker='o', markersize=3, linewidth=1.5, color='green', alpha=0.8)
        ax2.plot(val_steps, quick_val_psnrs, label='Validation PSNR', marker='s', markersize=3, linewidth=1.5, color='orange', alpha=0.8)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Training vs Validation PSNR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. SSIM progress (top right)
    ax3 = axes[0, 2]
    if quick_val_ssims:
        ax3.plot(val_steps, quick_val_ssims, label='Validation SSIM', marker='s', markersize=3, color='purple', alpha=0.8)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('SSIM')
        ax3.set_title('SSIM Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Overfitting indicator (bottom left)
    ax4 = axes[1, 0]
    if train_losses and quick_val_losses and len(train_losses) == len(quick_val_losses):
        loss_diff = [abs(t - v) for t, v in zip(train_losses, quick_val_losses)]
        ax4.plot(val_steps, loss_diff, marker='o', markersize=3, color='purple', alpha=0.8)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('|Train - Val Loss|')
        ax4.set_title('Overfitting Indicator')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    
    # 5. Training time per iteration (bottom middle)
    ax5 = axes[1, 1]
    if iteration_times and len(iteration_times) > 10:
        recent_times = iteration_times[-100:]  # Last 100 iterations
        ax5.plot(recent_times, alpha=0.6, color='brown')
        ax5.set_xlabel('Recent Iterations')
        ax5.set_ylabel('Time (seconds)')
        ax5.set_title('Training Time per Iteration')
        ax5.grid(True, alpha=0.3)
    
    # 6. Convergence summary (bottom right)
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate summary statistics
    if train_losses and quick_val_losses and train_psnrs and quick_val_psnrs:
        current_train_loss = train_losses[-1] if train_losses else 0
        current_val_loss = quick_val_losses[-1] if quick_val_losses else 0
        current_train_psnr = train_psnrs[-1] if train_psnrs else 0
        current_val_psnr = quick_val_psnrs[-1] if quick_val_psnrs else 0
        
        # Calculate improvement
        if len(train_losses) > 1:
            loss_improvement = ((train_losses[0] - current_train_loss) / train_losses[0]) * 100
        else:
            loss_improvement = 0
        
        if len(train_psnrs) > 1:
            psnr_improvement = current_train_psnr - train_psnrs[0]
        else:
            psnr_improvement = 0
        
        summary_text = f"""
Current Status:
• Iteration: {val_steps[-1] if val_steps else 0}
• Training Loss: {current_train_loss:.6f}
• Validation Loss: {current_val_loss:.6f}
• Training PSNR: {current_train_psnr:.2f} dB
• Validation PSNR: {current_val_psnr:.2f} dB

Improvement:
• Loss: {loss_improvement:.1f}% reduction
• PSNR: {psnr_improvement:+.2f} dB gain

Best Validation PSNR: {metrics.get('best_val_psnr', 0):.2f} dB
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if save_dir:
        plot_path = os.path.join(save_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Progress plot saved to: {plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def animate_progress(metrics_file, save_dir, interval=5000):
    """Create an animated plot that updates automatically."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NeRF Training Progress (Live)', fontsize=16, fontweight='bold')
    
    def update(frame):
        # Clear all axes
        for ax in axes.flat:
            ax.clear()
        
        # Load latest metrics
        metrics = load_metrics(metrics_file)
        if not metrics:
            return []
        
        # Recreate the plot
        create_progress_plot(metrics, None, show_plot=False)
        
        # Update the current figure
        plt.draw()
        return []
    
    # Create animation
    ani = FuncAnimation(fig, update, interval=interval, repeat=True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot NeRF training progress')
    parser.add_argument('--metrics-file', type=str, 
                       default='outputs/checkpoints/metrics_latest.json',
                       help='Path to metrics JSON file')
    parser.add_argument('--save-dir', type=str, 
                       default='outputs/checkpoints',
                       help='Directory to save plots')
    parser.add_argument('--live', action='store_true',
                       help='Run in live mode with automatic updates')
    parser.add_argument('--interval', type=int, default=5000,
                       help='Update interval in milliseconds (for live mode)')
    
    args = parser.parse_args()
    
    if args.live:
        print(f"Starting live monitoring of {args.metrics_file}")
        print("Press Ctrl+C to stop")
        try:
            animate_progress(args.metrics_file, args.save_dir, args.interval)
        except KeyboardInterrupt:
            print("\nLive monitoring stopped.")
    else:
        # Single plot
        metrics = load_metrics(args.metrics_file)
        create_progress_plot(metrics, args.save_dir)

if __name__ == '__main__':
    main() 