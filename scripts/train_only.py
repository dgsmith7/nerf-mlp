#!/usr/bin/env python3
"""
Training-only script for NeRF MLP model.
This version removes all validation to test if the model can learn properly.
"""

import sys
import os
# Add the current directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from torch.utils.data import DataLoader
from nerfmlp import NeRFMLP, NeRFDataset, NeRFRenderer, auto_tune_batch_size
from datetime import datetime
import matplotlib.pyplot as plt
import time
import psutil
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(pred, target):
    """Calculate PSNR between predicted and target images"""
    with torch.no_grad():
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def get_gradient_norm(model):
    """Calculate the L2 norm of gradients"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def main():
    parser = argparse.ArgumentParser(description='Train NeRF MLP model (training only)')
    parser.add_argument('--datadir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--img_wh', nargs=2, type=int, default=[64, 64], help='Image width and height')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--iters', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='outputs/train_only', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Dataset and dataloader
    dataset = NeRFDataset(args.datadir, img_wh=args.img_wh)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    print(f'Total rays: {len(dataset)} | Batch size: {args.batch_size}')
    print(f'Target iterations: {args.iters:,}')
    
    # Model and renderer
    model = NeRFMLP().to(device)
    
    # Calculate dynamic near/far based on camera positions
    poses = dataset.poses
    positions = np.array([pose[:3, 3] for pose in poses])
    dists = np.linalg.norm(positions, axis=1)
    near = max(0.1, dists.min() - 0.5)
    far = dists.max() + 0.5
    print(f"Dynamic near: {near}, far: {far}")
    
    renderer = NeRFRenderer(model, device, near=near, far=far)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    # Metrics tracking
    train_losses = []
    train_psnrs = []
    iteration_times = []
    
    step = 0
    data_iter = iter(dataloader)
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Starting training...")
    
    while step < args.iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        iter_start_time = time.time()
        
        # Move batch to device
        rays_o = batch['ray_o'].to(device)
        rays_d = batch['ray_d'].to(device)
        target_rgbs = batch['rgb'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_rgbs = renderer._render_rays(rays_o, rays_d)['rgb_map']
        
        # Loss calculation
        loss = torch.mean((pred_rgbs - target_rgbs) ** 2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            current_psnr = calculate_psnr(pred_rgbs, target_rgbs)
        
        # Record metrics
        train_losses.append(loss.item())
        train_psnrs.append(current_psnr)
        iteration_times.append(time.time() - iter_start_time)
        
        # Print progress
        if step % 100 == 0:
            memory_usage = get_memory_usage()
            grad_norm = get_gradient_norm(model)
            current_time = time.time() - start_time
            
            # Calculate rolling median for time
            recent_times = iteration_times[-100:] if len(iteration_times) >= 100 else iteration_times
            median_time = float(np.median(recent_times))
            
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Iter {step:,} | "
                  f"Loss: {loss.item():.6f} | PSNR: {current_psnr:.2f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Grad: {grad_norm:.4f} | Mem: {memory_usage:.1f}GB | "
                  f"Time: {median_time:.3f}s (median)")
        
        # Save checkpoint every 1000 steps
        if step % 1000 == 0 and step > 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_{step}.pth'))
            
            # Save metrics
            metrics = {
                'step': step,
                'train_losses': train_losses,
                'train_psnrs': train_psnrs,
                'iteration_times': iteration_times,
                'config': {
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'total_iterations': args.iters,
                    'img_wh': args.img_wh
                }
            }
            
            import json
            def convert_for_json(obj):
                """Recursively convert numpy types to Python native types for JSON serialization."""
                if isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(v) for v in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                else:
                    return obj
            
            with open(os.path.join(args.save_dir, f'metrics_{step}.json'), 'w') as f:
                json.dump(convert_for_json(metrics), f, indent=2)
            
            print(f"💾 Checkpoint saved at iteration {step:,}")
        
        step += 1
    
    # Final save
    total_time = time.time() - start_time
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_final.pth'))
    print(f'🎉 Training complete in {total_time/3600:.2f} hours ({total_time:.0f} seconds)')
    
    # Plot training progress
    print("Generating training plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training Loss
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training PSNR
    ax2.plot(train_psnrs, label='Train PSNR', color='green')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Training PSNR')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_progress.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save final metrics
    final_metrics = {
        'train_losses': convert_for_json(train_losses),
        'train_psnrs': convert_for_json(train_psnrs),
        'iteration_times': convert_for_json(iteration_times),
        'total_training_time': float(total_time),
        'final_loss': float(train_losses[-1]) if train_losses else None,
        'final_psnr': float(train_psnrs[-1]) if train_psnrs else None,
        'config': {
            'batch_size': int(args.batch_size),
            'learning_rate': float(args.lr),
            'total_iterations': int(args.iters),
            'img_wh': [int(x) for x in args.img_wh]
        }
    }
    
    with open(os.path.join(args.save_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"Training plots and metrics saved to {args.save_dir}")
    print(f"Final Loss: {train_losses[-1]:.6f}")
    print(f"Final PSNR: {train_psnrs[-1]:.2f} dB")

if __name__ == '__main__':
    main() 