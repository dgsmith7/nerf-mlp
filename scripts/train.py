import sys
import os
# Add the current directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from torch.utils.data import DataLoader
from nerfmlp import NeRFMLP, NeRFDataset, NeRFRenderer, auto_tune_batch_size
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import psutil
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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

def calculate_psnr(pred, target):
    """Calculate PSNR between predicted and target images"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    return psnr(target_np, pred_np, data_range=1.0)

def calculate_ssim(pred, target):
    """Calculate SSIM between predicted and target images"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Handle small images by adjusting window size
    min_dim = min(pred_np.shape[0], pred_np.shape[1])
    win_size = min(7, min_dim) if min_dim >= 7 else 3
    
    try:
        # Try with channel_axis parameter (newer scikit-image versions)
        return ssim(target_np, pred_np, data_range=1.0, win_size=win_size, channel_axis=-1)
    except TypeError:
        # Fallback to multichannel parameter (older versions)
        return ssim(target_np, pred_np, data_range=1.0, win_size=win_size, multichannel=True)

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

def format_time_duration(seconds):
    """Format time duration in a human-readable format"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def calculate_etc(current_step, total_steps, start_time, iteration_times):
    """Calculate estimated time of completion using robust statistics."""
    if not iteration_times or current_step == 0:
        return None
    # Use rolling median to avoid outliers
    recent_times = iteration_times[-100:]
    median_iter_time = float(np.median(recent_times))
    mean_iter_time = float(np.mean(recent_times))
    # If mean is much higher than median, warn about instability
    eta_unstable = mean_iter_time > 3 * median_iter_time
    avg_iter_time = median_iter_time
    remaining_iterations = total_steps - current_step
    estimated_remaining_seconds = remaining_iterations * avg_iter_time
    completion_time = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
    return {
        'remaining_time': estimated_remaining_seconds,
        'completion_time': completion_time,
        'avg_iter_time': avg_iter_time,
        'progress_percent': (current_step / total_steps) * 100,
        'eta_unstable': eta_unstable,
        'mean_iter_time': mean_iter_time,
        'median_iter_time': median_iter_time
    }

def validate(model, renderer, val_loader, device, subset_size=None):
    """
    Validate model with optional subset sampling for faster validation
    Returns comprehensive metrics
    """
    model.eval()
    val_loss = 0
    val_psnr = 0
    val_ssim = 0
    count = 0
    subset_count = 0
    with torch.no_grad():
        for batch in val_loader:
            if subset_size is not None and subset_count >= subset_size:
                break
            ray_o = batch['ray_o'].to(device)
            ray_d = batch['ray_d'].to(device)
            target_rgb = batch['rgb'].to(device)
            rgb_pred = renderer._render_rays(ray_o, ray_d)['rgb_map']
            # Calculate metrics
            loss = torch.mean((rgb_pred - target_rgb) ** 2)
            batch_psnr = calculate_psnr(rgb_pred, target_rgb)
            batch_ssim = calculate_ssim(rgb_pred, target_rgb)
            val_loss += loss.item() * ray_o.shape[0]
            val_psnr += batch_psnr * ray_o.shape[0]
            val_ssim += batch_ssim * ray_o.shape[0]
            count += ray_o.shape[0]
            subset_count += 1
    model.train()
    return {
        'loss': val_loss / count if count > 0 else 0,
        'psnr': val_psnr / count if count > 0 else 0,
        'ssim': val_ssim / count if count > 0 else 0
    }

def main():
    parser = argparse.ArgumentParser(description='Train NeRF MLP')
    parser.add_argument('--datadir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='train', help='Dataset split (train/val/test)')
    parser.add_argument('--img_wh', type=int, nargs=2, default=[1024, 1024], help='Image width and height (e.g., 1024 1024 for high-res, 400 400 for fast)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size (default 256 for 1024x1024, increase for lower res)')
    parser.add_argument('--iters', type=int, default=200000, help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='outputs/checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--quick_val_interval', type=int, default=1000, help='Quick validation every N iterations (default: 1000)')
    parser.add_argument('--full_val_interval', type=int, default=10000, help='Full validation every N iterations (default: 10000)')
    parser.add_argument('--quick_val_res', type=int, nargs=2, default=[256, 256], help='Quick validation resolution (default: 256 256)')
    parser.add_argument('--quick_val_subset', type=int, default=10, help='Number of images for quick validation (default: 10)')
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load datasets
    dataset = NeRFDataset(args.datadir, split=args.split, img_wh=tuple(args.img_wh))
    val_dataset = NeRFDataset(args.datadir, split='val', img_wh=tuple(args.img_wh))
    
    # Quick validation dataset (lower resolution, subset)
    quick_val_dataset = NeRFDataset(args.datadir, split='val', img_wh=tuple(args.quick_val_res))
    
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    quick_val_loader = DataLoader(quick_val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"Total rays: {len(dataset)} | Batch size: {batch_size}")
    print(f"Quick validation: {args.quick_val_res[0]}x{args.quick_val_res[1]} resolution, {args.quick_val_subset} images")
    print(f"Full validation: {args.img_wh[0]}x{args.img_wh[1]} resolution, all images")
    print(f"Target iterations: {args.iters:,}")

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    step = 0
    train_losses = []
    train_psnrs = []
    quick_val_losses = []
    quick_val_psnrs = []
    quick_val_ssims = []
    full_val_losses = []
    full_val_psnrs = []
    full_val_ssims = []
    val_steps = []
    best_val_loss = float('inf')
    best_val_psnr = 0
    running_train_loss = 0
    running_train_psnr = 0
    running_train_count = 0
    
    # Timing and performance tracking
    start_time = time.time()
    iteration_times = []
    summary_interval = 1000  # Print summary every 1000 steps
    last_val_loss = None
    last_val_psnr = None
    last_val_step = None
    training_step_times = []  # Only training step times (exclude validation/checkpoint)
    # Add summary interval counters
    summary_train_loss = 0
    summary_train_psnr = 0
    summary_train_count = 0
    
    while step < args.iters:
        iter_start_time = time.time()
        
        for batch in dataloader:
            ray_o = batch['ray_o'].to(device)
            ray_d = batch['ray_d'].to(device)
            target_rgb = batch['rgb'].to(device)
            
            # Render
            rgb_pred = renderer._render_rays(ray_o, ray_d)['rgb_map']
            loss = torch.mean((rgb_pred - target_rgb) ** 2)
            
            # Calculate training metrics
            batch_psnr = calculate_psnr(rgb_pred, target_rgb)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Calculate gradient norm before optimizer step
            grad_norm = get_gradient_norm(model)
            
            optimizer.step()
            
            running_train_loss += loss.item() * ray_o.shape[0]
            running_train_psnr += batch_psnr * ray_o.shape[0]
            running_train_count += ray_o.shape[0]
            
            # Update summary counters
            summary_train_loss += loss.item() * ray_o.shape[0]
            summary_train_psnr += batch_psnr * ray_o.shape[0]
            summary_train_count += ray_o.shape[0]
            
            iter_time = time.time() - iter_start_time
            iteration_times.append(iter_time)
            training_step_times.append(iter_time)
            
            if step % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                memory_gb = get_memory_usage()
                avg_iter_time = np.median(training_step_times[-100:]) if training_step_times else 0
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Iter {step:,} | "
                      f"Loss: {loss.item():.6f} | PSNR: {batch_psnr:.2f} | "
                      f"LR: {current_lr:.2e} | Grad: {grad_norm:.4f} | "
                      f"Mem: {memory_gb:.1f}GB | Time: {avg_iter_time:.3f}s (median)")
            
            # Quick validation for monitoring progress
            if step % args.quick_val_interval == 0 and step > 0:
                avg_train_loss = running_train_loss / running_train_count
                avg_train_psnr = running_train_psnr / running_train_count
                train_losses.append(avg_train_loss)
                train_psnrs.append(avg_train_psnr)
                running_train_loss = 0
                running_train_psnr = 0
                running_train_count = 0
                
                quick_metrics = validate(model, renderer, quick_val_loader, device, args.quick_val_subset)
                quick_val_losses.append(quick_metrics['loss'])
                quick_val_psnrs.append(quick_metrics['psnr'])
                quick_val_ssims.append(quick_metrics['ssim'])
                val_steps.append(step)
                last_val_loss = quick_metrics['loss']
                last_val_psnr = quick_metrics['psnr']
                last_val_step = step
                
                # Calculate ETC
                etc_info = calculate_etc(step, args.iters, start_time, iteration_times)
                # Convergence score: % improvement in quick val loss/PSNR over last 5 validations
                convergence_str = ""
                if len(quick_val_losses) > 5:
                    prev_loss = quick_val_losses[-6]
                    prev_psnr = quick_val_psnrs[-6]
                    loss_impr = 100 * (prev_loss - quick_metrics['loss']) / (abs(prev_loss) + 1e-8)
                    psnr_impr = quick_metrics['psnr'] - prev_psnr
                    convergence_str = f" | ΔLoss(5): {loss_impr:+.2f}% | ΔPSNR(5): {psnr_impr:+.2f}dB"
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Iter {step:,} | "
                      f"Avg Train Loss: {avg_train_loss:.6f} | Avg Train PSNR: {avg_train_psnr:.2f} | "
                      f"Quick Val Loss: {quick_metrics['loss']:.6f} | Quick Val PSNR: {quick_metrics['psnr']:.2f} | "
                      f"Quick Val SSIM: {quick_metrics['ssim']:.4f}{convergence_str}")
                if etc_info:
                    eta_str = etc_info['completion_time'].strftime('%Y-%m-%d %H:%M:%S')
                    eta_human = format_time_duration(etc_info['remaining_time'])
                    print(f"📊 Progress: {etc_info['progress_percent']:.1f}% | ETA: {eta_human} ({eta_str}) | Avg: {etc_info['median_iter_time']:.3f}s/iter (median)")
                    if etc_info['eta_unstable']:
                        print(f"⚠️ ETA may be unstable (mean iter time {etc_info['mean_iter_time']:.3f}s > 3x median {etc_info['median_iter_time']:.3f}s). Validation/checkpointing may be skewing estimate.")
                else:
                    print(f"📊 Progress: {(step / args.iters) * 100:.1f}% | ETA: Calculating...")
                
                # Save best model based on quick validation PSNR (better metric than loss)
                if quick_metrics['psnr'] > best_val_psnr:
                    best_val_psnr = quick_metrics['psnr']
                    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_best.pth'))
                    print(f"🏆 Best model saved at iter {step:,} with quick val PSNR {quick_metrics['psnr']:.2f}")
                
                # Periodic metrics/model save for in-progress assessment
                metrics_state = {
                    'step': step,
                    'train_losses': convert_for_json(train_losses),
                    'train_psnrs': convert_for_json(train_psnrs),
                    'quick_val_losses': convert_for_json(quick_val_losses),
                    'quick_val_psnrs': convert_for_json(quick_val_psnrs),
                    'quick_val_ssims': convert_for_json(quick_val_ssims),
                    'val_steps': convert_for_json(val_steps),
                    'iteration_times': convert_for_json(iteration_times),
                    'best_val_psnr': float(best_val_psnr)
                }
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics_state
                }, os.path.join(args.save_dir, 'metrics_latest.pth'))
                import json
                with open(os.path.join(args.save_dir, 'metrics_latest.json'), 'w') as f:
                    json.dump(metrics_state, f, indent=2)
                
                print("-" * 80)  # Separator for better readability
            
            # Save model and metrics every 1000 steps for redundancy
            if step % 1000 == 0 and step > 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_{step}_latest.pth'))
                metrics_state = {
                    'step': step,
                    'train_losses': convert_for_json(train_losses),
                    'train_psnrs': convert_for_json(train_psnrs),
                    'quick_val_losses': convert_for_json(quick_val_losses),
                    'quick_val_psnrs': convert_for_json(quick_val_psnrs),
                    'quick_val_ssims': convert_for_json(quick_val_ssims),
                    'val_steps': convert_for_json(val_steps),
                    'iteration_times': convert_for_json(iteration_times),
                    'best_val_psnr': float(best_val_psnr)
                }
                import json
                with open(os.path.join(args.save_dir, f'metrics_{step}_latest.json'), 'w') as f:
                    json.dump(metrics_state, f, indent=2)
            
            if step % 10000 == 0 and step > 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_{step}.pth'))
            
            if step % summary_interval == 0 and step > 0:
                avg_train_loss = summary_train_loss / summary_train_count if summary_train_count > 0 else float('nan')
                avg_train_psnr = summary_train_psnr / summary_train_count if summary_train_count > 0 else float('nan')
                # ETA based only on training step times
                if training_step_times:
                    median_train_time = np.median(training_step_times[-100:])
                    remaining_steps = args.iters - step
                    eta_seconds = remaining_steps * median_train_time
                    eta_str = format_time_duration(eta_seconds)
                    eta_time = (datetime.now() + timedelta(seconds=eta_seconds)).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    eta_str = 'N/A'
                    eta_time = 'N/A'
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | SUMMARY @ Iter {step:,}")
                print(f"  Avg Train Loss (last {summary_interval}): {avg_train_loss:.6f}")
                print(f"  Avg Train PSNR (last {summary_interval}): {avg_train_psnr:.2f}")
                if last_val_loss is not None:
                    print(f"  Last Validation @ Iter {last_val_step:,}: Loss={last_val_loss:.6f}, PSNR={last_val_psnr:.2f}")
                else:
                    print("  Last Validation: (not yet run)")
                print(f"  ETA: {eta_str} (approx {eta_time}) [based on training steps only]")
                print("=" * 80)
                # Reset summary counters
                summary_train_loss = 0
                summary_train_psnr = 0
                summary_train_count = 0
            
            step += 1
            if step >= args.iters:
                break
    
    # Final save
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_final.pth'))
    total_time = time.time() - start_time
    print(f'🎉 Training complete in {total_time/3600:.2f} hours ({total_time:.0f} seconds)')

    # Always run a final full validation at the end
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Running final full validation at end of training...")
    final_full_metrics = validate(model, renderer, val_loader, device)
    full_val_losses.append(float(final_full_metrics['loss']))
    full_val_psnrs.append(float(final_full_metrics['psnr']))
    full_val_ssims.append(float(final_full_metrics['ssim']))
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | FINAL | Full Val Loss: {final_full_metrics['loss']:.6f} | Full Val PSNR: {final_full_metrics['psnr']:.2f} | Full Val SSIM: {final_full_metrics['ssim']:.4f}")
    # Save final full validation checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'full_val_loss': float(final_full_metrics['loss']),
        'full_val_psnr': float(final_full_metrics['psnr']),
        'full_val_ssim': float(final_full_metrics['ssim']),
        'quick_val_loss': quick_val_losses[-1] if quick_val_losses else None,
        'quick_val_psnr': quick_val_psnrs[-1] if quick_val_psnrs else None
    }, os.path.join(args.save_dir, f'model_full_val_final.pth'))

    # Plotting
    print("Generating comprehensive plots...")
    
    # Create a larger figure with better organization
    fig = plt.figure(figsize=(20, 16))
    
    # Main convergence plot (larger, more prominent)
    ax1 = plt.subplot(3, 3, (1, 2))  # Spans 2 columns
    ax1.plot(val_steps, train_losses, label='Training Loss', marker='o', markersize=4, linewidth=2, color='blue', alpha=0.8)
    ax1.plot(val_steps, quick_val_losses, label='Validation Loss', marker='s', markersize=4, linewidth=2, color='red', alpha=0.8)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization of loss changes
    
    # PSNR convergence
    ax2 = plt.subplot(3, 3, (1, 3))  # Spans 2 columns
    ax2.plot(val_steps, train_psnrs, label='Training PSNR', marker='o', markersize=4, linewidth=2, color='green', alpha=0.8)
    ax2.plot(val_steps, quick_val_psnrs, label='Validation PSNR', marker='s', markersize=4, linewidth=2, color='orange', alpha=0.8)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Training vs Validation PSNR', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Loss difference (overfitting indicator)
    ax3 = plt.subplot(3, 3, 4)
    if len(train_losses) == len(quick_val_losses):
        loss_diff = [abs(t - v) for t, v in zip(train_losses, quick_val_losses)]
        ax3.plot(val_steps, loss_diff, marker='o', markersize=3, color='purple', alpha=0.8)
        ax3.set_xlabel('Iteration', fontsize=10)
        ax3.set_ylabel('|Train - Val Loss|', fontsize=10)
        ax3.set_title('Overfitting Indicator', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # Learning rate schedule (if applicable)
    ax4 = plt.subplot(3, 3, 5)
    # For now, plot constant LR, but this could be enhanced for LR scheduling
    lr_values = [args.lr] * len(val_steps)
    ax4.plot(val_steps, lr_values, marker='o', markersize=3, color='brown', alpha=0.8)
    ax4.set_xlabel('Iteration', fontsize=10)
    ax4.set_ylabel('Learning Rate', fontsize=10)
    ax4.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # SSIM Progress
    ax5 = plt.subplot(3, 3, 6)
    if quick_val_ssims:
        ax5.plot(val_steps, quick_val_ssims, label='Quick Val SSIM', marker='s', markersize=3, color='green', alpha=0.8)
        if full_val_ssims:
            full_val_steps = [args.full_val_interval * (i+1) for i in range(len(full_val_ssims))]
            ax5.plot(full_val_steps, full_val_ssims, label='Full Val SSIM', marker='^', markersize=4, color='orange', alpha=0.8)
        ax5.set_xlabel('Iteration', fontsize=10)
        ax5.set_ylabel('SSIM', fontsize=10)
        ax5.set_title('SSIM Progress', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
    
    # Training time per iteration
    ax6 = plt.subplot(3, 3, 7)
    if len(iteration_times) > 100:
        recent_times = iteration_times[-1000:]  # Last 1000 iterations
        ax6.plot(recent_times, alpha=0.6, color='purple')
        ax6.set_xlabel('Recent Iterations', fontsize=10)
        ax6.set_ylabel('Time (seconds)', fontsize=10)
        ax6.set_title('Training Time per Iteration', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    # Full validation metrics (if available)
    if full_val_losses:
        ax7 = plt.subplot(3, 3, 8)
        full_val_steps = [args.full_val_interval * (i+1) for i in range(len(full_val_losses))]
        ax7.plot(full_val_steps, full_val_losses, label='Full Val Loss', marker='^', markersize=4, color='red', alpha=0.8)
        ax7.set_xlabel('Iteration', fontsize=10)
        ax7.set_ylabel('Loss', fontsize=10)
        ax7.set_title('Full Validation Loss', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3)
        ax7.set_yscale('log')
        
        ax8 = plt.subplot(3, 3, 9)
        ax8.plot(full_val_steps, full_val_psnrs, label='Full Val PSNR', marker='^', markersize=4, color='orange', alpha=0.8)
        ax8.set_xlabel('Iteration', fontsize=10)
        ax8.set_ylabel('PSNR (dB)', fontsize=10)
        ax8.set_title('Full Validation PSNR', fontsize=12, fontweight='bold')
        ax8.legend(fontsize=10)
        ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'comprehensive_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create a separate, focused convergence plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss convergence (prominent)
    ax1.plot(val_steps, train_losses, label='Training Loss', marker='o', markersize=4, linewidth=2, color='blue')
    ax1.plot(val_steps, quick_val_losses, label='Validation Loss', marker='s', markersize=4, linewidth=2, color='red')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # PSNR convergence (prominent)
    ax2.plot(val_steps, train_psnrs, label='Training PSNR', marker='o', markersize=4, linewidth=2, color='green')
    ax2.plot(val_steps, quick_val_psnrs, label='Validation PSNR', marker='s', markersize=4, linewidth=2, color='orange')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Training vs Validation PSNR', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'convergence_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save comprehensive validation data
    validation_data = {
        'train_losses': convert_for_json(train_losses),
        'train_psnrs': convert_for_json(train_psnrs),
        'quick_val_losses': convert_for_json(quick_val_losses),
        'quick_val_psnrs': convert_for_json(quick_val_psnrs),
        'quick_val_ssims': convert_for_json(quick_val_ssims),
        'full_val_losses': convert_for_json(full_val_losses),
        'full_val_psnrs': convert_for_json(full_val_psnrs),
        'full_val_ssims': convert_for_json(full_val_ssims),
        'val_steps': convert_for_json(val_steps),
        'iteration_times': convert_for_json(iteration_times),
        'total_training_time': float(total_time),
        'best_val_psnr': float(best_val_psnr),
        'config': {
            'quick_val_res': [int(x) for x in args.quick_val_res],
            'quick_val_subset': int(args.quick_val_subset),
            'full_val_res': [int(x) for x in args.img_wh],
            'quick_val_interval': int(args.quick_val_interval),
            'full_val_interval': int(args.full_val_interval),
            'batch_size': int(args.batch_size),
            'learning_rate': float(args.lr),
            'total_iterations': int(args.iters)
        }
    }
    
    with open(os.path.join(args.save_dir, 'comprehensive_metrics.json'), 'w') as f:
        json.dump(validation_data, f, indent=2)
    
    print(f"Comprehensive plots and metrics saved to {args.save_dir}")
    print(f"📊 Main convergence plot: {os.path.join(args.save_dir, 'convergence_plot.png')}")
    print(f"📊 Detailed metrics plot: {os.path.join(args.save_dir, 'comprehensive_metrics.png')}")

if __name__ == '__main__':
    main() 