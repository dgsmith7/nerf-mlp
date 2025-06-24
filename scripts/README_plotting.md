# Training Progress Monitoring Tools

This directory contains tools for monitoring and visualizing NeRF training progress.

## Quick Progress Viewer

**Script:** `view_progress.py`

Quickly check the current training status without generating plots.

```bash
python scripts/view_progress.py
```

**Output:** Displays current iteration, metrics, improvements, and performance stats in the terminal.

## Training Progress Plotter

**Script:** `plot_training_progress.py`

Generate comprehensive plots of training progress.

### Basic Usage

```bash
python scripts/plot_training_progress.py
```

### Live Monitoring

```bash
python scripts/plot_training_progress.py --live
```

### Options

- `--metrics-file`: Path to metrics JSON file (default: `outputs/checkpoints/metrics_latest.json`)
- `--save-dir`: Directory to save plots (default: `outputs/checkpoints`)
- `--live`: Run in live mode with automatic updates
- `--interval`: Update interval in milliseconds for live mode (default: 5000)

## Generated Plots

### 1. Convergence Plot (`convergence_plot.png`)

- **Training vs Validation Loss**: Shows loss convergence with log scale
- **Training vs Validation PSNR**: Shows PSNR improvement over time

### 2. Comprehensive Metrics (`comprehensive_metrics.png`)

- **Main convergence plots** (larger, more prominent)
- **Overfitting indicator**: Shows the gap between training and validation loss
- **Learning rate schedule**: Current learning rate over time
- **SSIM progress**: Structural similarity index over time
- **Training time per iteration**: Performance monitoring
- **Full validation metrics**: If available

### 3. Training Progress (`training_progress.png`)

- **6-panel overview** with all key metrics
- **Summary statistics** showing current status and improvements
- **Real-time updates** when used with `--live` flag

## Metrics Tracked

- **Training Loss**: MSE loss on training data
- **Validation Loss**: MSE loss on validation data
- **Training PSNR**: Peak Signal-to-Noise Ratio on training data
- **Validation PSNR**: Peak Signal-to-Noise Ratio on validation data
- **Validation SSIM**: Structural Similarity Index on validation data
- **Iteration Times**: Time per training iteration
- **Best Validation PSNR**: Highest PSNR achieved during training

## Usage During Training

1. **Start training** with the main training script
2. **Monitor progress** in real-time:
   ```bash
   python scripts/plot_training_progress.py --live
   ```
3. **Quick status check**:
   ```bash
   python scripts/view_progress.py
   ```
4. **Generate final plots** after training completes

## Interpreting the Plots

### Good Training Signs

- Training and validation loss both decreasing
- PSNR increasing over time
- Small gap between training and validation loss
- Consistent iteration times

### Warning Signs

- Validation loss increasing while training loss decreases (overfitting)
- Large gap between training and validation loss
- PSNR plateauing or decreasing
- Inconsistent iteration times

## File Locations

- **Metrics data**: `outputs/checkpoints/metrics_latest.json`
- **Generated plots**: `outputs/checkpoints/`
- **Model checkpoints**: `outputs/checkpoints/model_*.pth`
