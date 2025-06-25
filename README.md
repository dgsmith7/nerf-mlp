# NeRF MLP Implementation

A PyTorch implementation of Neural Radiance Fields (NeRF) using Multi-Layer Perceptrons (MLPs), optimized for Apple Silicon (MPS) and designed for benchmarking different raymarching techniques.

## References

- **Original Paper**: [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934) (Mildenhall et al., ECCV 2020)
- **Official Implementation**: [bmild/nerf](https://github.com/bmild/nerf) (TensorFlow)

## Overview

This project implements the core NeRF architecture from the original paper using PyTorch, with several key features:

### MLP Architecture Description

The neural network is a **fully connected Multi-Layer Perceptron (MLP)** that takes 5D coordinates (3D position + 2D viewing direction) as input and outputs color and density for volume rendering. Here's how it works:

1. **Input Processing**:

   - 3D positions are encoded using positional encoding with 10 frequency bands (L=10)
   - 2D viewing directions are encoded with 4 frequency bands (L=4)
   - This transforms the inputs from 3+2=5 dimensions to 63+27=90 dimensions

2. **Network Structure**:

   - **8 fully connected layers** with 256 hidden units each
   - **ReLU activations** between layers
   - **Skip connections** at layer 4 (residual connections for better gradient flow)
   - **Two-stage output**: density (σ) and view-dependent color (RGB)

3. **Volume Rendering**:
   - Uses classical volume rendering to integrate colors and densities along rays
   - Supports hierarchical sampling (coarse + fine networks)
   - Dynamic near/far plane calculation based on camera positions

The MLP acts as a continuous 3D representation of the scene, allowing novel view synthesis by querying the network at any 3D point and viewing direction.

## Project Structure

```
nerf-mlp/
├── nerfmlp/                    # Core NeRF implementation
│   ├── __init__.py            # Package initialization
│   ├── model.py               # NeRF MLP model architecture
│   ├── data.py                # Dataset loading and processing
│   └── renderer.py            # Volume rendering implementation
├── scripts/                   # Training and utility scripts
│   ├── train.py               # Main training script with validation
│   ├── train_only.py          # Training-only script (no validation)
│   ├── render_example.py      # Test rendering script
│   ├── plot_training_progress.py  # Real-time training progress plotting
│   ├── view_progress.py       # Quick progress viewer
│   └── README_plotting.md     # Plotting tools documentation
├── data/                      # Dataset files
│   └── lego/                  # Lego dataset (synthetic)
├── outputs/                   # All outputs
│   ├── checkpoints/           # Model checkpoints and metrics
│   └── test_retrain/          # Test training outputs
├── example/                   # Example code and weights
└── requirements.txt           # Python dependencies
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ with MPS support
- Apple Silicon Mac (M1/M2/M3) for MPS acceleration
- NumPy, Matplotlib, scikit-image

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd nerf-mlp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with default settings (Lego dataset)
python scripts/train.py --datadir data/lego

# Train with custom settings
python scripts/train.py \
    --datadir data/lego \
    --img_wh 512 512 \
    --batch_size 512 \
    --iters 200000 \
    --lr 5e-4 \
    --save_dir outputs/my_experiment

# Training-only (no validation) for quick testing
python scripts/train_only.py --datadir data/lego --iters 5000
```

### Resuming Training

The training script supports automatic resumption from checkpoints, allowing you to continue training if it's interrupted:

```bash
# Resume from latest checkpoint (recommended)
python scripts/train.py --datadir data/lego --resume outputs/checkpoints/metrics_latest.pth

# Resume from best model
python scripts/train.py --datadir data/lego --resume outputs/checkpoints/model_best.pth

# Resume from specific step
python scripts/train.py --datadir data/lego --resume outputs/checkpoints/model_5000_latest.pth

# Resume with different parameters (e.g., continue for more iterations)
python scripts/train.py --datadir data/lego --iters 300000 --lr 1e-4 --resume outputs/checkpoints/metrics_latest.pth
```

**Available Checkpoints for Resumption:**

- `metrics_latest.pth` - Complete checkpoint (model + optimizer + metrics) - **Recommended**
- `model_best.pth` - Best model based on validation PSNR
- `model_{step}_latest.pth` - Model at specific step (every 1000 steps)
- `model_{step}.pth` - Model at specific step (every 10000 steps)
- `model_final.pth` - Final model after training

**What Gets Restored:**

- Model weights and optimizer state
- Training step number (continues from exact point)
- All training and validation metrics history
- Timing data for accurate ETA calculations
- Best validation PSNR tracking

**Important Notes:**

- Use the same dataset path when resuming
- You can change training parameters (learning rate, iterations, etc.) when resuming
- Progress will be shown: "Resuming training from step X (target: Y)"
- ETA calculations remain accurate since timing data is preserved

### Monitoring Training Progress

```bash
# Quick status check
python scripts/view_progress.py

# Generate training plots
python scripts/plot_training_progress.py

# Live monitoring with auto-updates
python scripts/plot_training_progress.py --live
```

### Rendering Test Images

```bash
# Render test images from best checkpoint
python scripts/render_example.py

# Render from specific checkpoint
python scripts/render_example.py --model_path outputs/checkpoints/model_1000_latest.pth

# Render with custom resolution
python scripts/render_example.py --img_wh 400 400

# Render multiple views with custom output prefix
python scripts/render_example.py --out_prefix outputs/my_test --num_views 10
```

**Features:**

- **Spherical Camera Detection**: Automatically handles datasets with cameras on spheres
- **Progress Testing**: Test different checkpoints as training progresses
- **Multiple Views**: Renders 5 different camera angles by default

**Testing Training Progress:**

```bash
# Test every 1000 steps
python scripts/render_example.py --model_path outputs/checkpoints/model_1000_latest.pth --out_prefix outputs/progress_1000

# Test every 5000 steps
python scripts/render_example.py --model_path outputs/checkpoints/model_5000_latest.pth --out_prefix outputs/progress_5000

# Test best model so far
python scripts/render_example.py --model_path outputs/checkpoints/model_best.pth --out_prefix outputs/progress_best
```

**Expected Results:**

- **Early training (1000-5000 steps)**: Dark images - normal for early NeRF training
- **Mid training (10,000-20,000 steps)**: Gradually brighter images as model learns
- **Late training (50,000+ steps)**: Bright, colorful images with good detail

## Configuration

### Successful Training Configurations

Based on extensive testing, here are the proven configurations that work well:

#### **High Resolution Training (1024x1024)**

```bash
python scripts/train.py \
    --datadir data/lego \
    --img_wh 1024 1024 \
    --batch_size 512 \
    --lr 1e-4 \
    --iters 200000
```

**Key Parameters:**

- **Batch Size**: 512 (2x increase for 16x resolution increase)
- **Learning Rate**: 1e-4 (4x reduction for stability)
- **Expected PSNR**: 20-25+ dB after 20,000+ iterations
- **Training Time**: ~4 hours for 200,000 iterations

#### **Quick Testing (64x64)**

```bash
python scripts/train.py \
    --datadir data/lego \
    --img_wh 64 64 \
    --batch_size 128 \
    --iters 10000
```

**Key Parameters:**

- **Batch Size**: 128 (proportional to resolution)
- **Learning Rate**: 5e-4 (standard for low resolution)
- **Expected PSNR**: 15-20 dB after 5,000+ iterations
- **Training Time**: ~10 minutes for 10,000 iterations

#### **Medium Resolution (512x512)**

```bash
python scripts/train.py \
    --datadir data/lego \
    --img_wh 512 512 \
    --batch_size 256 \
    --lr 2e-4 \
    --iters 100000
```

### Training Parameters

| Parameter              | Default             | Description                                |
| ---------------------- | ------------------- | ------------------------------------------ |
| `--datadir`            | Required            | Path to dataset directory                  |
| `--img_wh`             | 1024 1024           | Image width and height                     |
| `--batch_size`         | 256                 | Batch size (increase for lower resolution) |
| `--iters`              | 200000              | Number of training iterations              |
| `--lr`                 | 5e-4                | Learning rate                              |
| `--save_dir`           | outputs/checkpoints | Directory to save checkpoints              |
| `--quick_val_interval` | 1000                | Quick validation interval                  |
| `--quick_val_res`      | 256 256             | Quick validation resolution                |
| `--quick_val_subset`   | 10                  | Number of images for quick validation      |
| `--full_val_interval`  | 10000               | Full validation interval                   |
| `--resume`             | None                | Path to checkpoint to resume from          |

### Model Architecture Details

- **Positional Encoding**: L=10 for position (3D → 63D), L=4 for view direction (3D → 27D)
- **Network Depth**: 8 fully connected layers
- **Hidden Units**: 256 per layer
- **Skip Connections**: At layer 4 for better gradient flow
- **Activation**: ReLU
- **Output**: RGB (3) + density (1) = 4 values per 3D point

## Training Features

### Progress Monitoring

The training script provides comprehensive monitoring:

- **Real-time Console Output**: Every 100 iterations shows current loss, PSNR, learning rate, gradient norm, memory usage, and iteration time
- **Summary Reports**: Every 1000 iterations shows average training metrics and most recent validation results
- **Validation**: Quick validation every 1000 iterations, full validation every 10000 iterations
- **ETA Calculation**: Based on training steps only (excludes validation time for accuracy)

### Metrics Tracked

- **Training Loss**: MSE loss on training data
- **Training PSNR**: Peak Signal-to-Noise Ratio on training data
- **Validation Loss**: MSE loss on validation data
- **Validation PSNR**: Peak Signal-to-Noise Ratio on validation data
- **Validation SSIM**: Structural Similarity Index on validation data
- **Iteration Times**: Time per training iteration
- **Best Validation PSNR**: Highest PSNR achieved during training

### Visualization Tools

- **Convergence Plots**: Training vs validation loss and PSNR over time
- **Overfitting Detection**: Shows gap between training and validation loss
- **Performance Monitoring**: Iteration times and learning rate schedule
- **Real-time Updates**: Live plotting during training

## Outputs

All outputs are saved to the `outputs/` directory:

### Checkpoints (`outputs/checkpoints/`)

- `model_best.pth`: Best model based on validation PSNR
- `model_final.pth`: Final model after training
- `model_<step>.pth`: Periodic checkpoints every 1000 steps
- `metrics_latest.json`: Latest training metrics
- `comprehensive_metrics.json`: Complete training history

### Plots (`outputs/checkpoints/`)

- `convergence_plot.png`: Training vs validation loss and PSNR
- `comprehensive_metrics.png`: Multi-panel training overview
- `training_progress.png`: Real-time progress monitoring

### Rendered Images (`outputs/`)

- `rendered_example_*.png`: Test renders from trained model

## Performance Optimization

### Apple Silicon (MPS) Support

- Optimized for M1/M2/M3 Macs using Metal Performance Shaders
- Automatic device detection and fallback to CPU if needed
- Memory-efficient training with gradient accumulation

### Dynamic Near/Far Planes

- Automatically calculates optimal near/far bounds from camera positions
- Prevents the model from learning to output zeros everywhere
- More robust than static near/far values

### Validation Strategies

- **Quick Validation**: Low resolution (256x256) on subset of images for fast feedback
- **Full Validation**: High resolution on all validation images for accurate metrics
- **Configurable Intervals**: Balance between monitoring frequency and training speed

## Troubleshooting

### Common Issues

1. **White or Black Rendered Images**:

   - **Cause**: Incorrect near/far plane calculation for spherical camera arrangements
   - **Solution**: The renderer now automatically detects spherical cameras and uses scene-based near/far bounds
   - **Detection**: Look for "📐 Detected spherical camera arrangement" in render output
   - **Expected**: Near/far should be ~2.4-9.5 for Lego dataset (not 3.5-4.5)

2. **Dark Rendered Images in Early Training**:

   - **Cause**: Normal for early NeRF training (iterations 1,000-10,000)
   - **Solution**: Continue training - images get brighter as model learns
   - **Expected**: Images get brighter as training progresses (20,000+ iterations)

3. **High Memory Usage**: Reduce batch size or image resolution
4. **Slow Training**: Use lower resolution for quick experiments, increase batch size if memory allows
5. **Poor Convergence**: Check learning rate, ensure validation data is properly loaded

### Camera Position and Near/Far Plane Fix

**Problem**: The original near/far calculation failed for datasets with spherical camera arrangements (like Lego), where all cameras are at the same distance from the scene center.

**Symptoms**:

- All cameras at distance 4.031 (perfect sphere)
- Near/far range of only 1.0 units (3.531 to 4.531)
- White or black rendered images
- Training PSNR stuck at ~1.0 dB

**Solution Implemented**:

```python
# Automatic detection of spherical camera arrangements
if dist_std < 0.01:  # Cameras are on a sphere
    scene_center = np.mean(positions, axis=0)
    scene_radius = np.linalg.norm(positions - scene_center, axis=1).max()
    near = max(0.1, scene_radius * 0.5)  # Start at half scene radius
    far = scene_radius * 2.0  # Extend to twice scene radius
```

**Results**:

- **Before**: Near=3.531, Far=4.531 (1.0 unit range)
- **After**: Near=2.365, Far=9.458 (7.1 unit range)
- **Training**: PSNR improved from 1.0 dB to 17+ dB
- **Rendering**: No more white/black images

### Performance Tips

- Start with lower resolution (e.g., 256x256) for quick experiments
- Use `train_only.py` for initial testing without validation overhead
- Monitor memory usage and adjust batch size accordingly
- Use the live plotting feature to catch issues early
- For high resolution training, use the proven configurations above

## Future Work

This implementation serves as a foundation for benchmarking different raymarching techniques in GLSL shaders. Planned extensions include:

- Web-based benchmarking interface
- Multiple raymarching shader implementations
- Performance comparison tools
- Real-time rendering capabilities

## Citation

If you use this implementation in your research, please cite the original NeRF paper:

```bibtex
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}
```

# nerf-mlp
