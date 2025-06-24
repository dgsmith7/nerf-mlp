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

# The script will:
# - Load the best model from outputs/checkpoints/model_best.pth
# - Render 5 test views from the test dataset
# - Save images as outputs/rendered_example_*.png
```

## Configuration

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

1. **Black Rendered Images**: Usually indicates incorrect near/far planes - the dynamic calculation should fix this
2. **High Memory Usage**: Reduce batch size or image resolution
3. **Slow Training**: Use lower resolution for quick experiments, increase batch size if memory allows
4. **Poor Convergence**: Check learning rate, ensure validation data is properly loaded

### Performance Tips

- Start with lower resolution (e.g., 256x256) for quick experiments
- Use `train_only.py` for initial testing without validation overhead
- Monitor memory usage and adjust batch size accordingly
- Use the live plotting feature to catch issues early

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
