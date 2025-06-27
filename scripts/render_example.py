import os
import sys
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from PIL import Image
from nerfmlp import NeRFMLP, NeRFRenderer, NeRFDataset

def linear_to_srgb(img):
    """
    Convert linear RGB to sRGB with proper gamma correction.
    
    The standard linear to sRGB conversion (inverse of sRGB to linear):
    - For values <= 0.0031308: srgb = linear * 12.92
    - For values > 0.0031308: srgb = 1.055 * linear^(1/2.4) - 0.055
    """
    img = img.astype(np.float32)
    srgb = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(img, 1/2.4) - 0.055
    )
    return srgb

def main():
    parser = argparse.ArgumentParser(description='Render test images from trained NeRF model')
    parser.add_argument('--model_path', type=str, default='outputs/checkpoints/model_best.pth',
                       help='Path to model checkpoint (default: outputs/checkpoints/model_best.pth)')
    parser.add_argument('--use_fine_weights', action='store_true',
                       help='Use fine model weights from lego_example_weights')
    parser.add_argument('--N_samples', type=int, default=None,
                       help='Number of coarse samples per ray (default: auto-detect)')
    parser.add_argument('--N_importance', type=int, default=None,
                       help='Number of fine samples per ray (default: auto-detect)')
    parser.add_argument('--white_bkgd', action='store_true',
                       help='Use white background (default: auto-detect)')
    parser.add_argument('--no_white_bkgd', action='store_true',
                       help='Use black background')
    parser.add_argument('--datadir', type=str, default='./data/lego',
                       help='Path to dataset directory (default: ./data/lego)')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (default: train)')
    parser.add_argument('--img_wh', type=int, nargs=2, default=[400, 400],
                       help='Image width and height (default: 400 400)')
    parser.add_argument('--num_views', type=int, default=5,
                       help='Number of views to render')
    parser.add_argument('--out_prefix', type=str, default='rendered_example',
                       help='Output filename prefix')
    parser.add_argument('--view_idx', type=int, default=None,
                       help='Specific view index to render (if not specified, renders multiple views)')
    parser.add_argument('--near', type=float, default=None, help='Near bound override')
    parser.add_argument('--far', type=float, default=None, help='Far bound override')
    parser.add_argument('--coord_scale', type=float, default=None, help='Coordinate scaling factor to match model expectations')
    parser.add_argument('--gamma_correction', action='store_true', help='Apply linear to sRGB gamma correction (can wash out colors)')
    parser.add_argument('--brightness_boost', type=float, default=1.0, help='Brightness multiplier (default: 1.0, try 1.1-1.3 for subtle boost)')
    args = parser.parse_args()

    # --- Config ---
    datadir = args.datadir
    split = args.split
    img_wh = tuple(args.img_wh)
    model_path = args.model_path
    out_prefix = args.out_prefix
    num_views = args.num_views
    view_idx = args.view_idx
    use_fine_weights = args.use_fine_weights
    N_samples = args.N_samples
    N_importance = args.N_importance
    white_bkgd = args.white_bkgd
    no_white_bkgd = args.no_white_bkgd
    near = args.near
    far = args.far
    coord_scale = args.coord_scale

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # --- Load dataset for camera poses ---
    dataset = NeRFDataset(datadir, split=split, img_wh=img_wh)
    H, W = img_wh
    focal = dataset.focal
    poses = np.array(dataset.poses)  # Convert to numpy array
    
    print(f"Loaded {len(poses)} poses from {args.split} split")
    print(f"Focal length: {focal}")
    print(f"Image dimensions: ({H}, {W})")

    # Apply coordinate scaling to camera poses if specified
    if coord_scale is not None:
        print(f"🔧 Using coordinate scaling factor: {coord_scale}")
        poses_original = poses.copy()
        poses_for_bounds = poses.copy()
        poses_for_bounds[:, :3, 3] *= coord_scale  # Scale camera positions ONLY for bounds calculation
        print(f"Original camera range: X[{poses_original[:, 0, 3].min():.3f}, {poses_original[:, 0, 3].max():.3f}] Y[{poses_original[:, 1, 3].min():.3f}, {poses_original[:, 1, 3].max():.3f}] Z[{poses_original[:, 2, 3].min():.3f}, {poses_original[:, 2, 3].max():.3f}]")
        print(f"Scaled camera range: X[{poses_for_bounds[:, 0, 3].min():.3f}, {poses_for_bounds[:, 0, 3].max():.3f}] Y[{poses_for_bounds[:, 1, 3].min():.3f}, {poses_for_bounds[:, 1, 3].max():.3f}] Z[{poses_for_bounds[:, 2, 3].min():.3f}, {poses_for_bounds[:, 2, 3].max():.3f}]")
        # IMPORTANT: Keep original poses for ray generation!
        poses_for_rendering = poses_original.copy()
    else:
        coord_scale = 1.0  # Default to no scaling
        poses_for_bounds = poses.copy()
        poses_for_rendering = poses.copy()
    
    # Print first few camera positions for debugging  
    for i in range(min(3, len(poses_for_rendering))):
        pos = poses_for_rendering[i, :3, 3]
        print(f"Camera {i} position (for rendering): {pos}")

    # Calculate dynamic near/far bounds based on SCALED scene geometry
    scene_center = np.array([pose[:3, 3] for pose in poses_for_bounds]).mean(axis=0)
    camera_distances = np.linalg.norm(np.array([pose[:3, 3] for pose in poses_for_bounds]) - scene_center, axis=1)
    scene_radius = camera_distances.max()
    
    # Use training-consistent bounds, with command line overrides if provided
    if near is not None and far is not None:
        dynamic_near, dynamic_far = near, far
        print(f"Using command line bounds: near={dynamic_near}, far={dynamic_far}")
    else:
        # Use training bounds (2.0, 6.0) instead of dynamic calculation
        # Dynamic bounds often mismatch what the model learned during training
        dynamic_near, dynamic_far = 2.0, 6.0
        print(f'Using training-consistent bounds: near={dynamic_near}, far={dynamic_far}')
        print(f'(Dynamic would have been: near={float(camera_distances.min() - scene_radius):.3f}, far={float(camera_distances.max() + scene_radius):.3f})')
    
    # Ensure reasonable bounds for 360° spherical scenes
    dynamic_near = max(dynamic_near, 0.01)  # Prevent negative or zero near
    dynamic_far = max(dynamic_far, dynamic_near + 0.1)  # Ensure far > near

    # --- Load model ---
    model = NeRFMLP().to(device)
    
    # Check if we should use the fine example weights
    if use_fine_weights:
        model_path = 'data/lego_example_weights/model_fine_200000.npy'
        if not os.path.exists(model_path):
            print(f"Error: Fine weights not found at {model_path}")
            return
    elif not os.path.exists(model_path):
        # If the user-specified model doesn't exist, show error and available options
        print(f"Error: Specified model not found at {model_path}")
        
        # Show available models as suggestions
        checkpoint_dir = 'outputs/checkpoints/'
        if os.path.exists(checkpoint_dir):
            print("Available models in outputs/checkpoints/:")
            for f in sorted(os.listdir(checkpoint_dir)):
                if f.endswith('.pth'):
                    print(f"  - {checkpoint_dir}{f}")
        
        # Check for other common locations
        other_locations = ['outputs/lego_full_training/', 'outputs/test_fixed_preprocessing/']
        for loc in other_locations:
            if os.path.exists(loc):
                models = [f for f in os.listdir(loc) if f.endswith('.pth')]
                if models:
                    print(f"Available models in {loc}:")
                    for f in sorted(models):
                        print(f"  - {loc}{f}")
        
        print(f"\nTip: Use --use_fine_weights to load the official example weights")
        print(f"Or specify a valid model path with --model_path")
        return
    
    print(f"Loading model from: {model_path}")
    if model_path.endswith('.npy'):
        # Load weights from numpy file (assume list of arrays)
        weights = np.load(model_path, allow_pickle=True)
        if isinstance(weights, np.lib.npyio.NpzFile):
            weights = [weights[key] for key in weights.files]
        elif isinstance(weights, np.ndarray) and weights.dtype == object:
            weights = list(weights)
        print("Shapes of loaded .npy arrays:")
        for i, arr in enumerate(weights):
            print(f"  Weight {i}: {arr.shape}")
        print("Model parameter names/types and shapes (expected):")
        param_list = []
        param_names = []
        for idx, l in enumerate(model.pts_linears):
            param_list.append(l.weight.data)
            param_names.append(f"pts_linears[{idx}].weight")
            param_list.append(l.bias.data)
            param_names.append(f"pts_linears[{idx}].bias")
        param_list.append(model.bottleneck_linear.weight.data)
        param_names.append("bottleneck_linear.weight")
        param_list.append(model.bottleneck_linear.bias.data)
        param_names.append("bottleneck_linear.bias")
        param_list.append(model.sigma_linear.weight.data)
        param_names.append("sigma_linear.weight")
        param_list.append(model.sigma_linear.bias.data)
        param_names.append("sigma_linear.bias")
        if model.use_viewdirs:
            param_list.append(model.view_linear.weight.data)
            param_names.append("view_linear.weight")
            param_list.append(model.view_linear.bias.data)
            param_names.append("view_linear.bias")
        param_list.append(model.rgb_linear.weight.data)
        param_names.append("rgb_linear.weight")
        param_list.append(model.rgb_linear.bias.data)
        param_names.append("rgb_linear.bias")
        for i, (p, n) in enumerate(zip(param_list, param_names)):
            print(f"  Model param {i}: {n} {tuple(p.shape)}")
        # Now try loading (should work)
        model.load_from_numpy(weights)
        print("Loaded weights from .npy file using load_from_numpy.")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Configure renderer parameters
    # Auto-detect configuration based on model type
    if use_fine_weights:
        # Official example weights configuration
        render_N_samples = N_samples if N_samples is not None else 64
        render_N_importance = N_importance if N_importance is not None else 64
        render_white_bkgd = True if white_bkgd else (False if no_white_bkgd else True)
        print("🎯 Using official example weights configuration")
    else:
        # Your custom weights - use more standard NeRF settings
        render_N_samples = N_samples if N_samples is not None else 64
        render_N_importance = N_importance if N_importance is not None else 128
        render_white_bkgd = True if white_bkgd else (False if no_white_bkgd else True)
        print("🎯 Using custom weights configuration")
    
    print(f"📊 Renderer config: N_samples={render_N_samples}, N_importance={render_N_importance}, white_bkgd={render_white_bkgd}")
    
    renderer = NeRFRenderer(
        model, device, 
        near=dynamic_near, far=dynamic_far, 
        N_samples=render_N_samples,
        N_importance=render_N_importance,
        white_bkgd=render_white_bkgd,
        perturb=0.0,           # DISABLE perturbation during inference
        raw_noise_std=0.0,     # DISABLE noise during inference
        coord_scale=coord_scale  # Apply coordinate scaling
    )

    # --- Render multiple views ---
    with torch.no_grad():
        if view_idx is not None:
            pose_idx = view_idx % len(poses_for_rendering)
            pose = poses_for_rendering[pose_idx]
            print(f"Rendering view {pose_idx} (user-specified)")
            print(f"Camera position: {pose[:3, 3]}")
            i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
            dirs = np.stack([(i - W/2)/focal, -(j - H/2)/focal, -np.ones_like(i)], -1)
            rays_d = (dirs @ pose[:3, :3].T).reshape(-1, 3)
            rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
            rays_o = torch.from_numpy(rays_o.copy()).float().to(device)
            rays_d = torch.from_numpy(rays_d).float().to(device)
            rgb = renderer.render(rays_o, rays_d, H, W, focal)
            rgb = rgb.cpu().numpy()
            print(f"Rendered RGB range (linear): {rgb.min():.3f} to {rgb.max():.3f}")
            print(f"Rendered RGB mean (linear): {rgb.mean():.3f}")
            
            # Apply brightness boost if requested
            if args.brightness_boost != 1.0:
                rgb = rgb * args.brightness_boost
                print(f"Applied brightness boost: {args.brightness_boost}x")
            
            if args.gamma_correction:
                # Convert from linear RGB to sRGB for proper display
                rgb_srgb = linear_to_srgb(rgb)
                print(f"Rendered RGB range (sRGB): {rgb_srgb.min():.3f} to {rgb_srgb.max():.3f}")
                print(f"Rendered RGB mean (sRGB): {rgb_srgb.mean():.3f}")
                rgb_final = rgb_srgb
            else:
                print("Using linear RGB output (default - no gamma correction)")
                rgb_final = rgb
            
            rgb = (np.clip(rgb_final, 0, 1) * 255).astype(np.uint8)
            out_path = f'{out_prefix}_view{pose_idx}.png'
            Image.fromarray(rgb).save(out_path)
            print(f"Rendered image saved to {out_path}")
            print("---")
        else:
            for idx in range(num_views):
                pose_idx = idx % len(poses_for_rendering)  # cycle if fewer than 5 poses
                pose = poses_for_rendering[pose_idx]
                print(f"Rendering view {idx} using pose {pose_idx}")
                print(f"Camera position: {pose[:3, 3]}")
                i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
                dirs = np.stack([(i - W/2)/focal, -(j - H/2)/focal, -np.ones_like(i)], -1)
                rays_d = (dirs @ pose[:3, :3].T).reshape(-1, 3)
                rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
                rays_o = torch.from_numpy(rays_o.copy()).float().to(device)
                rays_d = torch.from_numpy(rays_d).float().to(device)
                rgb = renderer.render(rays_o, rays_d, H, W, focal)
                rgb = rgb.cpu().numpy()
                print(f"Rendered RGB range (linear): {rgb.min():.3f} to {rgb.max():.3f}")
                print(f"Rendered RGB mean (linear): {rgb.mean():.3f}")
                
                # Apply brightness boost if requested
                if args.brightness_boost != 1.0:
                    rgb = rgb * args.brightness_boost
                    print(f"Applied brightness boost: {args.brightness_boost}x")
                
                if args.gamma_correction:
                    # Convert from linear RGB to sRGB for proper display
                    rgb_srgb = linear_to_srgb(rgb)
                    print(f"Rendered RGB range (sRGB): {rgb_srgb.min():.3f} to {rgb_srgb.max():.3f}")
                    print(f"Rendered RGB mean (sRGB): {rgb_srgb.mean():.3f}")
                    rgb_final = rgb_srgb
                else:
                    print("Using linear RGB output (default - no gamma correction)")
                    rgb_final = rgb
                
                rgb = (np.clip(rgb_final, 0, 1) * 255).astype(np.uint8)
                out_path = f'{out_prefix}_{idx}.png'
                Image.fromarray(rgb).save(out_path)
                print(f"Rendered image saved to {out_path}")
                print("---")

if __name__ == '__main__':
    main() 