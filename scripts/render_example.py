import os
import sys
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from PIL import Image
from nerfmlp import NeRFMLP, NeRFRenderer, NeRFDataset

def main():
    parser = argparse.ArgumentParser(description='Render test images from trained NeRF model')
    parser.add_argument('--model_path', type=str, default='outputs/checkpoints/model_best.pth',
                       help='Path to model checkpoint (default: outputs/checkpoints/model_best.pth)')
    parser.add_argument('--datadir', type=str, default='./data/lego',
                       help='Path to dataset directory (default: ./data/lego)')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (default: train)')
    parser.add_argument('--img_wh', type=int, nargs=2, default=[400, 400],
                       help='Image width and height (default: 400 400)')
    parser.add_argument('--num_views', type=int, default=5,
                       help='Number of views to render (default: 5)')
    parser.add_argument('--out_prefix', type=str, default='outputs/rendered_example',
                       help='Output file prefix (default: outputs/rendered_example)')
    args = parser.parse_args()

    # --- Config ---
    datadir = args.datadir
    split = args.split
    img_wh = tuple(args.img_wh)
    model_path = args.model_path
    out_prefix = args.out_prefix
    num_views = args.num_views

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # --- Load dataset for camera poses ---
    dataset = NeRFDataset(datadir, split=split, img_wh=img_wh)
    H, W = img_wh
    focal = dataset.focal
    poses = dataset.poses
    n_poses = len(poses)
    
    print(f"Loaded {n_poses} poses from {split} split")
    print(f"Focal length: {focal}")
    print(f"Image dimensions: {img_wh}")
    
    # Print first few camera positions for debugging
    for i in range(min(3, n_poses)):
        pos = poses[i][:3, 3]
        print(f"Camera {i} position: {pos}")

    # --- Dynamic near/far calculation ---
    positions = np.array([pose[:3, 3] for pose in poses])
    
    # Check if cameras are on a sphere (common in NeRF datasets)
    dists = np.linalg.norm(positions, axis=1)
    dist_std = dists.std()
    
    if dist_std < 0.01:  # Cameras are on a sphere
        print(f"📐 Detected spherical camera arrangement (distance std: {dist_std:.6f})")
        # Use scene bounds instead of camera distances
        scene_center = np.mean(positions, axis=0)
        scene_radius = np.linalg.norm(positions - scene_center, axis=1).max()
        
        # For spherical scenes, use a wider range that covers the scene
        near = max(0.1, scene_radius * 0.5)  # Start at half the scene radius
        far = scene_radius * 2.0  # Extend to twice the scene radius
        print(f"🎯 Scene-based near/far: near={near:.3f}, far={far:.3f} (scene_radius={scene_radius:.3f})")
    else:
        # Original calculation for non-spherical scenes
        near = max(0.1, dists.min() - 0.5)
        far = dists.max() + 0.5
        print(f"📐 Camera-based near/far: near={near:.3f}, far={far:.3f}")
    
    print(f"Dynamic near: {near}, far: {far}")

    # --- Load model ---
    model = NeRFMLP().to(device)
    
    # Check if best model exists, otherwise try final model
    if not os.path.exists(model_path):
        model_path = 'outputs/checkpoints/model_final.pth'
        if not os.path.exists(model_path):
            print(f"Error: No model found at {model_path}")
            print("Available models in outputs/checkpoints/:")
            if os.path.exists('outputs/checkpoints/'):
                for f in os.listdir('outputs/checkpoints/'):
                    if f.endswith('.pth'):
                        print(f"  - {f}")
            return
    
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    renderer = NeRFRenderer(model, device, near=near, far=far)

    # --- Render multiple views ---
    with torch.no_grad():
        for idx in range(num_views):
            pose_idx = idx % n_poses  # cycle if fewer than 5 poses
            pose = poses[pose_idx]
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
            
            print(f"Rendered RGB range: {rgb.min():.3f} to {rgb.max():.3f}")
            print(f"Rendered RGB mean: {rgb.mean():.3f}")
            
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            out_path = f'{out_prefix}_{idx}.png'
            Image.fromarray(rgb).save(out_path)
            print(f"Rendered image saved to {out_path}")
            print("---")

if __name__ == '__main__':
    main() 