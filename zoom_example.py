#!/usr/bin/env python3
"""
Demonstration of zoomed-in novel view rendering.
Shows how to create custom camera positions for close-up shots.
"""

import numpy as np
import torch
from PIL import Image
import os
import sys

# Add the project root to the path so we can import our modules
sys.path.append('.')

from nerfmlp.model import NeRFMLP
from nerfmlp.renderer import NeRFRenderer
from nerfmlp.data import NeRFDataset


def look_at_matrix(eye, target, up=np.array([0, 1, 0])):
    """Create a camera pose matrix using look-at parameters."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up_corrected = np.cross(right, forward)
    up_corrected = up_corrected / np.linalg.norm(up_corrected)
    
    # NeRF uses -Z as forward direction
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up_corrected
    pose[:3, 2] = -forward  # Negative for NeRF convention
    pose[:3, 3] = eye
    
    return pose[:3, :]  # Return 3x4 matrix like NeRF expects


def main():
    # Configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model_path = 'data/lego_example_weights/model_fine_200000.npy'
    output_dir = 'outputs/zoom_examples'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = NeRFMLP().to(device)
    
    if model_path.endswith('.npy'):
        weights = np.load(model_path, allow_pickle=True)
        if isinstance(weights, np.lib.npyio.NpzFile):
            weights = [weights[key] for key in weights.files]
        elif isinstance(weights, np.ndarray) and weights.dtype == object:
            weights = list(weights)
        model.load_from_numpy(weights)
        print("Loaded weights from .npy file")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded weights from .pth file")
    
    model.eval()
    
    # Load dataset to get original focal length
    dataset = NeRFDataset('data/lego', split='train', img_wh=(400, 400))
    original_focal = dataset.focal
    
    print(f"Original focal length: {original_focal:.2f}")
    
    # Define different zoom scenarios
    scenarios = [
        {
            'name': 'normal_distance',
            'camera_pos': np.array([2.0, 2.0, 2.0]),  # Closer than training (~4.03)
            'focal_multiplier': 1.0,
            'description': 'Closer camera, normal focal length'
        },
        {
            'name': 'telephoto_zoom',
            'camera_pos': np.array([3.0, 3.0, 3.0]),  # Moderate distance
            'focal_multiplier': 2.5,
            'description': 'Moderate distance, telephoto lens'
        },
        {
            'name': 'extreme_closeup',
            'camera_pos': np.array([1.2, 1.2, 1.2]),  # Very close
            'focal_multiplier': 1.5,
            'description': 'Very close camera with slight zoom'
        },
        {
            'name': 'detail_shot',
            'camera_pos': np.array([0.8, 1.5, 0.8]),  # Close from slight angle
            'focal_multiplier': 3.0,
            'description': 'Close angled view with strong telephoto'
        }
    ]
    
    # Model center (LEGO bulldozer is roughly at origin)
    target = np.array([0.0, 0.0, 0.0])
    
    print("\nRendering zoom scenarios...")
    print("=" * 60)
    
    for scenario in scenarios:
        print(f"\n📸 {scenario['name']}: {scenario['description']}")
        
        # Create camera pose
        camera_pos = scenario['camera_pos']
        pose = look_at_matrix(camera_pos, target)
        
        # Calculate focal length
        focal = original_focal * scenario['focal_multiplier']
        
        # Calculate distance for near/far bounds
        distance_to_target = float(np.linalg.norm(camera_pos - target))
        near = max(0.1, distance_to_target - 1.5)
        far = distance_to_target + 1.5
        
        print(f"   Camera position: {camera_pos}")
        print(f"   Distance to target: {distance_to_target:.2f}")
        print(f"   Focal length: {focal:.2f} (multiplier: {scenario['focal_multiplier']}x)")
        print(f"   Near/Far bounds: {near:.2f} / {far:.2f}")
        
        # Create renderer
        renderer = NeRFRenderer(
            model, device,
            near=near, far=far,
            N_samples=64, N_importance=64,
            white_bkgd=True,
            perturb=0.0,  # No noise for clean inference
            raw_noise_std=0.0
        )
        
        # Render the view
        H, W = 400, 400
        
        with torch.no_grad():
            # Generate rays
            i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
            dirs = np.stack([(i - W/2)/focal, -(j - H/2)/focal, -np.ones_like(i)], -1)
            rays_d = (dirs @ pose[:3, :3].T).reshape(-1, 3)
            rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
            
            # Convert to tensors
            rays_o = torch.from_numpy(rays_o.copy()).float().to(device)
            rays_d = torch.from_numpy(rays_d).float().to(device)
            
            # Render
            print("   Rendering...")
            rgb = renderer.render(rays_o, rays_d, H, W, focal)
            rgb = rgb.cpu().numpy()
            
            # Statistics
            print(f"   RGB range: {rgb.min():.3f} to {rgb.max():.3f}")
            print(f"   RGB mean: {rgb.mean():.3f}")
            
            # Save image
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            output_path = os.path.join(output_dir, f"{scenario['name']}.png")
            Image.fromarray(rgb).save(output_path)
            print(f"   ✅ Saved: {output_path}")
    
    print(f"\n🎉 All zoom examples saved to: {output_dir}/")
    print("\nCompare the different zoom techniques:")
    for scenario in scenarios:
        print(f"   • {scenario['name']}.png - {scenario['description']}")


if __name__ == '__main__':
    main() 