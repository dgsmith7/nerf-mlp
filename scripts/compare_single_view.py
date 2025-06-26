import argparse
import os
import sys
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import torch
from nerfmlp import NeRFMLP, NeRFRenderer, NeRFDataset

def main():
    parser = argparse.ArgumentParser(description='Compare NeRF rendering to ground truth for a single view.')
    parser.add_argument('--view_idx', type=int, required=True, help='Index of the view to compare (e.g., 10)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to NeRF model checkpoint')
    parser.add_argument('--datadir', type=str, default='data/lego', help='Path to dataset directory')
    parser.add_argument('--img_wh', type=int, nargs=2, default=[400, 400], help='Image width and height')
    parser.add_argument('--output', type=str, required=True, help='Path to save side-by-side comparison image')
    parser.add_argument('--gt_dir', type=str, default=None, help='Directory for ground truth images (default: data/lego/train)')
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    img_wh = tuple(args.img_wh)
    gt_dir = args.gt_dir or os.path.join(args.datadir, 'train')
    gt_path = os.path.join(gt_dir, f"r_{args.view_idx}.png")
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth image {gt_path} does not exist.")
        exit(1)
    gt_img = Image.open(gt_path).convert('RGB')

    # Load dataset and get pose
    dataset = NeRFDataset(args.datadir, split='train', img_wh=img_wh)
    poses = dataset.poses
    n_poses = len(poses)
    pose_idx = args.view_idx % n_poses
    pose = poses[pose_idx]
    focal = dataset.focal
    H, W = img_wh

    # Dynamic near/far (copied from render_example.py)
    positions = np.array([p[:3, 3] for p in poses])
    dists = np.linalg.norm(positions, axis=1)
    dist_std = dists.std()
    if dist_std < 0.01:
        scene_center = np.mean(positions, axis=0)
        scene_radius = np.linalg.norm(positions - scene_center, axis=1).max()
        near = max(0.1, scene_radius * 0.5)
        far = scene_radius * 2.0
    else:
        near = max(0.1, dists.min() - 0.5)
        far = dists.max() + 0.5

    # Load model with support for both .npy and .pth formats
    model = NeRFMLP().to(device)
    
    print(f"Loading model from: {args.model_path}")
    if args.model_path.endswith('.npy'):
        # Load weights from numpy file (official example weights)
        weights = np.load(args.model_path, allow_pickle=True)
        if isinstance(weights, np.lib.npyio.NpzFile):
            weights = [weights[key] for key in weights.files]
        elif isinstance(weights, np.ndarray) and weights.dtype == object:
            weights = list(weights)
        model.load_from_numpy(weights)
        print("✅ Loaded weights from .npy file using load_from_numpy")
        # Configure for official weights
        N_samples, N_importance = 64, 64
    else:
        # Load PyTorch checkpoint
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("✅ Loaded weights from .pth file")
        # Configure for custom weights  
        N_samples, N_importance = 64, 128
    
    model.eval()
    renderer = NeRFRenderer(model, device, near=near, far=far, 
                           N_samples=N_samples, N_importance=N_importance)

    # Render the view
    with torch.no_grad():
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        dirs = np.stack([(i - W/2)/focal, -(j - H/2)/focal, -np.ones_like(i)], -1)
        rays_d = (dirs @ pose[:3, :3].T).reshape(-1, 3)
        rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
        rays_o = torch.from_numpy(rays_o.copy()).float().to(device)
        rays_d = torch.from_numpy(rays_d).float().to(device)
        rgb = renderer.render(rays_o, rays_d, H, W, focal)
        rgb = rgb.cpu().numpy()
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        pred_img = Image.fromarray(rgb)

    # Resize to match ground truth
    pred_img = pred_img.resize(gt_img.size)
    side_by_side = np.concatenate([np.array(gt_img), np.array(pred_img)], axis=1)
    Image.fromarray(side_by_side).save(args.output)
    print(f"Saved side-by-side comparison to {args.output}")

if __name__ == '__main__':
    main() 