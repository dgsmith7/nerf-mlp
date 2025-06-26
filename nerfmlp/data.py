import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def srgb_to_linear(img):
    """
    Convert sRGB image to linear RGB with proper gamma correction.
    
    The standard sRGB to linear conversion as used in NeRF:
    - For values <= 0.04045: linear = srgb / 12.92
    - For values > 0.04045: linear = ((srgb + 0.055) / 1.055)^2.4
    """
    img = img.astype(np.float32)
    linear = np.where(
        img <= 0.04045,
        img / 12.92,
        np.power((img + 0.055) / 1.055, 2.4)
    )
    return linear

class NeRFDataset(Dataset):
    def __init__(self, datadir, split='train', img_wh=(400, 400), white_bkgd=True):
        super().__init__()
        self.datadir = datadir
        self.split = split
        self.img_wh = img_wh
        self.white_bkgd = white_bkgd
        self._load_meta()
        self._load_images_and_poses()
        self._generate_rays()

    def _load_meta(self):
        # Load transforms JSON
        with open(os.path.join(self.datadir, f'transforms_{self.split}.json'), 'r') as f:
            self.meta = json.load(f)

    def _load_images_and_poses(self):
        self.images = []
        self.poses = []
        for i, frame in enumerate(self.meta['frames']):
            fname = os.path.join(self.datadir, self.split, frame['file_path'].split('/')[-1] + '.png')
            img = Image.open(fname).convert('RGBA')  # Load with alpha channel
            img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
            img = np.array(img) / 255.0  # Convert to 0-1 range first
            
            # Handle alpha compositing with white background (standard NeRF preprocessing)
            if img.shape[2] == 4:  # RGBA
                rgb = img[..., :3]
                alpha = img[..., 3:]
                if self.white_bkgd:
                    # Composite with white background: rgb = rgb * alpha + white * (1 - alpha)
                    rgb = rgb * alpha + (1 - alpha)  # white background = 1
                else:
                    # Keep alpha channel
                    rgb = np.concatenate([rgb, alpha], axis=-1)
                img = rgb
            
            # Convert from sRGB to linear RGB (proper gamma correction)
            img = srgb_to_linear(img)
            
            self.images.append(img)
            pose = np.array(frame['transform_matrix'], dtype=np.float32)
            self.poses.append(pose)
        # Convert to numpy arrays and check shapes
        self.images = np.stack(self.images, axis=0)  # (N_images, H, W, 3)
        self.poses = np.stack(self.poses, axis=0)    # (N_images, 4, 4)
        
        # Print shape information for debugging
        print(f"Loaded {len(self.images)} images with shape: {self.images.shape}")
        print(f"Image value range: [{self.images.min():.3f}, {self.images.max():.3f}]")
        self.focal = 0.5 * self.img_wh[0] / np.tan(0.5 * self.meta['camera_angle_x'])

    def _generate_rays(self):
        # Generate rays for all images
        H, W = self.img_wh
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        dirs = np.stack([(i - W/2)/self.focal, -(j - H/2)/self.focal, -np.ones_like(i)], -1)  # (H, W, 3)
        self.all_rays_o = []
        self.all_rays_d = []
        self.all_rgbs = []
        for img_idx in range(len(self.images)):
            pose = self.poses[img_idx]
            rays_d = (dirs @ pose[:3, :3].T).reshape(-1, 3)
            rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
            rgb_flat = self.images[img_idx].reshape(-1, 3)
            self.all_rays_o.append(rays_o)
            self.all_rays_d.append(rays_d)
            self.all_rgbs.append(rgb_flat)
        self.all_rays_o = np.concatenate(self.all_rays_o, axis=0)
        self.all_rays_d = np.concatenate(self.all_rays_d, axis=0)
        self.all_rgbs = np.concatenate(self.all_rgbs, axis=0)

    def __len__(self):
        return self.all_rays_o.shape[0]

    def __getitem__(self, idx):
        return {
            'ray_o': torch.from_numpy(self.all_rays_o[idx]).float(),
            'ray_d': torch.from_numpy(self.all_rays_d[idx]).float(),
            'rgb': torch.from_numpy(self.all_rgbs[idx]).float()
        }

def auto_tune_batch_size(dataset, max_mem_gb=32, min_batch=64, max_batch=4096):
    # Dummy function: in practice, you would try increasing batch size until OOM, then back off
    # Here, just return a safe default
    return min(max_batch, max(min_batch, int(1e6 / len(dataset)))) 