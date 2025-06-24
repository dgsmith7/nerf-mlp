import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class NeRFDataset(Dataset):
    def __init__(self, datadir, split='train', img_wh=(400, 400)):
        super().__init__()
        self.datadir = datadir
        self.split = split
        self.img_wh = img_wh
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
            img = Image.open(fname).convert('RGB')
            img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
            img = np.array(img) / 255.0
            self.images.append(img)
            pose = np.array(frame['transform_matrix'], dtype=np.float32)
            self.poses.append(pose)
        self.images = np.stack(self.images, axis=0)  # (N_images, H, W, 3)
        self.poses = np.stack(self.poses, axis=0)    # (N_images, 4, 4)
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