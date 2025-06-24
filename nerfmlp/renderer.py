import torch
import torch.nn.functional as F
from .model import PositionalEncoding

class NeRFRenderer:
    def __init__(self, model, device, 
                 pos_enc_L=10, dir_enc_L=4, 
                 N_samples=64, N_importance=64, 
                 near=2.0, far=6.0, white_bkgd=True):
        self.model = model
        self.device = device
        self.N_samples = N_samples
        self.N_importance = N_importance
        self.near = near
        self.far = far
        self.white_bkgd = white_bkgd
        self.pos_enc = PositionalEncoding(pos_enc_L).to(device)
        self.dir_enc = PositionalEncoding(dir_enc_L).to(device)

    def render(self, rays_o, rays_d, H, W, focal, chunk=1024*32):
        """
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        H, W: image height and width
        focal: focal length
        chunk: process rays in chunks to avoid OOM
        Returns: (H, W, 3) image
        """
        N_rays = rays_o.shape[0]
        results = []
        for i in range(0, N_rays, chunk):
            results.append(self._render_rays(
                rays_o[i:i+chunk], rays_d[i:i+chunk]))
        rgb_map = torch.cat([r['rgb_map'] for r in results], 0)
        return rgb_map.view(H, W, 3)

    def _render_rays(self, rays_o, rays_d):
        # Stratified sampling along each ray
        N_rays = rays_o.shape[0]
        t_vals = torch.linspace(0., 1., steps=self.N_samples, device=self.device)
        z_vals = self.near * (1.-t_vals) + self.far * t_vals
        z_vals = z_vals.expand([N_rays, self.N_samples])
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
        # Positional encoding
        pts_flat = pts.reshape(-1, 3)
        pts_enc = self.pos_enc(pts_flat)
        # View direction encoding
        viewdirs = rays_d / (rays_d.norm(dim=-1, keepdim=True) + 1e-8)
        viewdirs_enc = self.dir_enc(viewdirs)
        viewdirs_enc = viewdirs_enc.unsqueeze(1).expand(-1, self.N_samples, -1).reshape(-1, viewdirs_enc.shape[-1])
        # Query model
        raw = self.model(pts_enc, viewdirs_enc)
        raw = raw.view(N_rays, self.N_samples, 4)
        rgb = torch.sigmoid(raw[..., :3])
        sigma = F.relu(raw[..., 3])
        # Volume rendering
        deltas = z_vals[..., 1:] - z_vals[..., :-1]
        delta_inf = 1e10 * torch.ones_like(deltas[..., :1])
        deltas = torch.cat([deltas, delta_inf], -1)
        alpha = 1. - torch.exp(-sigma * deltas)
        T = torch.cumprod(torch.cat([torch.ones((N_rays, 1), device=self.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        weights = alpha * T
        rgb_map = (weights.unsqueeze(-1) * rgb).sum(dim=1)
        depth_map = (weights * z_vals).sum(dim=1)
        acc_map = weights.sum(dim=1)
        if self.white_bkgd:
            rgb_map = rgb_map + (1. - acc_map.unsqueeze(-1))
        return {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map} 