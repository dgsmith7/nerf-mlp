import torch
import torch.nn.functional as F
from .model import PositionalEncoding

class NeRFRenderer:
    def __init__(self, model, device, 
                 pos_enc_L=10, dir_enc_L=4, 
                 N_samples=64, N_importance=128, 
                 near=2.0, far=6.0, white_bkgd=True, perturb=1.0, raw_noise_std=0.0, coord_scale=1.0):
        self.model = model
        self.device = device
        self.N_samples = N_samples
        self.N_importance = N_importance
        self.near = near
        self.far = far
        self.white_bkgd = white_bkgd
        self.perturb = perturb
        self.raw_noise_std = raw_noise_std
        self.coord_scale = coord_scale  # New: coordinate scaling factor
        self.pos_enc = PositionalEncoding(pos_enc_L).to(device)
        self.dir_enc = PositionalEncoding(dir_enc_L).to(device)

    def render(self, rays_o, rays_d, H, W, focal, chunk=1024*16):
        """
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        H, W: image height and width
        focal: focal length
        chunk: process rays in chunks to avoid OOM (optimized for M3 Pro)
        Returns: (H, W, 3) image
        """
        N_rays = rays_o.shape[0]
        results = []
        
        # Optimize for M3 Pro MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Use smaller chunks for better M3 Pro performance
            chunk = min(chunk, 1024*8)
        
        for i in range(0, N_rays, chunk):
            with torch.no_grad():  # Ensure inference mode for better performance
                results.append(self._render_rays(
                    rays_o[i:i+chunk], rays_d[i:i+chunk]))
        rgb_map = torch.cat([r['rgb_map'] for r in results], 0)
        return rgb_map.view(H, W, 3)

    def _render_rays(self, rays_o, rays_d):
        N_rays = rays_o.shape[0]
        device = self.device
        
        # === Coarse sampling ===
        t_vals = torch.linspace(0., 1., steps=self.N_samples, device=device)
        z_vals = self.near * (1.-t_vals) + self.far * t_vals
        z_vals = z_vals.expand([N_rays, self.N_samples])
        
        if self.perturb > 0:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
        pts_flat = pts.reshape(-1, 3)
        
        # Apply coordinate scaling to match model expectations
        if self.coord_scale != 1.0:
            pts_flat = pts_flat * self.coord_scale
        
        pts_enc = self.pos_enc(pts_flat)
        
        viewdirs = rays_d / (rays_d.norm(dim=-1, keepdim=True) + 1e-8)
        viewdirs_enc = self.dir_enc(viewdirs)
        viewdirs_enc = viewdirs_enc.unsqueeze(1).expand(-1, self.N_samples, -1).reshape(-1, viewdirs_enc.shape[-1])
        
        raw = self.model(pts_enc, viewdirs_enc)
        raw = raw.view(N_rays, self.N_samples, 4)
        
        rgb_map_coarse, depth_map_coarse, acc_map_coarse, weights = self._raw2outputs(
            raw, z_vals, rays_d)

        # === Fine sampling (hierarchical) ===
        if self.N_importance > 0:
            rgb_map_0, depth_map_0, acc_map_0 = rgb_map_coarse, depth_map_coarse, acc_map_coarse
            
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = self._sample_pdf(z_vals_mid, weights[..., 1:-1], self.N_importance, det=(self.perturb == 0.))
            z_samples = z_samples.detach()

            z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_fine.unsqueeze(2)
            pts_fine_flat = pts_fine.reshape(-1, 3)
            
            # Apply coordinate scaling to match model expectations
            if self.coord_scale != 1.0:
                pts_fine_flat = pts_fine_flat * self.coord_scale
            
            pts_fine_enc = self.pos_enc(pts_fine_flat)
            
            viewdirs_fine_enc = self.dir_enc(viewdirs)
            viewdirs_fine_enc = viewdirs_fine_enc.unsqueeze(1).expand(-1, z_vals_fine.shape[1], -1).reshape(-1, viewdirs_fine_enc.shape[-1])
            
            raw_fine = self.model(pts_fine_enc, viewdirs_fine_enc)
            raw_fine = raw_fine.view(N_rays, z_vals_fine.shape[1], 4)
            
            rgb_map_fine, depth_map_fine, acc_map_fine, weights_fine = self._raw2outputs(
                raw_fine, z_vals_fine, rays_d)
                
            return {'rgb_map': rgb_map_fine, 'depth_map': depth_map_fine, 'acc_map': acc_map_fine,
                    'rgb_map_coarse': rgb_map_0, 'depth_map_coarse': depth_map_0, 'acc_map_coarse': acc_map_0}
        else:
            return {'rgb_map': rgb_map_coarse, 'depth_map': depth_map_coarse, 'acc_map': acc_map_coarse}

    def _raw2outputs(self, raw, z_vals, rays_d):
        """
        Transforms model's predictions to semantically meaningful values.
        Based exactly on the official NeRF implementation.
        """
        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        
        # The 'distance' from the last integration time is infinity.
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], -1)
        
        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        # Extract RGB of each sample position along each ray.
        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        
        # Add noise to model's predictions for density. Can be used to 
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if self.raw_noise_std > 0.:
            noise = torch.randn_like(raw[..., 3]) * self.raw_noise_std
        
        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = 1.0 - torch.exp(-F.relu(raw[..., 3] + noise) * dists)
        
        # Compute weight for RGB of each sample along each ray. A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # This is the key fix: use exclusive=True equivalent
        ones = torch.ones_like(alpha[..., :1])
        transmittance = torch.cumprod(torch.cat([ones, 1. - alpha + 1e-10], -1), -1)[..., :-1]
        weights = alpha * transmittance
        
        # Computed weighted color of each sample along each ray.
        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
        
        # Estimated depth map is expected distance.
        depth_map = torch.sum(weights * z_vals, dim=-1)
        
        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = torch.sum(weights, -1)
        
        # To composite onto a white background, use the accumulated alpha map.
        if self.white_bkgd:
            rgb_map = rgb_map + (1. - acc_map.unsqueeze(-1))
        
        return rgb_map, depth_map, acc_map, weights

    def _sample_pdf(self, bins, weights, N_samples, det=False):
        """
        Hierarchical sampling based on the official NeRF implementation.
        """
        device = weights.device
        
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
        
        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., N_samples, device=device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)
        
        # Invert CDF
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, 0)
        above = torch.clamp(inds, max=cdf.shape[-1] - 1)
        inds_g = torch.stack([below, above], -1)
        
        matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
        bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)
        
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        
        return samples 