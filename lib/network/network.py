from __future__ import annotations
import imp
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from .utils import raw2outputs, sample_along_ray, sample_pdf
from lib.config.config import cfg
from .NeRF import NeRF


class Network(nn.Module):
    def __init__(self, down_ratio=2):
        super(Network, self).__init__()
        self.nerf_0 = NeRF(fr_pos=cfg.fr_pos)
    
    def render_rays(self, rays, batch, near=0., far=100., B = 1):
        N_rays = cfg.N_rays
        rays_o, rays_d = rays[:, :, :3], rays[:, :, 3:6]
        scale_factor = torch.norm(rays_d, p=2, dim=2)
        near_depth, far_depth = near * torch.ones((1,rays_d.shape[1])).to(rays), far * torch.ones((1,rays_d.shape[1])).to(rays)
        z_vals = sample_along_ray(near_depth, far_depth, cfg.N_samples)  
        z_vals = z_vals.reshape(z_vals.shape[0], z_vals.shape[1], -1)  # [1, N_rays, N_samples]
        xyz = rays_o[:, :, None] + rays_d[:, :, None] * z_vals[:, :, :, None] / scale_factor[...,None,None]
        xyz /= cfg.dist
        ray_dir = rays[..., 3:6]
        ray_dir = ray_dir[:, :, None].repeat(1, 1, cfg.N_samples, 1)
        raw = self.nerf_0(xyz, ray_dir)
        ret_0 = raw2outputs(raw, z_vals/scale_factor[...,None], rays_d)

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, ret_0['weights'][...,1:-1], cfg.cascade_samples, det=False)
        z_samples = z_samples.detach()  # [1, N_rays, cascade_samples]

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)  # [1, N_rays, samples_all]
        xyz = rays_o[:, :, None] + rays_d[:, :, None] * z_vals[:, :, :, None] / scale_factor[...,None,None]
        xyz /= cfg.dist
        ray_dir = rays[..., 3:6]
        ray_dir = ray_dir[:, :, None].repeat(1, 1, cfg.samples_all, 1)
        raw = self.nerf_0(xyz, ray_dir)
        ret_1 = raw2outputs(raw, z_vals/scale_factor[...,None], rays_d)

        outputs = {}
        for key in ret_1:
            outputs[key + '_0'] = ret_1[key]

        return outputs
    
    def batchify_rays(self, rays, batch):
        all_ret = {}
        chunk = cfg.chunk_size
        for i in range(0, rays.shape[1], chunk):
            ret = self.render_rays(rays[:,i:i+chunk], batch)
            torch.cuda.empty_cache()
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=1) for k in all_ret}
        return all_ret

    def forward(self, batch):
        rays, rgbs = batch['rays'], batch['rays_rgb']
        ret = self.batchify_rays(rays, batch)
        return ret
