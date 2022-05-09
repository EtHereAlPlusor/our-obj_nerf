from __future__ import annotations
import imp
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from .utils import raw2outputs, sample_along_ray, sample_pdf
from lib.config.config import cfg
from .NeRF import NeRF
merge_list_car = [27, 28, 29, 30, 31]
merge_list_box = [39]
merge_list_park = [9]
merge_list_gate = [35]

class Network(nn.Module):
    def __init__(self, down_ratio=2):
        super(Network, self).__init__()
        self.cascade = len(cfg.cascade_samples) > 1
        self.nerf_0 = NeRF(fr_pos=cfg.fr_pos)
    
    def render_rays(self, rays, batch, intersection, near=0., far=1.):
        B, N_rays, _, _ = intersection.shape
        rays_o, rays_d = rays[:, :, :3], rays[:, :, 3:6]
        scale_factor = torch.norm(rays_d, p=2, dim=2)
        near_depth, far_depth = intersection[..., 0].to(rays), intersection[..., 1].to(rays)
        # near_depth, far_depth = near * torch.ones((1,rays_d.shape[1],10)).to(rays), far * torch.ones((1,rays_d.shape[1],10)).to(rays)
        z_vals = sample_along_ray(near_depth, far_depth, cfg.bbox_sp)
        z_vals_bound = torch.cat([z_vals[...,0]-1e-5,z_vals[...,-1]+1e-5],dim=2)
        PDF = torch.ones(z_vals.shape)
        PDF = PDF.reshape(B, N_rays,-1)
        z_vals = z_vals.reshape(z_vals.shape[0], z_vals.shape[1], -1)
        z_vals, sort_index = torch.sort(z_vals,2)
        PDF = torch.take(PDF.to(z_vals), sort_index)
        z_vals = sample_pdf(z_vals, PDF[...,1:], cfg.cascade_samples[0], det=True)
        z_vals = torch.cat([z_vals, z_vals_bound],dim=2)
        idx0_bg, idx1_bg, idx2_bg = torch.where(z_vals<0.)
        z_vals[idx0_bg, idx1_bg, idx2_bg] = cfg.dist + 20 * torch.rand(len(idx0_bg)).to(rays)
        z_vals, _ = torch.sort(z_vals,2)
        xyz = rays_o[:, :, None] + rays_d[:, :, None] * z_vals[:, :, :, None] / scale_factor[...,None,None]
        xyz /= cfg.dist
        ray_dir = rays[..., 3:6]
        ray_dir = ray_dir[:, :, None].repeat(1, 1, cfg.samples_all, 1)
        raw = self.nerf_0(xyz, ray_dir)

        ret_0 = raw2outputs(raw, z_vals/scale_factor[...,None], rays_d)

        outputs = {}
        for key in ret_0:
            outputs[key + '_0'] = ret_0[key]

        return outputs
    
    def batchify_rays(self, rays, batch, intersection):
        all_ret = {}
        chunk = cfg.chunk_size
        for i in range(0, rays.shape[1], chunk):
            ret = self.render_rays(rays[:,i:i+chunk], batch, intersection[:,i:i+chunk])
            torch.cuda.empty_cache()
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=1) for k in all_ret}
        return all_ret

    def forward(self, batch):
        rays, rgbs, intersection = batch['rays'], batch['rays_rgb'], batch['intersection']
        ret = self.batchify_rays(rays, batch, intersection)
        return ret
