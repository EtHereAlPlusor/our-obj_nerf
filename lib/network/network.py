import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from .utils import raw2outputs, sample_along_ray, sample_pdf
from lib.config.config import cfg
from .NeRF import NeRF


class Network(nn.Module):
    """Structure of the network

    Using two NeRF MLPs respectively to render the scene and objects.
    
    """
    def __init__(self):
        super(Network, self).__init__()
        self.nerf_scn = NeRF(fr_pos=cfg.fr_pos)
        self.nerf_obj = NeRF(fr_pos=cfg.fr_pos)
    
    def render_rays(self, rays, batch, is_editing=False, near=0., far=100.):
        """render the rays
        
        Sample on the rays and query the points' colors c and densities sigma, 
        and then integral c and sigma to obtain the color of the ray

        Args:
            rays: contain rays_o and rays_d
            batch:
            is_editing: bool, specify the strategy that how we feed the two branches
                        if False, we use the sampled points to train the MLPs, and obtain outputs respectively
                        if True, we use different points on the ray to query the MLPs respectively, 
                                 and then cat them to render the whole scene
            near: float, specify the near bound we sample the points
            far: float, specify the far bound we sample the points
        
        Returns:
            outputs: dict, varies according to is_editing
                     if is_editing is False, will return 'rgb', 'depth', 'opacity' and 'weights' of the two branchs 
                                             without edits
                     if is_editing is True, will return 'rgb', 'depth', 'opacity' and 'weights' of the whole scene
                                             after editing
        
        """
        rays_o, rays_d = rays[:, :, :3], rays[:, :, 3:6]
        scale_factor = torch.norm(rays_d, p=2, dim=2)
        near_depth, far_depth = near * torch.ones((1,rays_d.shape[1])).to(rays), far * torch.ones((1,rays_d.shape[1])).to(rays)
        z_vals = sample_along_ray(near_depth, far_depth, cfg.N_samples)  
        z_vals = z_vals.reshape(z_vals.shape[0], z_vals.shape[1], -1)  # [1, N_rays, N_samples]
        xyz = rays_o[:, :, None] + rays_d[:, :, None] * z_vals[:, :, :, None] / scale_factor[...,None,None]
        xyz /= cfg.dist
        ray_dir = rays[..., 3:6]
        ray_dir = ray_dir[:, :, None].repeat(1, 1, cfg.N_samples, 1)
        outputs = {}

        if is_editing == False:
            raw_coarse = self.nerf_obj(xyz, ray_dir)
            ret_coarse = raw2outputs(raw_coarse, z_vals/scale_factor[...,None], rays_d)

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, ret_coarse['weights'][...,1:-1], cfg.cascade_samples, det=False)
            z_samples = z_samples.detach()  # [1, N_rays, cascade_samples]
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)  # [1, N_rays, samples_all]
            xyz = rays_o[:, :, None] + rays_d[:, :, None] * z_vals[:, :, :, None] / scale_factor[...,None,None]
            xyz /= cfg.dist
            ray_dir = rays[..., 3:6]
            ray_dir = ray_dir[:, :, None].repeat(1, 1, cfg.samples_all, 1)

            # object
            raw_fine = self.nerf_obj(xyz, ray_dir)
            ret_fine = raw2outputs(raw_fine, z_vals/scale_factor[...,None], rays_d)
            for key in ret_fine:
                outputs[key + '_fine_obj'] = ret_fine[key]

            # scene
            raw_fine = self.nerf_scn(xyz, ray_dir)
            ret_fine = raw2outputs(raw_fine, z_vals/scale_factor[...,None], rays_d)
            for key in ret_fine:
                outputs[key + '_fine_scn'] = ret_fine[key]

        else:
            # object
            offset = cfg.offset * torch.ones(xyz[:,:,:,1].shape).to(xyz.device) / cfg.dist
            y = (xyz[:,:,:,1] - offset).reshape(xyz.shape[0], xyz.shape[1], xyz.shape[2], 1)
            xyz_obj = torch.cat([torch.cat([xyz[:,:,:,0].unsqueeze(-1), y], -1), xyz[:,:,:,2].unsqueeze(-1)], -1)
            raw_coarse = self.nerf_obj(xyz_obj, ray_dir)
            ret_coarse = raw2outputs(raw_coarse, z_vals/scale_factor[...,None], rays_d)

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, ret_coarse['weights'][...,1:-1], cfg.cascade_samples, det=False)
            z_samples = z_samples.detach()  # [1, N_rays, cascade_samples]
            z_vals_obj, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)  # [1, N_rays, samples_all]
            xyz_obj = rays_o[:, :, None] + rays_d[:, :, None] * z_vals_obj[:, :, :, None] / scale_factor[...,None,None]
            xyz_obj /= cfg.dist
            offset = cfg.offset * torch.ones(xyz_obj[:,:,:,1].shape).to(xyz.device) / cfg.dist
            y = (xyz_obj[:,:,:,1] - offset).reshape(xyz_obj.shape[0], xyz_obj.shape[1], xyz_obj.shape[2], 1)
            xyz_obj = torch.cat([torch.cat([xyz_obj[:,:,:,0].unsqueeze(-1), y], -1), xyz_obj[:,:,:,2].unsqueeze(-1)], -1)
            ray_dir_obj = rays[..., 3:6]
            ray_dir_obj = ray_dir_obj[:, :, None].repeat(1, 1, cfg.samples_all, 1)

            raw_obj = self.nerf_obj(xyz_obj, ray_dir_obj)

            # scene
            raw_coarse = self.nerf_scn(xyz, ray_dir)
            ret_coarse = raw2outputs(raw_coarse, z_vals/scale_factor[...,None], rays_d)

            z_samples = sample_pdf(z_vals_mid, ret_coarse['weights'][...,1:-1], cfg.cascade_samples, det=False)
            z_samples = z_samples.detach()  # [1, N_rays, cascade_samples]
            z_vals_scn, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)  # [1, N_rays, samples_all]
            xyz_scn = rays_o[:, :, None] + rays_d[:, :, None] * z_vals_scn[:, :, :, None] / scale_factor[...,None,None]
            xyz_scn /= cfg.dist
            ray_dir_scn = rays[..., 3:6]
            ray_dir_scn = ray_dir_scn[:, :, None].repeat(1, 1, cfg.samples_all, 1)

            raw_scn = self.nerf_scn(xyz_scn, ray_dir_scn)

            # combine two branches
            z_vals = torch.cat([z_vals_obj, z_vals_scn], -1)
            # z_vals, indices = torch.sort(torch.cat([z_vals_obj, z_vals_scn], -1), -1)
            raw = torch.cat([raw_obj, raw_scn], -2)
            # raw_combine = torch.zeros_like(raw)
            # for i, ray_index in enumerate(indices[0]):
            #     for j, pts_index in enumerate(ray_index):
            #         raw_combine[0, i, j] = raw[0, i, pts_index]
            ret = raw2outputs(raw, z_vals/scale_factor[...,None], rays_d)
            # ret = raw2outputs(raw_obj, z_vals_obj/scale_factor[...,None], rays_d)
            for key in ret:
                outputs[key + '_fine'] = ret[key]

        return outputs
    
    def batchify_rays(self, rays, batch, is_editing=False):
        all_ret = {}
        chunk = cfg.chunk_size
        for i in range(0, rays.shape[1], chunk):
            ret = self.render_rays(rays[:,i:i+chunk], batch, is_editing)
            torch.cuda.empty_cache()
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=1) for k in all_ret}
        return all_ret

    def forward(self, batch, is_editing=False):
        rays, rgbs = batch['rays'], batch['rays_rgb']
        ret = self.batchify_rays(rays, batch, is_editing)
        return ret