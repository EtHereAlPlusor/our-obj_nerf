from cmath import inf
import torch
import torch.nn as nn
from lib.config.config import cfg
from torch.nn import functional as F
import numpy as np

class Loss(nn.Module):
    def __init__(self, net):
        super(Loss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net  # NeRF
        self.color_crit = nn.MSELoss(reduction='none')
        self.opacity_crit = nn.MSELoss(reduction='none')
        self.device = device

    def forward(self, batch):
        output = self.net(batch)
        scalar_stats = {}
        loss = 0

        mask = torch.zeros((1, cfg.N_rays)).to(self.device)
        bkgd = torch.ones((1, cfg.N_rays)).to(self.device)
        for rays_id in range(cfg.N_rays):
            pixel = batch['pixel_ids'][0, rays_id]
            if batch['instance_ids'][0, pixel] == cfg.instance_id:
                mask[0, rays_id] = 1.
                bkgd[0, rays_id] = 0.

        # rgb loss, joint optimization
        if 'rgb_fine_scn' and 'rgb_fine_obj' in output.keys():
        # if 'rgb_fine_scn' or 'rgb_fine_obj' in output.keys():
            color_loss_fine_scn = cfg.train.weight_color * torch.sum(self.color_crit(batch['rays_rgb'], output['rgb_fine_scn']), 2)
            color_loss_fine_obj = cfg.train.weight_color * mask * torch.sum(self.color_crit(batch['rays_rgb'], output['rgb_fine_obj']), 2)
            color_loss = torch.mean((cfg.train.weight_scn*color_loss_fine_scn + color_loss_fine_obj))
            # color_loss = torch.mean((color_loss_fine_obj))
            scalar_stats.update({'color': color_loss})
            loss += color_loss
            psnr_scn = -10.*torch.log10(torch.mean(color_loss_fine_scn).detach()).to(color_loss.device)
            psnr_obj = -10.*torch.log10(torch.mean(color_loss_fine_obj).detach()).to(color_loss.device)
            psnr_scn = torch.where(torch.isinf(psnr_scn), torch.full_like(psnr_scn, 0), psnr_scn)
            psnr_obj = torch.where(torch.isinf(psnr_obj), torch.full_like(psnr_obj, 0), psnr_obj)
            scalar_stats.update({'psnr_scn': psnr_scn})
            scalar_stats.update({'psnr_obj': psnr_obj})
        
        # opacity loss
        if 'opacity_fine_obj' in output.keys():
            opacity_loss_fine_obj = cfg.train.weight_opacity * self.opacity_crit(mask, output['opacity_fine_obj'])
            opacity_loss = torch.mean(opacity_loss_fine_obj)
            scalar_stats.update({'opacity': opacity_loss})
            loss += opacity_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

