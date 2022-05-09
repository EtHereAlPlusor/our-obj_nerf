from operator import imod
import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config.config import cfg
from torch.nn import functional as F
import math

class Loss(nn.Module):
    def __init__(self, net):
        super(Loss, self).__init__()
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        self.weights_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
        self.epsilon_max = 1.0
        self.epsilon_min = 0.2
        self.decay_speed = 0.00005
    
    def get_gaussian(self, depth_gt, depth_samples):
        return torch.exp(-(depth_gt - depth_samples)**2 / (2*self.epsilon**2))

    def get_weights_gt(self, depth_gt, depth_samples):
        # near
        depth_gt = depth_gt.view(*depth_gt.shape, 1)
        weights = self.get_gaussian(depth_gt, depth_samples).detach()
        # empty and dist
        weights[torch.abs(depth_samples-depth_gt)>self.epsilon]=0
        # normalize
        weights = weights / torch.sum(weights,dim=2,keepdims=True).clamp(min=1e-6)
        return weights.detach()

    def kl_loss(self, weights_gt, weights_es):
        return torch.log(weights_gt * weights_es).sum()

    def forward(self, batch):
        output = self.net(batch)
        scalar_stats = {}
        loss = 0
        
        # rgb loss
        if 'rgb_0' in output.keys():
            color_loss = cfg.train.weight_color * self.color_crit(batch['rays_rgb'], output['rgb_0'])
            scalar_stats.update({'color_mse_0': color_loss})
            loss += color_loss
            psnr = -10. * torch.log(color_loss.detach()) / torch.log(torch.Tensor([10.]).to(color_loss.device))
            scalar_stats.update({'psnr_0': psnr})
        
        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

