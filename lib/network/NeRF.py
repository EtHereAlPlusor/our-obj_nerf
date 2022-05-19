import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config.config import cfg
from .embedder import get_embedder

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, fr_pos=10, fr_view=4, skips=[4]):
        """
        """
        super(NeRF, self).__init__()
        self.skips = skips
        self.pe0, input_ch = get_embedder(fr_pos, 0)
        self.pe1, input_ch_views = get_embedder(fr_view, 0)
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + \
            [nn.Linear(W, W) if i not in self.skips else \
             nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.feature_linear = nn.Linear(W, W)
        self.sigma_linear = nn.Linear(W, 1)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        self.rgb_linear = nn.Linear(W//2, 3)

        self.views_linears.apply(weights_init)
        self.pts_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.sigma_linear.apply(weights_init)
        self.rgb_linear.apply(weights_init)

    def forward(self, xyz, ray_dir):
        B, N_rays, N_samples = xyz.shape[:3]
        xyz, ray_dir = xyz.reshape(-1, 3), ray_dir.reshape(-1, 3)
        ray_dir = ray_dir / ray_dir.norm(dim=-1, keepdim=True)

        input_pts, input_views = self.pe0(xyz), self.pe1(ray_dir)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # sigma
        sigma = self.sigma_linear(h)
        feature = self.feature_linear(h)

        # rgb
        h = torch.cat([feature, input_views], -1)
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)
        rgb = self.rgb_linear(h)  

        outputs = torch.cat([rgb, sigma], -1)
        return outputs.reshape(B, N_rays, N_samples, 4)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)