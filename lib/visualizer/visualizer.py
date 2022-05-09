import numpy as np
import torch
import cv2
from lib.config.config import cfg
import os
from tools.kitti360scripts.helpers.labels import id2label

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class Visualizer:
    def __init__(self, ):
        self.color_crit = lambda x, y: ((x - y)**2).mean()
        self.mse2psnr = lambda x: -10. * np.log(x) / np.log(torch.tensor([10.]))
        self.psnr = []

    def visualize(self, output, batch):
        b = len(batch['rays'])
        for b in range(b):
            h, w = batch['meta']['h'][b].item(), batch['meta']['w'][b].item()
            gt_img = batch['rays_rgb'][b].reshape(h, w, 3).detach().cpu().numpy()
            pred_img = torch.clamp(output['rgb_0'][b], min=0.,max=1.).reshape(h, w, 3).detach().cpu().numpy()
            pred_depth = output['depth_0'][b].reshape(h, w).detach().cpu().numpy()
            img_id = int(batch["meta"]["tar_idx"].item())
            result_dir = cfg.result_dir
            depth_result_dir = os.path.join(result_dir, batch['meta']['sequence'][0], 'depth')
            rgb_result_dir = os.path.join(result_dir, batch['meta']['sequence'][0], 'rgb')
            gt_result_dir = os.path.join(result_dir, batch['meta']['sequence'][0], 'ground_truth')
            print(result_dir)

            os.system("mkdir -p {}".format(depth_result_dir))
            # np.save('{}/img{:04d}_depth.npy'.format(depth_result_dir, img_id),(pred_depth))
            pred_depth = cv2.applyColorMap(cv2.convertScaleAbs(((pred_depth/pred_depth.max()) * 255).astype(np.uint8),alpha=2), cv2.COLORMAP_JET)
            cv2.imwrite('{}/img{:04d}_depth.png'.format(depth_result_dir, img_id), pred_depth)

            os.system("mkdir -p {}".format(rgb_result_dir))
            cv2.imwrite('{}/img{:04d}_rgb.png'.format(rgb_result_dir, img_id), to8b(pred_img))

            os.system("mkdir -p {}".format(gt_result_dir))
            cv2.imwrite('{}/img{:04d}_gt.png'.format(gt_result_dir, img_id), to8b(gt_img))