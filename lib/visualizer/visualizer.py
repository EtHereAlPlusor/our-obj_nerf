import numpy as np
import torch
import cv2
import os
import imageio

from tools.kitti360scripts.helpers.labels import id2label
from lib.config.config import cfg

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
            img_id = int(batch["meta"]["tar_idx"].item())
            scn_pred_img = torch.clamp(output['rgb_fine_scn'][b], min=0.,max=1.).reshape(h, w, 3).detach().cpu().numpy()
            scn_pred_depth = output['depth_fine_scn'][b].reshape(h, w).detach().cpu().numpy()
            obj_pred_img = torch.clamp(output['rgb_fine_obj'][b], min=0.,max=1.).reshape(h, w, 3).detach().cpu().numpy()
            obj_pred_depth = output['depth_fine_obj'][b].reshape(h, w).detach().cpu().numpy()

            result_dir = cfg.result_dir
            print(result_dir)
            gt_result_dir = os.path.join(result_dir, batch['meta']['sequence'][0], 'ground_truth')
            scn_depth_result_dir = os.path.join(result_dir, batch['meta']['sequence'][0], 'depth/scn')
            scn_rgb_result_dir = os.path.join(result_dir, batch['meta']['sequence'][0], 'rgb/scn')
            obj_depth_result_dir = os.path.join(result_dir, batch['meta']['sequence'][0], 'depth/obj')
            obj_rgb_result_dir = os.path.join(result_dir, batch['meta']['sequence'][0], 'rgb/obj')
            
            os.system("mkdir -p {}".format(gt_result_dir))
            imageio.imwrite('{}/img{:04d}_gt.png'.format(gt_result_dir, img_id), to8b(gt_img))

            # scene
            os.system("mkdir -p {}".format(scn_depth_result_dir))
            scn_pred_depth = cv2.applyColorMap(cv2.convertScaleAbs(((scn_pred_depth/scn_pred_depth.max()) * 255).astype(np.uint8),alpha=2), cv2.COLORMAP_JET)
            imageio.imwrite('{}/img{:04d}_depth.png'.format(scn_depth_result_dir, img_id), scn_pred_depth)
            os.system("mkdir -p {}".format(scn_rgb_result_dir))
            imageio.imwrite('{}/img{:04d}_rgb.png'.format(scn_rgb_result_dir, img_id), to8b(scn_pred_img))

            # object
            os.system("mkdir -p {}".format(obj_depth_result_dir))
            obj_pred_depth = cv2.applyColorMap(cv2.convertScaleAbs(((obj_pred_depth/obj_pred_depth.max()) * 255).astype(np.uint8),alpha=2), cv2.COLORMAP_JET)
            imageio.imwrite('{}/img{:04d}_depth.png'.format(obj_depth_result_dir, img_id), obj_pred_depth)
            os.system("mkdir -p {}".format(obj_rgb_result_dir))
            imageio.imwrite('{}/img{:04d}_rgb.png'.format(obj_rgb_result_dir, img_id), to8b(obj_pred_img))