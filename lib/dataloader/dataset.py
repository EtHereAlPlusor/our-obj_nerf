import imp
from tkinter import N
import numpy as np
import os
from glob import glob
from lib.dataloader.data_utils import *
from lib.config.config import cfg, args
import imageio
from multiprocessing import Pool
from tools.kitti360scripts.helpers.annotation import Annotation3D
import cv2

class Dataset:
    def __init__(self, cam2world_root, img_root, bbx_root, data_root, sequence, split):
        super(Dataset, self).__init__()
        # path and initialization
        self.split = split
        self.sequence = sequence
        self.start = cfg.start
        self.visible_id = os.path.join(data_root, 'visible_id', sequence)

        # load image_ids
        train_ids = np.arange(self.start, self.start + cfg.train_frames)
        test_ids = np.arange(self.start, self.start + cfg.train_frames)
        test_ids = np.array(cfg.val_list)
        if split == 'train':
            self.image_ids = train_ids
        elif split == 'val':
            self.image_ids = test_ids

        # load intrinsics
        calib_dir = os.path.join(data_root, 'calibration')
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        self.load_intrinsic(self.intrinsic_file)
        self.H = int(self.height * cfg.ratio)
        self.W = int(self.width  * cfg.ratio)
        self.K_00[:2] = self.K_00[:2] * cfg.ratio
        self.K_01[:2] = self.K_01[:2] * cfg.ratio
        self.intrinsic_00 = self.K_00[:, :-1]
        self.intrinsic_01 = self.K_01[:, :-1]
 
        # load cam2world poses
        self.cam2world_dict_00 = {}
        self.cam2world_dict_01 = {}
        self.pose_file = os.path.join(data_root, 'data_poses', sequence, 'poses.txt')
        poses = np.loadtxt(self.pose_file)
        frames = poses[:, 0]
        poses = np.reshape(poses[:, 1:], [-1, 3, 4])
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_01']
        for line in open(cam2world_root, 'r').readlines():
            value = list(map(float, line.strip().split(" ")))
            self.cam2world_dict_00[value[0]] = np.array(value[1:]).reshape(4, 4)
        for frame, pose in zip(frames, poses):
            pose = np.concatenate((pose, np.array([0., 0., 0.,1.]).reshape(1, 4)))
            self.cam2world_dict_01[frame] = np.matmul(np.matmul(pose, self.camToPose), np.linalg.inv(self.R_rect))
        self.translation = np.array(cfg.center_pose)
        if False:
            up = np.array([0,0,-1])
            close_depth, inf_depth = 1, 100
            dt = .75
            mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
            focal = mean_dz
            zdelta = close_depth * .2
            tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
            rads = np.percentile(np.abs(tt), 90, 0)
            c2w_path = self.cam2world_dict_00[self.spiral_frame] #c2w
            N_views = self.spiral_frame_num
            N_rots = 2
            render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
            self.render_poses = np.array(render_poses).astype(np.float32)

        # load images
        self.visible_id = os.path.join(data_root, 'visible_id', sequence)
        self.images_list_00 = {}
        self.images_list_01 = {}
        for idx in self.image_ids:
            frame_name = '%010d' % idx
            if os.path.exists(os.path.join(self.visible_id,frame_name + '.txt')) == False:
                continue
            image_file_00 = os.path.join(img_root, 'image_00/data_rect/%s.png' % frame_name)
            image_file_01 = os.path.join(img_root, 'image_01/data_rect/%s.png' % frame_name)
            if not os.path.isfile(image_file_00):
                raise RuntimeError('%s does not exist!' % image_file_00)
            self.images_list_00[idx] = image_file_00
            self.images_list_01[idx] = image_file_01

        # load annotation3D
        self.annotation3D = Annotation3D(bbx_root, sequence)
        self.bbx_static = {}
        self.bbx_static_annotationId = []
        self.bbx_static_center = []
        for annotationId in self.annotation3D.objects.keys():
            if len(self.annotation3D.objects[annotationId].keys()) == 1:
                if -1 in self.annotation3D.objects[annotationId].keys():
                    self.bbx_static[annotationId] = self.annotation3D.objects[annotationId][-1]
                    self.bbx_static_annotationId.append(annotationId)
        self.bbx_static_annotationId = np.array(self.bbx_static_annotationId)

        # load metas
        self.build_metas(self.cam2world_dict_00, self.cam2world_dict_01, self.images_list_00, self.images_list_01)

    def load_intrinsic(self, intrinsic_file):
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                self.K_00 = K
            elif line[0] == 'P_rect_01:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                intrinsic_loaded = True
                self.K_01 = K
            elif line[0] == 'R_rect_01:':
                R_rect = np.eye(4)
                R_rect[:3, :3] = np.array([float(x) for x in line[1:]]).reshape(3, 3)
            elif line[0] == "S_rect_01:":
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert (intrinsic_loaded == True)
        assert (width > 0 and height > 0)
        self.width, self.height = width, height
        self.R_rect = R_rect

    def build_metas(self, cam2world_dict_00, cam2world_dict_01, images_list_00, images_list_01):
        input_tuples = []
        for idx, frameId in enumerate(self.image_ids):
            pose = cam2world_dict_00[frameId]
            pose[:3, 3] = pose[:3, 3] - self.translation
            image_path = images_list_00[frameId]
            image = (np.array(imageio.imread(image_path)) / 255.).astype(np.float32)
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            rays = build_rays(self.intrinsic_00, pose, image.shape[0], image.shape[1])
            rays_rgb = image.reshape(-1, 3)
            input_tuples.append((rays, rays_rgb, frameId, self.intrinsic_00))
        print('load meta_00 done')
    
        if cfg.use_stereo == True:
            for idx, frameId in enumerate(self.image_ids):
                pose = cam2world_dict_01[frameId]
                pose[:3, 3] = pose[:3, 3] - self.translation
                image_path = images_list_01[frameId]
                image = (np.array(imageio.imread(image_path)) / 255.).astype(np.float32)
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                rays = build_rays(self.intrinsic_00, pose, image.shape[0], image.shape[1])
                rays_rgb = image.reshape(-1, 3)
                input_tuples.append((rays, rays_rgb, frameId, self.intrinsic_01))
            print('load meta_01 done')
        self.metas = input_tuples

    def __getitem__(self, index):
        rays, rays_rgb, frameId, intrinsics = self.metas[index]
        if self.split == 'train':
            rand_ids = np.random.permutation(len(rays))
            rays = rays[rand_ids[:cfg.N_rays]]
            rays_rgb = rays_rgb[rand_ids[:cfg.N_rays]]

        ret = {
            'rays': rays.astype(np.float32),
            'rays_rgb': rays_rgb.astype(np.float32),
            'intrinsics': intrinsics.astype(np.float32),
            'meta': {
                'sequence': '{}'.format(self.sequence)[0],
                'tar_idx': frameId,
                'h': self.H,
                'w': self.W
            }
        }
        return ret

    def __len__(self):
        return len(self.metas)
