import numpy as np
import os
import cv2
import imageio

from lib.dataloader.data_utils import *
from lib.config.config import cfg
from tools.kitti360scripts.helpers.annotation import Annotation3D

class Dataset:
    """Load and prepare data from specified path
    
    """

    def __init__(self, cam2world_root, img_root, instance_root, bbx_root, data_root, sequence, split):
        super(Dataset, self).__init__()
        # path and initialization
        self.split = split
        self.sequence = sequence
        self.start = cfg.start
        self.visible_id = os.path.join(data_root, 'visible_id', sequence)

        # load image_ids
        train_ids = np.arange(self.start, self.start + cfg.train_frames)
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
        self.intrinsic = self.K_00[:, :-1]
 
        # load cam2world poses
        self.cam2world_dict = {}
        self.pose_file = os.path.join(data_root, 'data_poses', sequence, 'poses.txt')
        poses = np.loadtxt(self.pose_file)
        frames = poses[:, 0]
        poses = np.reshape(poses[:, 1:], [-1, 3, 4])
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_01']
        for line in open(cam2world_root, 'r').readlines():
            value = list(map(float, line.strip().split(" ")))
            self.cam2world_dict[value[0]] = np.array(value[1:]).reshape(4, 4)

        # load images and instance
        self.images_list = {}
        self.instance_list = {}
        for idx in self.image_ids:
            frame_name = '%010d' % idx
            image_file = os.path.join(img_root, 'image_00/data_rect/%s.png' % frame_name)
            instance_file = os.path.join(instance_root, '%s.png' % frame_name)
            if not os.path.isfile(image_file):
                raise RuntimeError('%s does not exist!' % image_file)
            elif not os.path.isfile(instance_file):
                raise RuntimeError('%s does not exist!' % instance_file)
            self.images_list[idx] = image_file
            self.instance_list[idx] = instance_file

        # load annotation3D
        self.annotation3D = Annotation3D(bbx_root, sequence)
        self.bbx_static = {}
        self.bbx_static_globalId = []
        for globalId in self.annotation3D.objects.keys():               # global ID
            if len(self.annotation3D.objects[globalId].keys()) == 1:    
                if -1 in self.annotation3D.objects[globalId].keys():    # timestamp
                    self.bbx_static[globalId] = self.annotation3D.objects[globalId][-1]  # obj
                    self.bbx_static_globalId.append(globalId)
        self.bbx_static_globalId = np.array(self.bbx_static_globalId)
        if cfg.instance_id not in self.bbx_static_globalId:
            raise RuntimeError('Did not find the instance!')
        self.bbx_static_vertices = self.bbx_static[cfg.instance_id].vertices # x=forward, y=left, z=up
        line_center = np.zeros((4,3))
        for line_id in range(4):
            line_center[line_id, :] = 0.5*np.array([np.sum(self.bbx_static_vertices[2*line_id:2*line_id+2,0]), 
                                                    np.sum(self.bbx_static_vertices[2*line_id:2*line_id+2,1]),
                                                    np.sum(self.bbx_static_vertices[2*line_id:2*line_id+2,2])])
        self.bbx_static_center = np.zeros((1,3))
        for vertex_id in range(3):
            self.bbx_static_center[0, vertex_id] = 0.25*np.sum(line_center[:, vertex_id])       
        self.bbx_static_R = self.bbx_static[cfg.instance_id].R
        self.bbx_static_T = self.bbx_static[cfg.instance_id].T

        # load metas
        self.build_metas(self.cam2world_dict, self.images_list, self.instance_list)

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

    def build_metas(self, cam2world_dict, images_list, instance_list):
        input_tuples = []
        frameCount = 0
        translation = np.zeros((1,3))
        for idx, frameId in enumerate(self.image_ids):
            frameCount = frameCount + 1
            pose = cam2world_dict[frameId]
            translation = translation + pose[:3, 3]
        translation = translation / frameCount
        for idx, frameId in enumerate(self.image_ids):
            pose = cam2world_dict[frameId]
            pose[:3, 3] = pose[:3, 3] - translation
            image_path = images_list[frameId]
            image = (np.array(imageio.imread(image_path)) / 255.).astype(np.float32)
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            instance_path = instance_list[frameId]
            instance = imageio.imread(instance_path).astype(np.float32)
            instance = cv2.resize(instance, (self.W, self.H), interpolation=cv2.INTER_AREA)
            rays = build_rays(self.intrinsic, pose, image.shape[0], image.shape[1])
            rays_rgb = image.reshape(-1, 3)
            instance_ids = []
            for row in range(self.H):
                for col in range(self.W):
                    instance_ids.append(instance[row, col])
            instance_ids = np.array(instance_ids)
            input_tuples.append((rays, rays_rgb, frameId, self.intrinsic, instance_ids))
        # print('load meta_00 done')
    
        self.metas = input_tuples

    def __getitem__(self, index):
        rays, rays_rgb, frameId, intrinsics, instance_ids = self.metas[index]
        if self.split == 'train':
            rand_ids = np.random.permutation(len(rays))
            rays = rays[rand_ids[:cfg.N_rays]]
            rays_rgb = rays_rgb[rand_ids[:cfg.N_rays]]
            pixel_ids = rand_ids[:cfg.N_rays]
        else:
            pixel_ids = []

        ret = {
            'rays': rays.astype(np.float32),
            'rays_rgb': rays_rgb.astype(np.float32),
            'intrinsics': intrinsics.astype(np.float32),
            'meta': {
                'sequence': '{}'.format(self.sequence)[0],
                'tar_idx': frameId,
                'h': self.H,
                'w': self.W
            },
            'instance_ids': instance_ids,
            'pixel_ids': pixel_ids
        }
        return ret

    def __len__(self):
        return len(self.metas)
