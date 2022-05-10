from .yacs import CfgNode as CN
import argparse
import os
import numpy as np

cfg = CN()

# task settings
cfg.N_rays = 2048
cfg.train_frames = 100
cfg.val_list = []
cfg.start = 6060
cfg.ratio = 0.5
cfg.N_samples = 64
cfg.cascade_samples = 128
cfg.samples_all = 192
cfg.use_stereo = False
cfg.dist = 300

# module
cfg.train_dataset_module = ''
cfg.test_dataset_module = ''
cfg.val_dataset_module = ''
cfg.network_module = ''
cfg.loss_module = ''

# experiment name
cfg.exp_name = ''
cfg.pretrain = ''

# network
cfg.white_bkgd = False
cfg.distributed = False

# if load the pretrained network
cfg.resume = True

# task
cfg.task = ''

# epoch
cfg.ep_iter = -1
cfg.save_ep = -1
cfg.save_latest_ep = -1
cfg.log_interval = -1

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------

cfg.train = CN()
cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 5
cfg.train.num_workers = 8
cfg.train.collator = 'default'
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.train.weight_color = 1.

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 0.
cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})
cfg.train.batch_size = 1
cfg.train.acti_func = 'relu'
cfg.train.shuffle = True

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.val_dataset = ''
cfg.test.batch_size = 1
cfg.test.collator = 'default'
cfg.test.epoch = -1
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})

# trained model
cfg.trained_model_dir = 'trained_model/'
cfg.trained_config_dir = 'trained_config/'

# recorder
cfg.record_dir = 'record/'

# result
cfg.result_dir = 'result/'

# base
cfg.base_dir = 'logs/'


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('Task must be specified')
    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(cfg.gpu)
    cfg.exp_name = cfg.exp_name.replace('gittag', os.popen('git describe --tags --always').readline().strip())
    cfg.trained_model_dir = os.path.join(cfg.base_dir, cfg.task, cfg.exp_name, cfg.trained_model_dir)    
    cfg.trained_config_dir = os.path.join(cfg.base_dir, cfg.task, cfg.exp_name, cfg.trained_config_dir)
    cfg.record_dir = os.path.join(cfg.base_dir, cfg.task, cfg.exp_name, cfg.record_dir)
    cfg.result_dir = os.path.join(cfg.base_dir, cfg.task, cfg.exp_name, cfg.result_dir)
    cfg.local_rank = args.local_rank
    modules = [key for key in cfg if '_module' in key]
    for module in modules:
        cfg[module.replace('_module', '_path')] = cfg[module].replace('.', '/') + '.py'

def create_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    return cfg

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--launcher', type=str, default='none', choices=['none', 'pytorch'])
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = create_cfg(args)
