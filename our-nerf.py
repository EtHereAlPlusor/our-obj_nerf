import torch
import tqdm

from lib.dataloader.create_dataset import create_data_loader
from lib.network.create_network import create_network
from lib.train.create_trainer import create_trainer
from lib.train.optimizer import create_optimizer
from lib.train.scheduler import create_lr_scheduler, set_lr_scheduler
from lib.train.recorder import create_recorder
from lib.utils.net_utils import save_model, load_model
from lib.dataloader.data_utils import to_cuda
from lib.visualizer.create_visualizer import create_visualizer
from lib.config.config import cfg


def visualize(network):
    network.eval()
    data_loader = create_data_loader(cfg, is_train=False)
    visualizer = create_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            output = network(batch)
            visualizer.visualize(output, batch)


def train():
    network = create_network(cfg)
    
    trainer = create_trainer(cfg, network)
    optimizer = create_optimizer(cfg, network)
    scheduler = create_lr_scheduler(cfg, optimizer)
    recorder = create_recorder(cfg)

    begin_epoch = load_model(network,
                            optimizer,
                            scheduler,
                            recorder,
                            cfg.trained_model_dir,
                            resume=cfg.resume) 

    set_lr_scheduler(cfg, scheduler)

    train_loader = create_data_loader(cfg,
                                    is_train=True,
                                    is_distributed=cfg.distributed,
                                    max_iter=cfg.ep_iter)

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)
        
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, cfg.trained_model_dir, epoch)
        
        if (epoch + 1) % cfg.save_latest_ep == 0:
            save_model(network, optimizer, scheduler, recorder, cfg.trained_model_dir, epoch, last=True)
    
    visualize(network)


if __name__ == '__main__':
    train()

