import imp
import torch
import tqdm

from lib.dataloader.make_dataset import make_data_loader
from lib.network.make_network import make_network
from lib.train.make_trainer import make_trainer
from lib.train.optimizer import make_optimizer
from lib.train.scheduler import make_lr_scheduler, set_lr_scheduler
from lib.train.recorder import make_recorder
from lib.utils.net_utils import save_model, load_model, load_network, load_pretrain
from lib.dataloader.data_utils import to_cuda
from lib.visualizer.make_visualizer import make_visualizer
from lib.config.config import cfg

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    network = make_network(cfg)
    
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)

    begin_epoch = load_model(network,
                            optimizer,
                            scheduler,
                            recorder,
                            cfg.trained_model_dir,
                            resume=cfg.resume) 

    set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg,
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


def visualize(network):
    network.eval()
    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            output = network(batch)
            visualizer.visualize(output, batch)


if __name__ == '__main__':
    train()

