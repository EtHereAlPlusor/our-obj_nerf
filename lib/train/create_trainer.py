from .trainer import Trainer
import importlib

def _loss_factory(cfg, network):
    module = cfg.loss_module
    path = cfg.loss_path
    loss = importlib.machinery.SourceFileLoader(module, path).load_module().Loss(network)
    return loss

def create_trainer(cfg, network):
    network = _loss_factory(cfg, network)
    return Trainer(network)