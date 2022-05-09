from .trainer import Trainer
import imp

def _loss_factory(cfg, network):
    module = cfg.loss_module
    path = cfg.loss_path
    loss = imp.load_source(module, path).Loss(network)
    return loss

def create_trainer(cfg, network):
    network = _loss_factory(cfg, network)
    return Trainer(network)