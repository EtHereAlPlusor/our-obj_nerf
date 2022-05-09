import imp
from lib.config.config import cfg


def create_network(cfg):
    module = cfg.network_module
    path = cfg.network_path
    network = imp.load_source(module, path).Network()
    return network