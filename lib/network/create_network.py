import importlib


def create_network(cfg):
    module = cfg.network_module
    path = cfg.network_path
    network = importlib.machinery.SourceFileLoader(module, path).load_module().Network()
    return network