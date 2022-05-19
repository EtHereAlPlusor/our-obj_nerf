import importlib


def create_visualizer(cfg):
    module = cfg.visualizer_module
    path = cfg.visualizer_path
    visualizer = importlib.machinery.SourceFileLoader(module, path).load_module().Visualizer()
    return visualizer
