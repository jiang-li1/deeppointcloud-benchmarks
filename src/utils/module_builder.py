from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

def build_module(cfg, modules_lib):

    module_cls = getattr(modules_lib, cfg['module_name'])

    module_args = dict(cfg)

    for k in module_args:
        if type(module_args[k]) == DictConfig:
            module_args[k] = build_module(module_args[k], modules_lib)

    module_args.pop('module_name')
    return module_cls(**module_args)