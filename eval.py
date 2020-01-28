from os import path as osp
ROOT = osp.join(osp.dirname(osp.realpath(__file__)))
import open3d
import pdal
import torch
from torch import nn
from torch import autograd
import numpy as np
import torch.nn.functional as F
import hydra
from tqdm import tqdm as tq
import time
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import logging

# Import building function for model and dataset
from src import find_model_using_name, find_dataset_using_name

# Import BaseModel / BaseDataset for type checking
from src.models.base_model import BaseModel
from src.datasets.base_dataset import BaseDataset

# Import from metrics
from src.metrics.base_tracker import BaseTracker
from src.metrics.colored_tqdm import Coloredtqdm as Ctq
from src.metrics.model_checkpoint import get_model_checkpoint, ModelCheckpoint

# Utils import
from src.utils.model_building_utils.model_definition_resolver import resolve_model
from src.utils.colors import COLORS
from src.utils.config import merges_in_sub, set_format

from utils.visualization.eval_vis import visualize_classes, visualize_predictions, visualize_difference
from src.datasets.base_patch_dataset import ClassifiedPointCloud

def eval_model(model, loader: torch.utils.data.DataLoader, tracker, device, eval_name, use_cache=True):

    cache_fname = osp.join(ROOT, 'outputs', '.eval_cache', eval_name + '.pt') 

    if use_cache and osp.exists(cache_fname):
        print('Using cached eval: ', cache_fname)
        return torch.load(cache_fname)

    globalOutput = torch.full((loader.dataset.data.pos.shape[0], loader.dataset.num_classes), 0)
    globalTargets = torch.full((loader.dataset.data.pos.shape[0],), -1).to(torch.long)

    with Ctq(loader, total = len(loader.dataset)) as tq_test_loader:
        for data in tq_test_loader:
            data = data.to(device)
            with torch.no_grad():
                model.set_input(data)
                model.forward()

            globalOutput[data.global_index] = model.output.cpu()
            globalTargets[data.global_index] = model.labels.cpu()

            # predClass = torch.argmax(model.output.cpu(), 1)
            # globalClassif[data.global_index] = predClass

            # tracker.track(model)
            # tq_test_loader.set_postfix(**tracker.get_metrics(), color=COLORS.TEST_COLOR)

            # break

    torch.save((globalOutput, globalTargets), cache_fname)

    return globalOutput, globalTargets


def test(model: BaseModel, dataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint, log, eval_name = None):
    model.eval()
    tracker.reset("test")
    loader = dataset.test_dataloader()

    global globalOutput, globalTargets

    globalOutput, globalTargets = eval_model(model, loader, tracker, device, eval_name)

    globalClassif = torch.argmax(globalOutput.cpu(), 1)

    tracker.track_outputs(globalOutput, globalTargets)

    metrics = tracker.publish()
    tracker.print_summary()

    # visualize_classes(ClassifiedPointCloud.from_data(loader.dataset.data))
    # visualize_predictions(ClassifiedPointCloud.from_data(loader.dataset.data), globalClassif)
    visualize_difference(ClassifiedPointCloud.from_data(loader.dataset.data), globalClassif)



@hydra.main(config_path='conf/config.yaml')
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.training.cuda) else "cpu")
    print("DEVICE : {}".format(device))

    # Get task and model_name
    exp = cfg.experiment
    tested_task = exp.task
    tested_model_name = exp.model_name
    tested_dataset_name = exp.dataset

    # Find and create associated model
    model_config = getattr(cfg.models, tested_model_name, None)

    # Find which dataloader to use
    cfg_training = set_format(model_config, cfg.training)

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg_training.enable_cudnn

    # Find and create associated dataset
    dataset_config = getattr(cfg.data, tested_dataset_name, None)
    dataset_config.dataroot = hydra.utils.to_absolute_path(dataset_config.dataroot)
    dataset = find_dataset_using_name(tested_dataset_name, tested_task)(dataset_config, cfg_training, eval_mode=True)

    # Find and create associated model
    resolve_model(model_config, dataset, tested_task)
    model_config = merges_in_sub(model_config, [cfg_training, dataset_config])
    model = find_model_using_name(model_config.architecture, tested_task, model_config, dataset)

    # Optimizer
    lr_params = cfg_training.learning_rate
    model.set_optimizer(getattr(torch.optim, cfg_training.optimizer, None), lr_params=lr_params)

    # Set sampling / search strategies
    dataset.set_strategies(model, precompute_multi_scale=cfg_training.precompute_multi_scale)

    model = model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    log.info("Model size = %i", params)

    # metric tracker
    if cfg.wandb.log:
        import wandb

        wandb.init(project=cfg.wandb.project)
        # wandb.watch(model)

    tracker: BaseTracker = dataset.get_tracker(model, tested_task, dataset, cfg.wandb, cfg.tensorboard)

    checkpoint = get_model_checkpoint(
        model, exp.checkpoint_dir, tested_model_name, exp.resume, cfg_training.weight_name
    )

    # Run training / evaluation
    test(model, dataset, device, tracker, checkpoint, log, eval_name=cfg.experiment.name)

if __name__ == "__main__":
    main()