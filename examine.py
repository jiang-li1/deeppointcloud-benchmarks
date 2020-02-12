import open3d
import torch
import numpy as np
import hydra
import time
import traceback
import logging
from omegaconf import OmegaConf

# Import building function for model and dataset
from src import instantiate_model, instantiate_dataset
# Import BaseModel / BaseDataset for type checking
from src.models.base_model import BaseModel
from src.data.base_dataset import BaseDataset

# Import from metrics
from src.metrics.base_tracker import BaseTracker
from src.metrics.colored_tqdm import Coloredtqdm as Ctq
from src.metrics.model_checkpoint import get_model_checkpoint, ModelCheckpoint

# Utils import
from src.utils.model_building_utils.model_definition_resolver import resolve_model
from src.utils.colors import COLORS
from src.utils.config import set_format
from src.utils.model_examination import get_modules_of_type

from src.data.pointcloud import ClassifiedPointCloud, PointCloud

from utils.visualization.eval_vis import visualize_predictions, visualize_difference, visualize_classes, visualize_cloud, visualize_subcloud, visualize_subcloud_knn
from utils.visualization.pcd_utils import clear_vis

from src.modules.RandLANet.modules import RandlaBlock

global model, dataset, tracker, train_loader, data, i, device, epoch, checkpoint, log, train_iterator, iter_data_time, tq_train_loader

log = logging.getLogger(__name__)

def train_it(len=1):
    global model, dataset, tracker, train_loader, data, i, device, epoch, checkpoint, log, train_iterator, iter_data_time, tq_train_loader

    for _ in range(len):
        i, data = next(train_iterator)

        data = data.to(device)  # This takes time

        # print(data.pos[0][0])
        # import pdb; pdb.set_trace()
        print(data.name)

        model.set_input(data)
        t_data = time.time() - iter_data_time

        iter_start_time = time.time()

        try:
            model.optimize_parameters(dataset.batch_size)
        except Exception as e:
            traceback.print_exc()
            import pdb; pdb.set_trace()

        if i % 10 == 0 or True:
            tracker.track(model)

        # import pdb; pdb.set_trace()

        print(tracker.get_instantaneous_metrics())

        tq_train_loader.set_postfix(
            **(tracker.get_metrics() if i % 10 == 0 else tracker.get_instantaneous_metrics()),
            data_loading=float(t_data),
            iteration=float(time.time() - iter_start_time),
            color=COLORS.TRAIN_COLOR
        )
        iter_data_time = time.time()

def vis_output(difference=True, classes=False, predictions=False):
    output = model.get_output()
    output = torch.argmax(output.cpu(), 1)

    cpcd = ClassifiedPointCloud.from_data(data.to('cpu'))

    if difference:
        visualize_difference(cpcd, output, data.inner_idx)
    if classes:
        visualize_classes(cpcd)
    if predictions:
        visualize_predictions(cpcd, output, data.inner_idx)

def vis_layers(conv=RandlaBlock, vis_sampling=True, vis_knn=False):

    convs = get_modules_of_type(model, conv)

    # range indexes into the previous layer
    samp_indexes = [conv.layer_info.samp_idx for conv in convs]

    # mask indexes into data.pos
    layer_indexes = []

    # false_mask = torch.zeros((len(data.pos),)).to(torch.bool)
    # invert_idx = torch.zeros((len(data.pos,))).to(torch.bool)
    cumulative_idx = torch.arange(len(data.pos))
    # arange = torch.arange((len(data.pos)))
    # cumulative_idx = torch.ones((len(data.pos),)).to(torch.bool)
    for si in samp_indexes:
        if si is None:
            # mask = torch.zeros((len(data.pos),)).to(torch.bool)
            # mask[cumulative_idx] = True
            layer_indexes.append(cumulative_idx.clone())
        else:
            cumulative_idx = cumulative_idx[si]
            # mask = torch.zeros((len(data.pos),)).to(torch.bool)
            # mask[cumulative_idx] = True
            layer_indexes.append(cumulative_idx.clone())

    if vis_sampling:
        for i, conv in enumerate(convs):
            visualize_subcloud(PointCloud(data.pos), layer_indexes[i], window_name='Sampling Layer {}'.format(i))

    if vis_knn:
        for i, conv in enumerate(convs):
            visualize_subcloud_knn(PointCloud(data.pos), layer_indexes[i], conv.layer_info.edge_index, window_name='KNN Layer {}'.format(i))





def train_epoch_start():
    global model, dataset, tracker, train_loader, data, i, device, epoch, checkpoint, log, train_iterator, iter_data_time, tq_train_loader

    model.train()
    tracker.reset("train")
    train_loader = dataset.train_dataloader()
    # train_loader.dataset.load()
    iter_data_time = time.time()
    with Ctq(train_loader) as tq_train_loader:
        train_iterator = enumerate(tq_train_loader)
        # for i, data in enumerate(tq_train_loader):
        train_it()
        return

def train_epoch_finish():
    global model, dataset, tracker, train_loader, data, i, device, epoch, checkpoint, log, train_iterator, iter_data_time, tq_train_loader

    metrics = tracker.publish()
    tracker.print_summary()
    checkpoint.save_best_models_under_current_metrics(model, metrics)
    log.info("Learning rate = %f" % model.learning_rate)


def eval_epoch(model: BaseModel, dataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint, log):
    model.eval()
    tracker.reset("val")
    loader = dataset.val_dataloader()
    with Ctq(loader) as tq_val_loader:
        for data in tq_val_loader:
            data = data.to(device)
            with torch.no_grad():
                model.set_input(data)
                model.forward()

            tracker.track(model)
            tq_val_loader.set_postfix(**tracker.get_metrics(), color=COLORS.VAL_COLOR)

    metrics = tracker.publish()
    tracker.print_summary()
    checkpoint.save_best_models_under_current_metrics(model, metrics)


def test_epoch(model: BaseModel, dataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint, log):
    model.eval()
    tracker.reset("test")
    loader = dataset.test_dataloader()
    # loader.dataset.load()
    with Ctq(loader) as tq_test_loader:
        for data in tq_test_loader:

            print(data.name)
            data = data.to(device)
            with torch.no_grad():
                model.set_input(data)
                model.forward()

            tracker.track(model)
            tq_test_loader.set_postfix(
                **tracker.get_instantaneous_metrics(), 
                color=COLORS.TEST_COLOR
            )

    metrics = tracker.publish()
    tracker.print_summary()
    checkpoint.save_best_models_under_current_metrics(model, metrics)


def run(cfg, model, dataset: BaseDataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint):
    for epoch in range(checkpoint.start_epoch, cfg.training.epochs):
        log.info("EPOCH %i / %i", epoch, cfg.training.epochs)
        train_epoch_start()

        return 

def run_test():

    if dataset.has_val_loader:
        eval_epoch(model, dataset, device, tracker, checkpoint, log)

    test_epoch(model, dataset, device, tracker, checkpoint, log)

    # Single test evaluation in resume case
    if checkpoint.start_epoch >= cfg.training.epochs:
        test_epoch(model, dataset, device, tracker, checkpoint, log)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    global model, dataset, tracker, train_loader, data, i, device, epoch, checkpoint, log

    if cfg.pretty_print:
        print(cfg.pretty())

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.training.cuda) else "cpu")
    log.info("DEVICE : {}".format(device))

    # Get task and model_name
    tested_task = cfg.data.get('task', cfg.task)
    tested_model_name = cfg.model_name

    # Find and create associated model
    model_config = getattr(cfg.models, tested_model_name, None)

    # Find which dataloader to use
    cfg_training = set_format(model_config, cfg.training)

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg_training.enable_cudnn

    # Find and create associated dataset
    dataset_config = cfg.data
    tested_dataset_class = getattr(dataset_config, "class")
    dataset_config.dataroot = hydra.utils.to_absolute_path(dataset_config.dataroot)
    dataset = instantiate_dataset(tested_dataset_class, tested_task)(dataset_config, cfg_training)

    # Find and create associated model
    resolve_model(model_config, dataset, tested_task)
    model_class = getattr(model_config, "class")
    model_config = OmegaConf.merge(model_config, cfg_training)
    model = instantiate_model(model_class, tested_task, model_config, dataset)

    log.info(model)

    # Optimizer
    otimizer_class = getattr(cfg_training.optimizer, "class")
    model.set_optimizer(
        getattr(torch.optim, otimizer_class, None), cfg_training.optimizer.params, cfg_training.learning_rate
    )

    # Set sampling / search strategies
    if cfg_training.precompute_multi_scale:
        dataset.set_strategies(model)

    model = model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    log.info("Model size = %i", params)

    # metric tracker
    if cfg.wandb.log:
        import wandb

        wandb.init(project=cfg.wandb.project)
        # wandb.watch(model)

    tracker = dataset.get_tracker(model, tested_task, dataset, cfg.wandb, cfg.tensorboard)

    checkpoint = get_model_checkpoint(
        model,
        cfg_training.checkpoint_dir,
        tested_model_name,
        cfg_training.resume,
        cfg_training.weight_name,
        "val" if dataset.has_val_loader else "test",
        cfg_training.optimizer.params,
    )

    # Run training / evaluation
    run(cfg, model, dataset, device, tracker, checkpoint)


if __name__ == "__main__":
    main()
