from typing import Dict
import torchnet as tnt
import torch
import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.metrics.confusion_matrix import ConfusionMatrix
from src.metrics.base_tracker import BaseTracker, meter_value


class AHNTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):

        super(AHNTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self._num_classes = dataset.num_classes
        self.labels = dataset.class_num_to_name.values()
        self.reset(stage)

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._confusion_matrix = ConfusionMatrix(self._num_classes)

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    def track(self, model: BaseModel):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        outputs = self._convert(model.get_output())
        targets = self._convert(model.get_labels())
        assert outputs.shape[0] == len(targets)
        self._confusion_matrix.count_predicted_batch(targets, np.argmax(outputs, 1))

        self._acc = 100 * self._confusion_matrix.get_overall_accuracy()
        self._macc = 100 * self._confusion_matrix.get_mean_class_accuracy()
        self._miou = 100 * self._confusion_matrix.get_average_intersection_union()
        self._conf_matrix = self._confusion_matrix.get_confusion_matrix()

    # def _get_str_conf_matrix(self):

    #     s = '\n'
    #     s += 'predicted\\actual\n'
    #     s += str(self._conf_matrix)
    #     return s

    #from https://gist.github.com/zachguo/10296432
    # def _get_str_conf_matrix(self, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    #     """pretty print for confusion matrixes"""
    #     cm = self._conf_matrix
    #     labels = self.labels
        
    #     s = '\n'
    #     columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    #     empty_cell = " " * columnwidth
        
    #     # Begin CHANGES
    #     fst_empty_cell = (columnwidth-3)//2 * " " + "pred\\actual" + (columnwidth-3)//2 * " "
        
    #     if len(fst_empty_cell) < len(empty_cell):
    #         fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    #     # Print header
    #     s += "    " + fst_empty_cell + " "
    #     # End CHANGES
        
    #     for label in labels:
    #         s += "%{0}s".format(columnwidth) % label + " "
            
    #     s += '\n'
    #     # Print rows
    #     for i, label1 in enumerate(labels):
    #         s += "    %{0}s".format(columnwidth) % label1 + " "
    #         for j in range(len(labels)):
    #             cell = "%{0}.1f".format(columnwidth) % cm[i, j]
    #             if hide_zeroes:
    #                 cell = cell if float(cm[i, j]) != 0 else empty_cell
    #             if hide_diagonal:
    #                 cell = cell if i != j else empty_cell
    #             if hide_threshold:
    #                 cell = cell if cm[i, j] > hide_threshold else empty_cell
    #             s += cell + " "
    #         s += '\n'
    #     return s

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_acc".format(self._stage)] = self._acc
        metrics["{}_macc".format(self._stage)] = self._macc
        metrics["{}_miou".format(self._stage)] = self._miou
        if verbose:
            # metrics['{}_conf_matrix'.format(self._stage)] = self._get_str_conf_matrix()
            metrics['{}_conf_matrix'.format(self._stage)] = pd.DataFrame(self._conf_matrix, ['pred ' + l for l in self.labels], self.labels)

        return metrics