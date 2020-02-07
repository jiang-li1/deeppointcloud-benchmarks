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

        self._i_loss = model.get_current_losses()['loss_seg']

        outputs = self._convert(model.get_output())
        targets = self._convert(model.get_labels())

        self.track_outputs(outputs, targets)        

    def track_outputs(self, outputs, targets):
        outputs = self._convert(outputs)
        targets = self._convert(targets)

        assert outputs.shape[0] == len(targets)
        self._confusion_matrix.count_predicted_batch(targets, np.argmax(outputs, 1))

        self._a_acc = 100 * self._confusion_matrix.get_overall_accuracy()
        self._a_macc = 100 * self._confusion_matrix.get_mean_class_accuracy()
        self._a_miou = 100 * self._confusion_matrix.get_average_intersection_union()

        instantanousCF = ConfusionMatrix(self._num_classes)
        instantanousCF.count_predicted_batch(targets, np.argmax(outputs, 1))
        self._i_acc = 100 * instantanousCF.get_overall_accuracy()
        self._i_iou = 100 * instantanousCF.get_overall_iou()



        # self._conf_matrix = self._confusion_matrix.get_confusion_matrix()

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

    def get_instantaneous_metrics(self) -> Dict[str, float]:

        return {name: getattr(self, '_' + name) for name in ['i_acc', 'i_loss', 'i_iou']}


    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_a_acc".format(self._stage)] = self._a_acc
        metrics["{}_a_macc".format(self._stage)] = self._a_macc
        metrics["{}_a_miou".format(self._stage)] = self._a_miou
        # metrics["{}_loss".format(self._stage)] = self._loss
        if verbose:
            # metrics['{}_conf_matrix'.format(self._stage)] = self._get_str_conf_matrix()

            confMat = self._confusion_matrix.get_confusion_matrix()
            confMat = np.concatenate((
                confMat,
                np.expand_dims(self._confusion_matrix.get_fn_per_class(), axis=0),
                np.expand_dims(self._confusion_matrix.get_fn_rate_per_class(), axis=0)
            ))

            fpPerClass = self._confusion_matrix.get_fp_per_class()
            fpPerClass = np.append(fpPerClass, [None]*2)
            fpRate = self._confusion_matrix.get_fp_rate_per_class()
            fpRate = np.append(fpRate, [None]*2)
            confMat = np.concatenate((
                confMat,
                np.expand_dims(fpPerClass, axis=1),
                np.expand_dims(fpRate, axis=1),
            ), axis=1)

            metrics['{}_conf_matrix'.format(self._stage)] = pd.DataFrame(
                confMat, 
                ['pred ' + l for l in self.labels] + ["FN (GT but not pred)", "FN Rate"], 
                [*self.labels, 'FP (pred but not GT)', 'FP Rate'])

            count = np.expand_dims(self._confusion_matrix.count_gt_per_class(), axis=0)
            prop = np.expand_dims(self._confusion_matrix.gt_proportion_per_class(), axis=0)
            metrics['{}_distribution'.format(self._stage)] = pd.DataFrame(
                np.concatenate((count, prop)),
                ['Count', 'Proportion'],
                self.labels
            )

            classIOU = self._confusion_matrix.get_intersection_union_per_class()[0]

            metrics["{}_iou_per_class".format(self._stage)] = pd.DataFrame(
                np.expand_dims(classIOU, axis=0),
                ['iou'],
                self.labels
            )


        return metrics