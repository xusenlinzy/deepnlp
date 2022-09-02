import torch
import logging
from ..base import Metric
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

logger = logging.getLogger(__name__)


def tensor_to_cpu(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("tensor type: expected one of (torch.Tensor)")
    return tensor.detach().cpu()


def tensor_to_numpy(tensor):
    _tensor = tensor_to_cpu(tensor)
    return _tensor.numpy()


class F1Score(Metric):
    """
    F1 Score
    """

    def __init__(self, thresh=0.5, normalizate=True, task_type='binary', average='binary', search_thresh=False):

        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']
        self.thresh = thresh
        self.task_type = task_type
        self.normalizate = normalizate
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self, y_prob):
        """
        对于f1评分的指标，一般我们需要对阈值进行调整，一般不会使用默认的0.5值，因此
        这里我们队Thresh进行优化
        :return:
        """
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold, best_score

    def update(self, preds, target):

        self.y_true = tensor_to_numpy(target)
        if self.normalizate and self.task_type == 'binary':
            y_prob = tensor_to_numpy(preds.sigmoid().data)
        elif self.normalizate and self.task_type == 'multiclass':
            y_prob = tensor_to_numpy(preds.softmax(-1).data)
        else:
            y_prob = tensor_to_numpy(preds)

        if self.task_type == 'binary':
            if self.thresh and not self.search_thresh:
                self.y_pred = (y_prob > self.thresh).astype(int)
            else:
                thresh, f1 = self.thresh_search(y_prob=y_prob)
                logger.info(f"Best Thresh: {thresh:.4f} - F1 Score: {f1:.4f}")

        if self.task_type == 'multiclass':
            self.y_pred = np.argmax(y_prob, 1)

    def value(self):
        return f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)

    def name(self):
        return 'f1'


class ClassReport(Metric):
    """
    classification report
    """

    def __init__(self, target_names=None):
        self.target_names = target_names

    def update(self, preds, target):
        _, y_pred = torch.max(preds, 1)
        self.y_pred = tensor_to_numpy(y_pred)
        self.y_true = tensor_to_numpy(target)

    def value(self):
        score = classification_report(y_true=self.y_true, y_pred=self.y_pred,
                                      target_names=self.target_names, digits=4)
        logger.info(f"\n\n classification report: {score}")

    def name(self):
        return "class_report"
