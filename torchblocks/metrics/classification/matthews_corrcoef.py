import torch
import logging
from torchmetrics import MatthewsCorrCoef as _MatthewsCorrCoef
from ..base import Metric

logger = logging.getLogger(__name__)


class MattewsCorrcoef(_MatthewsCorrCoef, Metric):
    """
    Matthews Correlation Coefficient
    """

    def __init__(self, num_classes, threshold=0.5):
        super(MattewsCorrcoef, self).__init__(num_classes=num_classes, threshold=threshold)

    def reset(self):
        default = torch.zeros(self.num_classes, self.num_classes)
        self.add_state("confmat", default=default, dist_reduce_fx="sum")

    def value(self):
        score = self.compute()
        return score.item()

    def name(self):
        return 'mcc'
