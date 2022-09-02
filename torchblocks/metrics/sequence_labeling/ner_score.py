import pandas as pd
from ..base import Metric
from .precision_recall_fscore import precision_recall_fscore_support


class SequenceLabelingScore(Metric):

    def __init__(self, labels, average="micro"):
        self.labels = labels
        self.average = average
        self.reset()

    def update(self, preds, target):
        self.preds.extend(preds)
        self.target.extend(target)

    def value(self):
        columns = ["label", "precision", "recall", "f1", "support"]
        values = []
        for label in [self.average] + sorted(self.labels):
            p, r, f, s = precision_recall_fscore_support(
                self.target, self.preds, average=self.average,
                labels=None if label == self.average else [label])
            values.append([label, p, r, f, s])
        df = pd.DataFrame(values, columns=columns)
        f1 = df[df['label'] == self.average]['f1'].item()
        return {
            "df": df, f"f1_{self.average}": f1,  # for monitor
        }

    def name(self):
        return "ner_score"

    def reset(self):
        self.preds = []
        self.target = []


class ExtractionScore(Metric):

    def __init__(self):
        self.reset()

    def update(self, preds, target):
        for p, t in zip(preds, target):
            self.rights += len(p & t)
            self.origins += len(t)
            self.founds += len(p) 

    def value(self):
        precision = self.rights / self.founds
        recall = self.rights / self.origins
        f1 = 2 * self.rights / (self.origins + self.founds)
        return {
            "precision": precision,
            "recall": recall,
            "f1_micro": f1
        }

    def name(self):
        return "extraction_score"

    def reset(self):
        self.rights = 0.0
        self.origins = 1e-10
        self.founds = 1e-10
