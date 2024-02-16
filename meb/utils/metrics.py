from typing import Iterable, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn


class MultiLabelBCELoss(nn.Module):
    """Multi-label loss wrapper

    Uses binary cross-entropy with logits to compute loss for
    multi-label cases.

    Parameters
    ----------
    weight : torch.tensor, optional
        Weights classes using the provided weight. See torch documentation.

    """

    def __init__(self, weight: Optional[torch.Tensor] = None, **kwargs):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(weight=weight, **kwargs)

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        loss = self.loss(outputs, labels.float())
        return loss
    
class MultiClassCELoss(nn.Module):
    """Multi-class cross entropy loss wrapper

    Uses cross-entropy with logits to compute loss for
    multi-label cases.

    Parameters
    ----------
    weight : torch.tensor, optional
        Weights classes using the provided weight. See torch documentation.

    """

    def __init__(self, weight: Optional[torch.Tensor] = None, **kwargs):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, **kwargs)

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        loss = self.loss(outputs, labels)
        return loss



class MultiLabelF1Score(nn.Module):
    """Multi-label F1-Score

    Computes F1-Score for multi-label cases. First, thresholds the output from a
    network to obtain binary predictions.

    Parameters
    ----------
    average : str, default None
        Determines how the different multi-labels are averaged together. See
        Scikit-learn documentation for more.
    threshold : float, default=0.0
        Determines at which value outputs from a network are thresholded for
        binary outputs.

    """

    def __init__(self, average: Optional[str] = None, threshold: float = 0.0):
        super().__init__()
        self.average = average
        self.threshold = threshold

    def forward(self, labels: torch.Tensor, outputs: torch.Tensor) -> List[float]:
        predictions = torch.where(outputs > self.threshold, 1, 0)
        if self.average is None:
            return f1_score(labels, predictions, average=None)
        # Each label separately to get f1 for each label
        if self.average is not None:
            return [
                f1_score(labels[:, i], predictions[:, i], average=self.average)
                for i in range(labels.shape[-1])
            ]


class MultiClassF1Score(nn.Module):
    """Multi class F1-Score

    Computes the F1-Score for multi class outputs.

    Parameters
    ----------
    average : str, default="macro"
        F1-Score is computed for individual classes as binary and the results are
        then aggregated together based on the average strategy. See scikit-learn
        documentation for more.
    """

    def __init__(self, average: str = "macro"):
        super().__init__()
        self.average = average

    def forward(self, labels: torch.tensor, outputs: torch.tensor) -> List[float]:
        _, predictions = outputs.max(1)
        result = f1_score(labels, predictions, average=self.average)
        return result


def robust_roc_auc(y, p, average: str):
    """Set cases with no positive samples to 0.5"""
    results = []
    for i in range(y.shape[1]):
        if len(np.unique(y[:, i])) != 2:
            results.append(0.5)
        else:
            res = roc_auc_score(y[:, i], p[:, i], average=average)
            results.append(res)
    return results


class MultiLabelAUC(nn.Module):
    def __init__(self, average: str = None):
        super().__init__()
        self.average = average

    def __call__(self, y, p):
        return robust_roc_auc(y, p, average=self.average)


class MultiTaskLoss(nn.Module):
    """Multi-task loss

    Uses cross-entropy for different tasks and aggregates the results together"""

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions: List[torch.Tensor], labels: torch.Tensor) -> float:
        task_num = len(predictions)
        losses = [self.criterion(predictions[i], labels[:, i]) for i in range(task_num)]
        return sum(losses)


class MultiTaskF1(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(labels: torch.Tensor, predictions: torch.Tensor) -> List[float]:
        task_num = predictions.shape[-1]
        f1s = [
            f1_score(labels[:, i], predictions[:, i], average="macro")
            for i in range(task_num)
        ]
        return f1s


class MultiMetric:
    """Allows the use of multiple different metrics at the same time"""

    def __init__(self, metric_objects: Iterable[nn.Module]):
        self.metrics = [metric() for metric in metric_objects]

    def __call__(self, y: torch.Tensor, p: torch.Tensor) -> List[float]:
        return [metric(y, p) for metric in self.metrics]
