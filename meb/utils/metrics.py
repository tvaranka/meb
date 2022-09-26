from typing import List

import torch
from torch import nn
from sklearn.metrics import f1_score


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions: List[torch.tensor], labels: torch.tensor) -> float:
        task_num = len(predictions)
        losses = [self.criterion(predictions[i], labels[:, i]) for i in range(task_num)]
        return sum(losses)


class MultiTaskF1(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(labels: torch.tensor, predictions: torch.tensor
                ) -> List[float]:
        task_num = predictions.shape[-1]
        f1s = [
            f1_score(labels[:, i], predictions[:, i], average="macro")
            for i in range(task_num)
        ]
        return f1s


class MultiLabelBCELoss(nn.Module):
    def __init__(self, weight: torch.tensor = None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(weight=weight)

    def forward(self, x, y):
        loss = self.loss(x, y.float())
        return loss


class MultiLabelF1Score(nn.Module):
    def __init__(self, average: str = None):
        super().__init__()
        self.average = average

    def forward(self, labels: torch.tensor, outputs: torch.tensor
    ) -> List[float]:
        predictions = torch.where(outputs > 0, 1, 0)
        if self.average is None:
            return f1_score(labels, predictions, average=None)
        # Each label separately to get f1 for each label
        if self.average is not None:
            return [f1_score(labels[:, i], predictions[:, i], average=self.average)
                    for i in range(labels.shape[-1])]


class MultiClassF1Score(nn.Module):
    def __init__(self, average: str = None):
        super().__init__()
        self.average = average

    def forward(self, labels: torch.tensor, outputs: torch.tensor
                ) -> List[float]:
        _, predictions = outputs.max(1)
        result = f1_score(labels, predictions, average=self.average)
        return result


class MultiMetric:
    def __init__(self, metric_objects):
        self.metrics = [metric() for metric in metric_objects]

    def __call__(self, y: torch.tensor, p: torch.tensor):
        return [metric(y, p) for metric in self.metrics]