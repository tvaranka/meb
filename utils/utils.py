from typing import List, Tuple, Union
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import f1_score


class MEData(Dataset):
    def __init__(self, frames: Union[np.ndarray, "CustomDataset"], labels: np.ndarray,
                 transform_temporal=None, transform_spatial=None):
        self.frames = frames
        self.labels = labels
        self.transform_temporal = transform_temporal
        self.transform_spatial = transform_spatial

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.transform_temporal:
            sample = self.frames.get_video_sampled(idx, self.transform_temporal)
            if len(sample.shape) == 3: # Grayscale, pad dims
                sample = np.expand_dims(sample, 0)
        else:
            sample = self.frames[idx]
        if self.transform_spatial:
            sample = torch.tensor(sample * 255.0).type(torch.uint8)
            sample = self.transform_spatial(sample)
            sample = (sample / 255.0).type(torch.float32)
        label = self.labels[idx]
        return sample, label


def reset_weights(m: nn.Module) -> None:
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def set_random_seeds(seed: int = 1) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
            f1_score(labels[:, i], predictions[:, i], average="binary")
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
        if self.average == "macro":
            return [f1_score(labels[:, i], predictions[:, i], average="macro")
                    for i in range(labels.shape[-1])]


class MultiClassF1Score(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(labels: torch.tensor, outputs: torch.tensor
                ) -> List[float]:
        _, predictions = outputs.max(1)
        result = f1_score(labels, predictions, average="macro")
        return result