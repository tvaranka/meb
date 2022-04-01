from typing import List, Tuple
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import f1_score


class MEData(Dataset):
    def __init__(self, frames: np.ndarray, labels: np.ndarray, transform=None):
        self.frames = frames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        sample = self.frames[idx]
        if self.transform:
            sample = self.transform(sample)
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
    def __init__(self, task_num: int):
        super(MultiTaskLoss, self).__init__()
        self.task_num = task_num
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions: torch.tensor, labels: torch.tensor) -> float:
        losses = [self.criterion(predictions[i], labels[:, i]) for i in range(self.task_num)]
        return sum(losses)


class MultiTaskF1(nn.Module):
    def __init__(self, task_num: int):
        super(MultiTaskF1, self).__init__()
        self.task_num = task_num

    def forward(self, labels: torch.tensor, predictions: torch.tensor
    ) -> List[float]:
        f1s = [
            f1_score(labels[:, i], predictions[:, i], average="macro")
            for i in range(self.task_num)
        ]
        return f1s
