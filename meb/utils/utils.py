from typing import List
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import f1_score


class MEData(Dataset):
    def __init__(self, frames: np.ndarray, labels: np.ndarray,
                 temporal_transform=None, spatial_transform=None):
        self.frames = frames
        self.labels = labels
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        sample = self.frames[idx]
        if len(sample.shape) == 4:  # Video
            sample = torch.tensor(sample)
            if self.temporal_transform:
                sample = self.temporal_transform(sample)
            if self.spatial_transform:
                sample = sample.permute(0, 3, 1, 2)  # THWC -> TCHW
                for i in range(len(sample)):
                    sample[i] = self.spatial_transform(sample[i])
                sample = sample.permute(0, 2, 3, 1)  # TCHW -> THWC
            sample = sample.permute(3, 0, 1, 2)  # THWC -> CTHW

        elif len(sample.shape) == 3:  # Optical flow
            if self.spatial_transform:
                sample = torch.tensor(sample * 255.0).type(torch.uint8)
                sample = self.spatial_transform(sample)
                sample = (sample / 255.0).type(torch.float32)
        else:
            raise NotImplementedError("Only works for len(video.shape) == 3 or 4")

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


# A constant dict for handy access to the commonly used action units
# per dataset
dataset_aus = {
    "casme2": ["AU1", "AU2", "AU4", "AU7", "AU12", "AU14", "AU15", "AU17"],
    "casme": ["AU1", "AU4", "AU9", "AU14"],
    "samm": ["AU2", "AU4", "AU7", "AU12"],
    "mmew": ["AU1", "AU2", "AU4", "AU5", "AU7", "AU10", "AU12", "AU14"],
    "fourd": ["AU1", "AU2", "AU4", "AU6", "AU7", "AU12", "AU17", "AU45"],
    "cross": ["AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10", "AU12", "AU14", "AU15", "AU17"],
}