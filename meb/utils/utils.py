from typing import List
from functools import partial
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset



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
                # Assumes that the transform is for video, e.g., pytorchvideo
                sample = self.spatial_transform(sample)
                sample = sample.permute(0, 2, 3, 1)  # TCHW -> THWC
            sample = sample.permute(3, 0, 1, 2)  # THWC -> CTHW

        elif len(sample.shape) == 3:  # Optical flow
            if self.spatial_transform:
                sample = torch.tensor(sample)
                sample = self.spatial_transform(sample)
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


class Printer:
    def __init__(self, config: "Config", label_type: str, split_column: str):
        self.cf = config
        self.label_type = label_type
        self.split_column = split_column

    @staticmethod
    def _metric_name(metric) -> str:
        if isinstance(metric, partial):
            metric_name = metric.func.__name__
        else:
            metric_name = metric.__name__
        return metric_name

    def print_test_validation(self, metrics: List[float]) -> None:
        print("Final results\n")
        for i, metric in enumerate(metrics):
            if len(metrics) > 1:
                print(self._metric_name(self.cf.evaluation_fn[i]))
            print("All AUs: ", list(zip(self.cf.action_units, np.around(np.array(metric) * 100, 2))))
            print("Mean: ", np.around(np.mean(metric) * 100, 2))
            print("\n")

    def print_train_test_evaluation(
            self,
            train_metrics: List[float],
            test_metrics: List[float],
            split_name: str,
            n: int
    ) -> None:
        if self.label_type == "au":
            for i in range(len(train_metrics)):
                if len(train_metrics) > 1:
                    print(self._metric_name(self.cf.evaluation_fn[i]))
                print(
                    f"{self.split_column.capitalize()}: {split_name}, n={n} | "
                    f"train_mean: {np.mean(train_metrics[i]):.4} | "
                    f"test_mean: {np.mean(test_metrics[i]):.4}"
                )
                print(f"Test per AU: {list(zip(self.cf.action_units, np.around(np.array(test_metrics[i]) * 100, 2)))}\n")
        else:
            for i in range(len(train_metrics)):
                if len(train_metrics) > 1:
                    print(self._metric_name(self.cf.evaluation_fn[i]))
                print(
                    f"{self.split_column.capitalize()}: {split_name}, n={n} | "
                    f"train_mean: {train_metrics[i]:.4} | "
                    f"test_mean: {test_metrics[i]:.4}"
                )

    def print_train_test_validation(
            self,
            train_metrics: List[float],
            test_metrics: List[float],
            epoch: int
    ) -> None:
        print(f"Validating at epoch {epoch + 1}\n")
        print("-" * 80)
        for i in range(len(train_metrics)):
            if len(train_metrics) > 1:
                print(self._metric_name(self.cf.evaluation_fn[i]))
            print(
                f"Training metric mean: {np.mean(train_metrics[i]):>6f}\n"
                f"Test metric per AU: {list(zip(self.cf.action_units, np.around(np.array(test_metrics[i]) * 100, 2)))}\n"
                f"Testing metric mean: {np.mean(test_metrics[i]):>6f}"
            )
        print("-" * 80)


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