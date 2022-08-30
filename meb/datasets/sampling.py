from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class BaseTemporalSubSample(ABC):
    """
    Abstract class for temporal subsampling techniques.
    Abstracts num_samples and temporal_dim.
    """

    def __init__(self, num_samples: int, temporal_dim: int = -4):
        self.num_samples = num_samples
        self.temporal_dim = temporal_dim

    @abstractmethod
    def __call__(self, x: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        pass

    def _check_validity(self, x: np.ndarray) -> None:
        assert isinstance(x, np.ndarray) or isinstance(x, torch.Tensor), "Only accepts numpy or torch arrays"
        assert len(x.shape) <= abs(self.temporal_dim)
        t = x.shape[self.temporal_dim]
        assert self.num_samples > 0, "Number of sampled frames needs to be larger than 0"
        assert t > 0, "Number of frames needs to be larger than 0"
        assert t >= self.num_samples, "Number of frames needs to be larger or equal than to be sampled"
        self.t = t


class UniformTemporalSubsample(BaseTemporalSubSample):
    def __init__(self, num_samples: int, temporal_dim: int = -4):
        super().__init__(num_samples, temporal_dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._check_validity(x)
        indices = np.linspace(0, self.t - 1, self.num_samples)
        indices = np.clip(indices, 0, self.t - 1).astype("int")
        return np.take(x, indices, self.temporal_dim)


class NoisyUniformTemporalSubsample(BaseTemporalSubSample):
    def __init__(self, num_samples: int, noise_level: int = "auto", temporal_dim: int = -4):
        super().__init__(num_samples, temporal_dim)
        self.noise_level = noise_level

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._check_validity(x)
        if self.noise_level == "auto":
            max_f = np.random.randint(self.num_samples, self.t - 1)
        indices = np.linspace(0, max_f, self.num_samples)
        indices = np.clip(indices, 0, self.t - 1).astype("int")
        return np.take(x, indices, self.temporal_dim)