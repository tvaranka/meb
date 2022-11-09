from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class BaseTemporalSubSample(ABC):
    """
    Abstract class for temporal subsampling techniques.
    Abstracts num_samples and temporal_dim.

    Parameters
    ----------
    num_samples : int
        The number of samples to be sampled
    temporal_dim : int
        The index of the temporal dimension. Defaults to -4, i.e., the second
        dimension in a shape 5 tensor.
    """

    def __init__(self, num_samples: int, temporal_dim: int = -4):
        self.num_samples = num_samples
        self.temporal_dim = temporal_dim

    @abstractmethod
    def __call__(self, x: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        pass

    def _check_validity(self, x: np.ndarray) -> None:
        assert isinstance(x, np.ndarray) or isinstance(
            x, torch.Tensor
        ), "Only accepts numpy or torch arrays"
        assert len(x.shape) <= abs(self.temporal_dim)
        t = x.shape[self.temporal_dim]
        assert (
            self.num_samples > 0
        ), "Number of sampled frames needs to be larger than 0"
        assert t > 0, "Number of frames needs to be larger than 0"
        assert (
            t >= self.num_samples
        ), "Number of frames needs to be larger or equal than to be sampled"
        self.t = t


class UniformTemporalSubsample(BaseTemporalSubSample):
    """Uniform sampling

    Sample frames based on a uniform distribution.
    """

    __doc__ += BaseTemporalSubSample.__doc__

    def __init__(self, num_samples: int, temporal_dim: int = -4):
        super().__init__(num_samples, temporal_dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._check_validity(x)
        indices = np.linspace(0, self.t - 1, self.num_samples)
        indices = np.clip(indices, 0, self.t - 1).astype("int")
        return np.take(x, indices, self.temporal_dim)


class NoisyUniformTemporalSubsample(BaseTemporalSubSample):
    """Noisy sampling

    Adds noise to the uniform sampling such that it is started later
    or stopped later

    Parameters
    ----------
    noise_type : str, optional, default "auto"
        Defines the type of added noise.
    prob : float, optional, default 0.5
        The probability of the augmentation being applied
    """

    __doc__ += BaseTemporalSubSample.__doc__

    def __init__(
        self,
        num_samples: int,
        temporal_dim: int = -4,
        noise_type: int = "auto",
        prob: float = 0.5,
    ):
        super().__init__(num_samples, temporal_dim)
        self.noise_type = noise_type
        self.prob = prob

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._check_validity(x)
        if self.noise_type == "auto":
            random_offset = np.random.randint(0, self.t - self.num_samples + 1)
        rand_num = np.random.random()
        if rand_num < self.prob:
            # Uniform
            indices = np.linspace(0, self.t - 1, self.num_samples)
        if rand_num > 1 - (self.prob / 2):
            indices = np.linspace(0 + random_offset, self.t, self.num_samples)
        else:
            indices = np.linspace(0, self.t - random_offset, self.num_samples)
        indices = np.clip(indices, 0, self.t - 1).astype("int")
        return np.take(x, indices, self.temporal_dim)
