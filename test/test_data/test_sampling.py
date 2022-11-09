import pytest
import torch
import numpy as np

from meb.datasets import UniformTemporalSubsample, NoisyUniformTemporalSubsample

np.randn = np.random.randn


@pytest.mark.parametrize(
    "sampling_class_type", [UniformTemporalSubsample, NoisyUniformTemporalSubsample]
)
@pytest.mark.parametrize("frame_number", [8, 12, 50, 100])
@pytest.mark.parametrize("array_type", [torch.ones, np.ones])
def test_sampling(sampling_class_type, frame_number, array_type):
    num_samples = 8
    x = array_type((frame_number, 3, 16, 16))
    sampler = sampling_class_type(num_samples)
    output = sampler(x)
    assert output.shape[0] == num_samples


@pytest.mark.parametrize(
    "sampling_class_type", [UniformTemporalSubsample, NoisyUniformTemporalSubsample]
)
@pytest.mark.parametrize("frame_number", [0, 2, 7])
@pytest.mark.parametrize("array_type", [torch.ones, np.ones])
def test_incorrect_sampling(sampling_class_type, frame_number, array_type):
    num_samples = 8
    x = array_type((frame_number, 3, 16, 16))
    sampler = sampling_class_type(num_samples)
    with pytest.raises(AssertionError):
        sampler(x)


@pytest.mark.parametrize(
    "sampling_class_type", [UniformTemporalSubsample, NoisyUniformTemporalSubsample]
)
@pytest.mark.parametrize("array_type", [torch.ones, np.ones])
def test_incorrect_shape(sampling_class_type, array_type):
    num_samples = 8
    sampler = sampling_class_type(num_samples)
    x = array_type((8, 16, 16))
    with pytest.raises(IndexError):
        sampler(x)

    x = array_type((3, 16, 16, 8))
    with pytest.raises(AssertionError):
        sampler(x)


@pytest.mark.parametrize("backend", [torch, np])
def test_noisy_uniform(backend):
    num_samples = 8
    np.random.seed(0)
    u_sampler = UniformTemporalSubsample(num_samples)
    nu_sampler = NoisyUniformTemporalSubsample(num_samples)
    x = backend.ones((100, 3, 16, 16))
    for i, a in enumerate(x):
        a *= i
    assert backend.all(u_sampler(x)[1:-1] != nu_sampler(x)[1:-1])
