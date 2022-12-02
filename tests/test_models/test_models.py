import pytest
import torch

from meb import models, utils

# Tests modified from https://krokotsch.eu/posts/deep-learning-unit-tests/


@pytest.mark.parametrize(
    "model_name, model_input_shape",
    [
        (models.SSSNet, (3, 64, 64)),
        (models.STSTNet, (3, 28, 28)),
        (models.OffApexNet, (2, 28, 28)),
    ],
)
@torch.no_grad()
def test_correct_shape(model_name, model_input_shape):
    inp = torch.randn((2,) + model_input_shape)
    model = model_name(num_classes=5)
    output = model(inp)
    assert output.shape == (2, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(
    "model_name, model_input_shape",
    [
        (models.SSSNet, (3, 64, 64)),
        (models.STSTNet, (3, 28, 28)),
        (models.OffApexNet, (2, 28, 28)),
    ],
)
@torch.no_grad()
def test_device_moving(model_name, model_input_shape):
    inp = torch.randn((2,) + model_input_shape)
    model = model_name(num_classes=5).eval()

    utils.set_random_seeds()
    outputs_cpu = model(inp)
    utils.set_random_seeds()
    model_on_gpu = model.to("cuda:0")
    outputs_gpu = model_on_gpu(inp.to("cuda:0"))

    assert 1e-4 > torch.sum(outputs_cpu - outputs_gpu.cpu())
