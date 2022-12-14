from contextlib import nullcontext
from functools import partial

import pytest
import torch
import torch.cuda.amp
import torch.nn as nn

from meb import models, utils
from meb.core import Config, CrossDatasetValidator
from meb.utils.utils import ConfigException, NullScaler, validate_config

from ..test_data.base import BaseTestDataset


def _copy_class_object(obj: object):
    """Copies class objects and their attributes."""
    return type("Object", (object,), dict(obj.__dict__))


def test_correct_config():
    cf = _copy_class_object(Config)
    cf.model = partial(models.SSSNet, num_classes=9)
    try:
        validate_config(cf, Config)
    except ConfigException:
        pytest.fail("An exception was raised when it should not have been.")


def test_incorrect_config():
    cf = _copy_class_object(Config)
    cf.evaluation_fn = utils.MultiLabelF1Score()
    with pytest.raises(ConfigException):
        validate_config(cf, Config)

    cf = _copy_class_object(Config)
    cf.criterion = nn.CrossEntropyLoss()
    with pytest.raises(ConfigException):
        validate_config(cf, Config)

    cf = _copy_class_object(Config)
    cf.model = models.SSSNet(num_classes=9)
    with pytest.raises(ConfigException):
        validate_config(cf, Config)

    cf = _copy_class_object(Config)
    cf.custom_feature = 3
    with pytest.warns(UserWarning):
        validate_config(cf, Config)


def test_validators():
    cf = _copy_class_object(Config)
    cf.model = partial(models.SSSNet, num_classes=9)
    v = CrossDatasetValidator(cf)
    assert v.split_column == "dataset"


def test_amp_autocast():
    cf = _copy_class_object(Config)
    cf.device = torch.device("cpu")
    cf.model = models.SSSNet
    v = CrossDatasetValidator(cf)
    assert v.amp_autocast == nullcontext
    v.setup_training()
    assert isinstance(v.loss_scaler, NullScaler)
    cf.loss_scaler = torch.cuda.amp.GradScaler
    v = CrossDatasetValidator(cf)
    assert v.amp_autocast == torch.cuda.amp.autocast


class TestValidator(BaseTestDataset):
    def test_validate_output(self):
        # Should be output list
        pass

    def test_validate_print(self):
        # Test final printing output
        pass

    def test_printer(self):
        # Mock
        pass
