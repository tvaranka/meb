from functools import partial

import pytest
import torch.nn as nn

from meb import models, utils
from meb.core import Config, CrossDatasetValidator
from meb.utils.utils import ConfigException, validate_config

from ..test_data.base import BaseTestDataset


def _copy_class_object(obj: object):
    """Copies class objects and their attributes."""
    return type("Object", (object,), dict(obj.__dict__))


def test_correct_config():
    cf = _copy_class_object(Config)
    cf.model = partial(models.SSSNet, num_classes=9)
    try:
        validate_config(cf)
    except ConfigException:
        pytest.fail("An exception was raised when it should not have been.")


def test_incorrect_config():
    cf = _copy_class_object(Config)
    cf.evaluation_fn = utils.MultiLabelF1Score()
    with pytest.raises(ConfigException):
        validate_config(cf)

    cf = _copy_class_object(Config)
    cf.criterion = nn.CrossEntropyLoss()
    with pytest.raises(ConfigException):
        validate_config(cf)

    cf = _copy_class_object(Config)
    cf.model = models.SSSNet(num_classes=9)
    with pytest.raises(ConfigException):
        validate_config(cf)


def test_validators():
    cf = _copy_class_object(Config)
    cf.model = partial(models.SSSNet, num_classes=9)
    v = CrossDatasetValidator(cf)
    assert v.split_column == "dataset"


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
