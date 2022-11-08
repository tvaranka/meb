from functools import partial

import pytest
import torch.nn as nn

from meb.core import Config
from meb import utils, models
from meb.utils.utils import ConfigException, validate_config


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
