import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest
import get_image_size

from meb import datasets
from .base import BaseTestDataset
from meb.datasets.dataset_utils import LazyDataLoader, LoadedDataLoader
from meb.utils.utils import ConfigException


class TestCustomDataset(BaseTestDataset):
    def test_sample_number_correspondence(self):
        c = self.per()
        assert c.data_frame.shape[0] == c.data.shape[0]

    def test_action_unit_extraction(self):
        c = self.per()
        expected = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
        result = np.array(c.data_frame.loc[:, "AU1":]) == expected
        assert result.mean() == 1

    def test_optical_flow_type(self):
        c = self.per(optical_flow=True)
        assert isinstance(c.data, np.ndarray)

    def test_lazy_loading_type(self):
        c = self.per()
        assert isinstance(c.data, LazyDataLoader)

    def test_loaded_loading_type(self):
        c = self.per(preload=True)
        assert isinstance(c.data, LoadedDataLoader)

    def test_color(self):
        c = self.per(color=True)
        assert len(c.data[0].shape) == 4

    def test_cropped(self):
        c = self.per(cropped=True)
        assert "crop" in c.data.data_path[0][0].lower()

    def test_resize(self):
        c = self.per(resize=24)
        assert c.data[0].shape[1:] == (24, 24)

    def test_magnify(self):
        c = self.per()
        cm = self.per(magnify=True)
        assert ~np.allclose(c.data[0].astype("float32") / 255.0, cm.data[0])


@pytest.mark.parametrize(
    "dataset_name", ["Smic", "Casme", "Casme2", "Fourd", "Casme3A", "Casme3C"]
)
def test_datasets(dataset_name, mocker):
    # Setup mocks for data loading
    mm = MagicMock(spec=pd.DataFrame)
    mm.drop.return_value = mm
    mm.rename.return_value = mm
    mm.iterrows = MagicMock(return_value=[[MagicMock()] * 2])
    mocker.patch.object(pd, "read_excel", lambda _: mm)
    mocker.patch.object(os, "listdir", lambda _: ["img1.jpg"])
    mocker.patch.object(get_image_size, "get_image_size", lambda _: (12, 12))
    mocker.patch.object(plt, "imread", lambda _: np.zeros((12, 12)))

    # Actual test
    c = getattr(datasets, dataset_name)(ignore_validation=True)
    try:
        c.data_frame
        c.data[0]
    except ConfigException:
        pytest.fail("An exception was raised when it should not have been.")
