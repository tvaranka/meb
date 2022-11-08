# from ..test_data.base import BaseTestDataset
# from meb.datasets.dataset_utils import LazyDataLoader, LoadedDataLoader
#
# import numpy as np
#
#
# class TestCustomDataset(BaseTestDataset):
#     def test_sample_number_correspondence(self):
#         c = self.per()
#         assert c.data_frame.shape[0] == c.data.shape[0]
#
#     def test_action_unit_extraction(self):
#         c = self.per()
#         expected = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
#         result = np.array(c.data_frame.loc[:, "AU1":]) == expected
#         assert result.mean() == 1
#
#     def test_optical_flow_type(self):
#         c = self.per(optical_flow=True)
#         assert isinstance(c.data, np.ndarray)
#
#     def test_lazy_loading_type(self):
#         c = self.per()
#         assert isinstance(c.data, LazyDataLoader)
#
#     def test_loaded_loading_type(self):
#         c = self.per(preload=True)
#         assert isinstance(c.data, LoadedDataLoader)
#
#     def test_color(self):
#         c = self.per(color=True)
#         assert len(c.data[0].shape) == 4
#
#     def test_cropped(self):
#         c = self.per(cropped=True)
#         assert "crop" in c.data.data_path[0][0].lower()
#
#     def test_resize(self):
#         c = self.per(resize=24)
#         assert c.data[0].shape[1:] == (24, 24)
#
#     def test_magnify(self):
#         c = self.per()
#         cm = self.per(magnify=True)
#         assert ~np.allclose(c.data[0].astype("float32") / 255.0, cm.data[0])


# from functools import partial
#
# import pytest
# import torch.nn as nn
#
# from meb.datasets.dataset_utils import LazyDataLoader, LoadedDataLoader
# from meb.core import Config  # , CrossDatasetValidator
# from meb import utils, models
# from meb.utils.utils import ConfigException, validate_config
#
# # from ..test_data.base import BaseTestDataset
#
# a = LazyDataLoader
# b = LoadedDataLoader
#
#
# def _copy_class_object(obj: object):
#     """Copies class objects and their attributes."""
#     return type("Object", (object,), dict(obj.__dict__))
#
#
# def test_correct_config():
#     cf = _copy_class_object(Config)
#     cf.model = partial(models.SSSNet, num_classes=9)
#     try:
#         validate_config(cf)
#     except ConfigException:
#         pytest.fail("An exception was raised when it should not have been.")
#
#
# def test_incorrect_config():
#     cf = _copy_class_object(Config)
#     cf.evaluation_fn = utils.MultiLabelF1Score()
#     with pytest.raises(ConfigException):
#         validate_config(cf)
#
#     cf = _copy_class_object(Config)
#     cf.criterion = nn.CrossEntropyLoss()
#     with pytest.raises(ConfigException):
#         validate_config(cf)
#
#     cf = _copy_class_object(Config)
#     cf.model = models.SSSNet(num_classes=9)
#     with pytest.raises(ConfigException):
#         validate_config(cf)
#
#
# # def test_validators():
# #     cf = _copy_class_object(Config)
# #     cf.model = partial(models.SSSNet, num_classes=9)
# #     v = CrossDatasetValidator(cf)
# #     assert v.split_column == "dataset"
# #
# #
# # class TestValidator(BaseTestDataset):
# #     def test_validate_output(self):
# #         # Should be output list
# #         pass
# #
# #     def test_validate_print(self):
# #         # Test final printing output
# #         pass
# #
# #     def test_printer(self):
# #         pass
