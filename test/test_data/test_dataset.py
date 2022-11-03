from .base import BaseTestDataset
import numpy as np


class TestCustomDataset(BaseTestDataset):
    def test_action_unit_extraction(self):
        expected = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
        result = np.array(self.per.data_frame.loc[:, "AU1":]) == expected
        assert result.mean() == 1
