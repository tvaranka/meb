from .base import BaseTestDataset


class TestCustomDataloader(BaseTestDataset):
    def test_indexing(self):
        c = self.per()
        d = c.data
        df = c.data_frame
        assert len(d[0].shape) == 3
        assert len(d[1:]) == 1
        assert len(d[:-1]) == 1
        assert len(d[[0, 1]]) == 2
        idx = df["subject"] == "g1"
        assert len(d[idx]) == 1
