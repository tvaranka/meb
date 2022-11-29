from functools import cached_property
from typing import Sequence

import pandas as pd

from meb.datasets import dataset_utils

cf = type(
    "Config",
    (object,),
    {
        "per_excel_path": "tests/data/per_annotation.xlsx",
        "per_dataset_path": "tests/data/PER",
        "per_cropped_dataset_path": "tests/data/PER_cropped",
        "per_optical_flow": "tests/data/per_of.npy",
    },
)


class PER(dataset_utils.Dataset, cf):
    def __init__(
        self,
        color: bool = False,
        resize: Sequence[int] | int = None,
        cropped: bool = True,
        optical_flow: bool = False,
        **kwargs,
    ) -> None:
        self.dataset_name = type(self).__name__.lower()
        self.dataset_path_format = "/{subject}/{material}/"
        super().__init__(color, resize, cropped, optical_flow, **kwargs)

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(self.per_excel_path)
        df = df.drop("notes", axis=1)
        df["n_frames"] = df["offset"] - df["onset"] + 1
        df = dataset_utils.extract_action_units(df)
        return df


class BaseTestDataset:
    @classmethod
    def setup_class(cls):
        # "per" is a fictitious dataset
        cls.per = PER
