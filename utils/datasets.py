import os
import re
from itertools import chain
from experiments.config.dataset_config import config
from typing import List, Tuple, Optional, Sequence, Union, Callable
from utils.get_image_size import get_image_size
import utils.dataset_utils as dataset_utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from skimage.transform import resize as sk_resize


class Smic(dataset_utils.Dataset):
    def __init__(
            self,
            color: bool = False,
            resize: Union[Sequence[int], int, None] = None,
            cropped: bool = True,
            optical_flow: bool = False,
    ) -> None:
        super().__init__(color, resize, cropped, optical_flow)
        self.dataset_name = "smic"
        self.dataset_path_format = "/{subject}/micro/{emotion}/{material}/"

    @property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(config.smic_excel_path)
        df = df.drop("Unnamed: 0", axis=1)
        df["n_frames"] = df["offset"] - df["onset"] + 1
        # Set apex at the halfway of onset and offset
        df["apex"] = df["onset"] + ((df["offset"] - df["onset"]) / 2).astype("int")
        return df


class Casme(dataset_utils.Dataset):
    def __init__(
            self,
            color: bool = False,
            resize: Union[Sequence[int], int, None] = None,
            cropped: bool = True,
            optical_flow: bool = False
    ) -> None:
        super().__init__(color, resize, cropped, optical_flow)
        self.dataset_name = "casme"
        self.dataset_path_format = "/sub{subject}/{material}/"

    @property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(config.casme_excel_path)
        df = df.drop(["Unnamed: 2", "Unnamed: 7"], axis=1)
        df = df.rename(
            columns={
                "Emotion": "emotion",
                "Subject": "subject",
                "Filename": "material",
                "OnsetF": "onset",
                "ApexF1": "apex",
                "OffsetF": "offset",
            }
        )
        df["subject"] = df["subject"].apply(lambda x: f"{x:02d}")
        # Fix mistakes in the files based on the image folders
        df.loc[[40, 42, 43], "onset"] = [100, 101, 57]
        df.loc[[40, 42, 43, 54, 140], "offset"] = [149, 119, 74, 40, 142]
        # Apex to middle frame
        df.loc[[40, 42, 43], "apex"] = [120, 110, 65]

        df["n_frames"] = df["offset"] - df["onset"] + 1
        df = dataset_utils.extract_action_units(df)
        return df


class Casme2(dataset_utils.Dataset):
    def __init__(
            self,
            color: bool = False,
            resize: Union[Sequence[int], int, None] = None,
            cropped: bool = True,
            optical_flow: bool = False
    ) -> None:
        super().__init__(color, resize, cropped, optical_flow)
        self.dataset_name = "casme2"
        self.dataset_path_format = "/sub{subject}/{material}/"

    @property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(config.casme2_excel_path)
        df = df.drop(["Unnamed: 2", "Unnamed: 6"], axis=1)
        df = df.rename(
            columns={
                "Estimated Emotion": "emotion",
                "Subject": "subject",
                "Filename": "material",
                "OnsetFrame": "onset",
                "ApexFrame": "apex",
                "OffsetFrame": "offset",
                "Action Units": "AU",
            }
        )
        df["subject"] = df["subject"].apply(lambda x: f"{x:02d}")
        # Sample 222 in megc but not in individual casme2
        # df = df.drop(222).reset_index()
        # Mistake in file, change offset to 91
        df.loc[60, "offset"] = 91
        # Missing apex in file, change based on optical flow
        samples_missing_apex = [29, 35, 43, 45, 51, 53, 60, 117, 118, 126, 136,
                                147, 155, 168, 170, 177, 202, 203, 234, 237, 238,
                                ]
        estimated_apexes = [279, 68, 77, 81, 166, 100, 78, 187, 89, 80, 88,
                            134, 231, 53, 329, 111, 91, 103, 98, 153, 98,
                            ]
        df.loc[samples_missing_apex, "apex"] = estimated_apexes
        df["n_frames"] = df["offset"] - df["onset"] + 1
        df = dataset_utils.extract_action_units(df)
        return df



class SAMM(dataset_utils.Dataset):
    def __init__(
            self,
            color: bool = False,
            resize: Union[Sequence[int], int, None] = None,
            cropped: bool = True,
            optical_flow: bool = False,
    ) -> None:
        super().__init__(color, resize, cropped, optical_flow)
        self.dataset_name = "samm"
        self.dataset_path_format = "/{subject}/{material}/"

    @property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(config.samm_excel_path)
        # Preprocess the dataframe as it contains some text
        cols = df.loc[12].tolist()
        data = df.iloc[13:].reset_index()
        new_cols = {df.columns.tolist()[i]: cols[i] for i in range(len(cols))}
        df = pd.DataFrame(data).rename(columns=new_cols)

        df = df.rename(
            columns={
                "Estimated Emotion": "emotion",
                "Subject": "subject",
                "Filename": "material",
                "Onset Frame": "onset",
                "Apex Frame": "apex",
                "Offset Frame": "offset",
                "Action Units": "AU",
            }
        )
        df = df.replace({"Others": "others"})
        # Mistake in file, change offset
        df.loc[56, "offset"] = 5739
        # Missing apex in file, change based on optical flow
        df.loc[[125, 132, 133], "apex"] = [1105, 4945, 5130]
        df["n_frames"] = df["offset"] - df["onset"] + 1
        # Samples that have "or" in the AU. Change them based on optical flow
        df.loc[[15, 18, 41, 64, 154], "AU"] = [
            "R12 + R15",
            "R12 + R15",
            "L14",
            "R14",
            "R14A",
        ]
        df = dataset_utils.extract_action_units(df)
        return df


class FourDMicro(dataset_utils.Dataset):
    def __init__(
            self,
            color: bool = False,
            resize: Union[Sequence[int], int, None] = None,
            cropped: bool = True,
            optical_flow: bool = False,
    ) -> None:
        super().__init__(color, resize, cropped, optical_flow)
        self.dataset_name = "fourDmicro"
        self.dataset_path_format = "/{subject}/{material}/"

    @property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(config.fourDmicro_excel_path)
        df = df.rename(
            columns={
                "sub_ID": "subject",
                "fold_name": "material",
                "Emotion": "emotion",
                "Onset": "onset",
                "Apex": "apex",
                "Offset": "offset",
                "AUs": "AU",
            }
        )
        df[["onset", "offset", "apex"]] = df[["onset", "offset", "apex"]].astype("int")
        df.insert(9, "apexf", df["apex"] - df["onset"] + 1)
        df.insert(10, "n_frames", df["offset"] - df["onset"] + 1)
        # Switch 144 and 145
        tmp = df.loc[144, "AU":]
        df.loc[144, "AU":] = df.loc[145, "AU":]
        df.loc[145, "AU":] = tmp

        # Remove (k)s from data?
        for i in range(df.shape[0]):
            a = [au[:-3] for au in df.loc[i, "AU"].split("+") if "(k)" in au]
            a = [au[2:] if "AU" in au else au for au in a]
            for au in a:
                df.loc[i, "AU{}".format(au)] = 0
        return df


class MMEW(dataset_utils.Dataset):
    def __init__(
            self,
            color: bool = False,
            resize: Union[Sequence[int], int, None] = None,
            cropped: bool = True,
            optical_flow: bool = False,
    ) -> None:
        super().__init__(color, resize, cropped, optical_flow)
        self.dataset_name = "mmew"
        self.dataset_path_format = "/{emotion}/{material}/"

    @property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(config.mmew_excel_path)
        df = df.drop("remarks", axis=1)
        df = df.rename(
            columns={
                "Action Units": "AU",
                "Subject": "subject",
                "OnsetFrame": "onset",
                "ApexFrame": "apex",
                "OffsetFrame": "offset",
                "Estimated Emotion": "emotion",
                "Filename": "material",
            }
        )
        # Incorrect apex and offset, change them
        samples_missing_apex = [33, 35, 36, 37, 38, 39, 40, 41, 42, 142, 143, 145, 146, 148, 154, 156]
        estimated_apexes = [40, 15, 46, 20, 25, 20, 23, 28, 32, 56, 32, 27, 22, 29, 31, 38]
        estimated_offsets = [75, 54, 59, 98, 80, 93, 58, 61, 68, 88, 80, 85, 61, 108, 72, 106]
        df.loc[samples_missing_apex, ["apex", "offset"]] = np.array([estimated_apexes, estimated_offsets]).T
        df = dataset_utils.extract_action_units(df)
        df = df.replace({"others": "repression"})
        df["n_frames"] = df["offset"] - df["onset"] + 1
        return df


class CrossDataset(dataset_utils.Dataset):
    def __init__(
            self,
            color: bool = False,
            resize: Union[Sequence[int], int, None] = None,
            cropped: bool = True,
            optical_flow: bool = False,
    ) -> None:
        super().__init__(color, resize, cropped, optical_flow)
        self.dataset_name = ""
        self.dataset_path_format = ""
        self.datasets = [Casme, Casme2, SAMM, FourDMicro, MMEW]

    @property
    def data(self) -> Union[np.ndarray, "LazyDataLoader"]:
        return self.multi_dataset_data()

    @property
    def data_frame(self) -> pd.DataFrame:
        cross_dataset_dfs = []
        for dataset in self.cross_datasets:
            dataset_obj = dataset(cropped=self.cropped, color=self.color, resize=self.resize)
            df = dataset_obj.data_frame
            df.insert(11, "dataset", dataset_obj.dataset_name)
            cross_dataset_dfs.append(df)
        df = pd.concat(cross_dataset_dfs, axis=0, sort=False)
        df = df.drop(["ApexF2", "index", "Inducement Code", "Duration", "Micro", "Objective Classes",
                      "Notes", "fold", "eye blink", "Positive", "Negative", "Surprise", "Repression", "Others",
                      "Onset", "Total", "apexf"
                      ], axis=1,
                     )
        df = df.reset_index().drop("index", axis=1)
        df["n_frames"] = df["n_frames"].astype(int)
        # Sort action units
        aus_sorted = sorted(df.loc[:, "AU1":].columns, key=lambda x: int(x[2:]))
        meta_columns = df.loc[:, :"dataset"].columns.tolist()
        columns_sorted = meta_columns + aus_sorted
        df = df[columns_sorted]
        # Add apex from starting 0
        df.insert(5, "apexf", df["apex"] - df["onset"])
        # Fill empty action units
        df = df.fillna(0)
        df.loc[:, "AU1":] = df.loc[:, "AU1":].astype(int)
        return df