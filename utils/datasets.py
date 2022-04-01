import os
from itertools import chain
from experiments.config.dataset_config import config
from typing import List, Tuple, Optional, Sequence, Union
from utils.get_image_size import get_image_size

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from skimage.transform import resize as sk_resize


def only_digit(s: str) -> str:
    """Returns the digits of a string, e.g. AU12R -> 12"""
    return "".join(i for i in s if i.isdigit())


def extract_action_units(df: pd.DataFrame) -> pd.DataFrame:
    """Exctracts the action units from a single column to individual columns"""
    df["AU"] = df["AU"].astype("str")
    unique_aus = set(
        chain.from_iterable(
            df["AU"].apply(lambda x: [only_digit(i) for i in x.split("+")]).tolist()
        )
    )

    sorted_aus = sorted([int(au) for au in unique_aus if au != ""])
    for au in sorted_aus:
        df["AU{}".format(au)] = 0

    for i in range(df.shape[0]):
        subject_aus = df["AU"].apply(lambda x: [only_digit(i) for i in x.split("+")])[i]
        for au in sorted_aus:
            if str(au) in subject_aus:
                df.loc[i, "AU{}".format(au)] = 1
    return df


class CustomDataset:
    def __init__(
        self, data_path: List[List[str]], color: bool = False, resize=None
    ) -> None:
        self.data_path = data_path
        self.color = color
        self.resize = resize

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, index: int):
        if isinstance(index, int):
            data_path = self.data_path[index]
            return self._get_video(data_path)
        elif isinstance(index, list) or isinstance(index, tuple):
            data_path = [v for i, v in enumerate(self.data_path) if i in index]
            return CustomDataset(data_path, color=self.color, resize=self.resize)
        elif isinstance(index, slice):
            data_path = self.data_path[index]
            return CustomDataset(data_path, color=self.color, resize=self.resize)

    def __repr__(self) -> str:
        return f"CustomDataset with {len(self)} items from {self.data_path[0][0]}"

    def _create_array(self, image_paths: List[str]) -> np.ndarray:
        # Get image size without loading it
        w, h = get_image_size(image_paths[0])
        # Number of frames
        f = len(image_paths)
        resize = self.resize
        if resize:
            if isinstance(resize, int):
                resize = (resize, resize)
            video = np.zeros((f, resize[0], resize[1]), dtype="uint8")
        else:
            video = np.zeros((f, h, w), dtype="uint8")
        if self.color:
            video = np.expand_dims(video, axis=-1)
            video = np.repeat(video, 3, axis=-1)

        return video

    def _get_video(self, image_paths: List[str]) -> np.ndarray:
        video = self._create_array(image_paths)

        for f, image_path in enumerate(image_paths):
            image = plt.imread(image_path)
            if not self.color:
                # Check if image is already grayscale (SAMM)
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Check if greyscale and color wanted (SAMM)
            else:
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)
                    image = np.repeat(image, 3, axis=-1)
            if self.resize:
                if isinstance(self.resize, int):
                    self.resize = (self.resize, self.resize)
                image = sk_resize(image, (self.resize[0], self.resize[1]))
                image = (image * 255.0).astype("uint8")
            video[f] = image

        return video

    @property
    def shape(self):
        return (len(self),) + self[0].shape


def get_video_paths(format_path: str, df: pd.DataFrame) -> List[List[str]]:
    """
    Takes a format_path which specifies the specific path for individual dataset
    and a dataframe from the dataset
    """
    video_paths = []

    for i, row in df.iterrows():
        video_path = format_path.format(
            subject=row["subject"], emotion=row["emotion"], material=row["material"]
        )
        image_paths = os.listdir(video_path)
        image_paths = [video_path + image_path for image_path in image_paths]
        video_paths.append(image_paths)
    return video_paths


def smic(
    cropped: bool = True, color: bool = False, resize: Union[Sequence[int], int, None] = None
) -> Tuple[pd.DataFrame, CustomDataset]:
    """Returns a pandas dataframe with the metadata and an iterable object with the data."""

    def process_df() -> pd.DataFrame:
        df = pd.read_excel(config.smic_excel_path)
        df = df.drop("Unnamed: 0", axis=1)
        df["n_frames"] = df["offset"] - df["onset"] + 1
        return df

    def process_data(df: pd.DataFrame) -> CustomDataset:
        dataset_path = (
            config.smic_cropped_dataset_path if cropped else config.smic_dataset_path
        )
        format_path = dataset_path + "/{subject}/micro/{emotion}/{material}/"
        video_paths = get_video_paths(format_path, df)
        dataset = CustomDataset(video_paths, color=color, resize=resize)
        return dataset

    df = process_df()
    dataset = process_data(df)
    return df, dataset


def casme(
    cropped: bool = True, color: bool = False, resize: Union[Sequence[int], int, None] = None
) -> Tuple[pd.DataFrame, CustomDataset]:
    def process_df() -> pd.DataFrame:
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
        df.loc[[40, 42, 43], "onset"] = [108, 101, 57]
        df.loc[[40, 42, 43, 54], "offset"] = [149, 119, 74, 40]
        # Apex to middle frame
        df.loc[[40, 42, 43], "apex"] = [120, 110, 65]

        df["n_frames"] = df["offset"] - df["onset"] + 1
        df = extract_action_units(df)
        return df

    def process_data(df: pd.DataFrame) -> CustomDataset:
        dataset_path = (
            config.casme_cropped_dataset_path if cropped else config.casme_dataset_path
        )
        format_path = dataset_path + "/sub{subject}/{material}/"
        video_paths = get_video_paths(format_path, df)
        dataset = CustomDataset(video_paths, color=color, resize=resize)
        return dataset

    df = process_df()
    dataset = process_data(df)
    return df, dataset


def casme2(
    cropped: bool = True, color: bool = False, resize: Union[Sequence[int], int, None] = None
) -> Tuple[pd.DataFrame, CustomDataset]:
    def process_df() -> pd.DataFrame:
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
        df["n_frames"] = df["offset"] - df["onset"] + 1
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
        df = extract_action_units(df)
        return df

    def process_data(df: pd.DataFrame) -> CustomDataset:
        dataset_path = (
            config.casme2_cropped_dataset_path if cropped else config.casme2_dataset_path
        )
        format_path = dataset_path + "/sub{subject}/{material}/"
        video_paths = get_video_paths(format_path, df)
        dataset = CustomDataset(video_paths, color=color, resize=resize)
        return dataset

    df = process_df()
    dataset = process_data(df)
    return df, dataset


def samm(
    cropped: bool = True, color: bool = False, resize: Union[Sequence[int], int, None] = None
) -> Tuple[pd.DataFrame, CustomDataset]:
    def process_df() -> pd.DataFrame:
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
        df = extract_action_units(df)
        return df

    def process_data(df: pd.DataFrame) -> CustomDataset:
        dataset_path = (
            config.samm_cropped_dataset_path if cropped else config.samm_dataset_path
        )
        format_path = dataset_path + "/{subject}/{material}/"
        video_paths = get_video_paths(format_path, df)
        # color is always false for samm as the images are grayscale
        dataset = CustomDataset(video_paths, color=False, resize=resize)
        return dataset

    df = process_df()
    dataset = process_data(df)
    return df, dataset


def fourDmicro(
    cropped: bool = True, color: bool = False, resize: Union[Sequence[int], int, None] = None
) -> Tuple[pd.DataFrame, CustomDataset]:
    df = pd.read_excel(config.fourDmicro_excel_path)
    df = df[:-1]
    df = df.rename(
        columns={
            "sub_ID'": "subject",
            "video_number'": "material",
            "video emo'": "video_emotion",
            "self-report emo'": "self-report emotion",
            "Emotion": "emotion",
            "Onset": "onset",
            "Apex": "apex",
            "Offset": "offset",
            "Length": "n_frames",
            "AUs": "AU",
        }
    )
    df.columns = [
        column[:-1] if column.endswith("'") else column for column in df.columns
    ]
    df = df.apply(
        lambda column: column.apply(
            lambda x: str(x).replace("'", "") if "'" in str(x) else x
        )
    )
    df = df.replace({"'": ""})
    df[["onset", "offset", "apex"]] = df[["onset", "offset", "apex"]].astype("int")
    # Apex out of bounds
    df.loc[26, "apex"] = 2490
    df["apexf"] = df["apex"] - df["onset"] + 1
    df = df[df.columns.tolist()[:11] + ["apexf"] + df.columns.tolist()[11:-1]]
    df = df.replace(
        {
            "0'": "neutral",
            "1'": "happy",
            "1,3'": "happy + surprise",
            "2'": "sad",
            "3'": "surprise",
            "4'": "fear",
            "4,5'": "fear/disgust",
            "5'": "disgust",
        }
    )
    df["subME_number"] = df["subME_number"].astype("int").astype("str")
    if cropped:
        subjects = sorted(os.listdir(config.fourDmicro_cropped_dataset_path))
    else:
        subjects = sorted(os.listdir(config.fourDmicro_dataset_path))
    subjects.remove("S13_1st_001_01_1")

    # df.loc[19, "subject"] = "S13_1st"
    df["subject"] = [
        subject[:-9] if not subject[:-9].endswith("_") else subject[:-10]
        for subject in subjects
    ]
    df["material"] = [subject[-8:] for subject in subjects]
    # Fix random dashes at the end of material
    df["material"] = df["material"].apply(lambda x: "0" + x if x.endswith("-") else x)

    df.loc[263, "emotion"] = "Surprise+Positive"
    tmp = df.loc[145]
    df.loc[145] = df.loc[144]
    df.loc[144] = tmp
    df.loc[245, "n_frames"] = 24
    # Remove static(marked as (k) in the file) AUs
    df.loc[77, "AU"] = df.loc[77, "AU"][:-1]
    df.loc[124, "AU"] = "AU4(k)+L7"
    df.loc[171, "AU"] = "AUR2+12(k)+45"
    for i in range(267):
        a = [au[:-3] for au in df.loc[i, "AU"].split("+") if "(k)" in au]
        a = [au[2:] if "AU" in au else au for au in a]
        for au in a:
            df.loc[i, "AU{}".format(au)] = 0

    dataset_path = (
        config.fourDmicro_cropped_dataset_path
        if cropped
        else config.fourDmicro_dataset_path
    )
    format_path = dataset_path + "/{subject}_{material}/"
    video_paths = get_video_paths(format_path, df)
    dataset = CustomDataset(video_paths, color=color, resize=resize)

    # Remove _1st and _2nd from df subjects
    second_subjects_dict = {
        "S13_1st": "S13",
        "S13_2nd": "S13",
        "S30_1st": "S30",
        "S30_2nd": "S30",
        "S49_2": "S49",
        "S52_2": "S52",
        "S52_2": "S52",
        "S58_1": "S58",
        "S58_2": "S58",
        "S59_2,": "S59",
    }
    df["subject"] = df["subject"].replace(second_subjects_dict)

    return df, dataset


def mmew(
    cropped: bool = True, color: bool = False, resize: Optional[Sequence[int]] = None
) -> Tuple[pd.DataFrame, CustomDataset]:
    def process_df() -> pd.DataFrame:
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
        df = extract_action_units(df)
        df = df.replace({"others": "repression"})
        df["n_frames"] = df["offset"] - df["onset"] + 1
        return df

    def process_data(df: pd.DataFrame) -> CustomDataset:
        dataset_path = (
            config.mmew_cropped_dataset_path if cropped else config.mmew_dataset_path
        )
        format_path = dataset_path + "/{emotion}/{material}/"
        video_paths = get_video_paths(format_path, df)
        dataset = CustomDataset(video_paths, color=color, resize=resize)
        return dataset

    df = process_df()
    dataset = process_data(df)
    return df, dataset


def megc_old(
    cropped: bool = True, color: bool = False, resize: Optional[Sequence[int]] = None
) -> Tuple[pd.DataFrame, CustomDataset]:

    df_samm, data_samm = samm(cropped=cropped, color=color, resize=resize)
    df_casme2, data_casme2 = casme2(cropped=cropped, color=color, resize=resize)
    df_smic, data_smic = smic(cropped=cropped, color=color, resize=resize)

    # Add dataset information to each dataset
    df_smic["dataset"] = "smic"
    df_casme2["dataset"] = "casme2"
    df_samm["dataset"] = "samm"

    # Remove fear, sadness and others from casme2
    indices = df_casme2[~df_casme2["emotion"].isin(["fear", "sadness", "others"])][
        "emotion"
    ].index.tolist()
    data_casme2 = data_casme2[indices]
    df_casme2 = df_casme2.loc[indices]
    # Set the correct emotions
    df_casme2.loc[
        df_casme2["emotion"].isin(["disgust", "repression"]), "emotion"
    ] = "negative"
    df_casme2.loc[df_casme2["emotion"] == "happiness", "emotion"] = "positive"

    # remove "others" from samm
    indices2 = df_samm[df_samm["emotion"] != "Other"]["emotion"].index.tolist()
    data_samm = data_samm[indices2]
    df_samm = df_samm[df_samm["emotion"] != "Other"]
    # Set the correct emotions
    df_samm.loc[
        df_samm["emotion"].isin(["Anger", "Contempt", "Disgust", "Sadness", "Fear"]),
        "emotion",
    ] = "negative"
    df_samm.loc[df_samm["emotion"] == "Happiness", "emotion"] = "positive"
    df_samm.loc[df_samm["emotion"] == "Surprise", "emotion"] = "surprise"

    # merge dataframes and iterators
    df = pd.concat([df_casme2, df_smic, df_samm], sort=True)
    df = df.reset_index()
    df = df.drop(
        [
            "Duration",
            "Inducement Code",
            "Micro",
            "Notes",
            "Objective Classes",
            "level_0",
            "index",
        ],
        axis=1,
    )
    columns_with_no_aus = [
        column for column in df.columns if "au" not in column.lower()
    ]
    df = df[columns_with_no_aus]

    # apex is the apex frame based on file number and apexf for image number in sequence
    df.loc[df.index[df["apex"].isnull()], "apex"] = (
        df[df["apex"].isnull()]["offset"] - df[df["apex"].isnull()]["onset"]
    ) / 2
    df["apexf"] = df["apex"] - df["onset"]
    df.loc[df["dataset"] == "smic", "apexf"] = df.loc[df["dataset"] == "smic", "apex"]
    df.loc[:, "apex"] = round(df["apex"].astype("float")).astype("int")
    df.loc[:, "apexf"] = round(df["apexf"].astype("float")).astype("int")

    # Drop some samples that are not part of MEGC
    df = df.drop([17, 26, 130])
    total_data_path = data_casme2.data_path + data_smic.data_path + data_samm.data_path
    data = CustomDataset(total_data_path)
    indices = df.index.tolist()
    data = data[indices]
    df = df.reset_index()

    return df, data


def megc(
    cropped: bool = True,
    color: bool = False,
    resize: Optional[Sequence[int]] = None,
    optical_flow: bool = False,
) -> Tuple[pd.DataFrame, Union[CustomDataset, np.ndarray]]:
    megc_datasets = [smic, casme2, samm]

    def process_df() -> Tuple[pd.DataFrame, List[int]]:
        cross_dataset_dfs = []
        for dataset in megc_datasets:
            df = dataset(cropped=cropped, color=color, resize=resize)[0]
            df.insert(5, "dataset", dataset.__name__)
            cross_dataset_dfs.append(df)
        df = pd.concat(cross_dataset_dfs, axis=0, sort=False)
        df = df.reset_index().drop("index", axis=1)
        # Add apex to smic as its missing, use half frame
        df.loc[df.index[df["apex"].isnull()], "apex"] = ((
                df[df["apex"].isnull()]["offset"] - df[df["apex"].isnull()]["onset"]
        ) / 2).astype(int)
        # Add frame and apex information
        df["n_frames"] = df["n_frames"].astype(int)
        df.insert(5, "apexf", df["apex"] - df["onset"])
        df.loc[df["dataset"] == "smic", "apexf"] = df.loc[df["dataset"] == "smic", "apex"]
        # Drop AUs and other information from the dataframe
        aus = df.columns[["AU" in column for column in df.columns]]
        df = df.drop(aus, axis=1)
        df = df.drop(["Inducement Code", "Duration", "Micro", "Objective Classes", "Notes", "level_0"], axis=1)
        #Set correct emotions for megc
        df.loc[
            df["emotion"].isin(["Anger", "Contempt", "Disgust", "Sadness", "Fear", "disgust", "repression"]),
            "emotion",
        ] = "negative"
        df.loc[df["emotion"].isin(["Happiness", "happiness"]), "emotion"] = "positive"
        df.loc[df["emotion"] == "Surprise", "emotion"] = "surprise"
        # Remove samples that do not belong to megc
        indices_to_remove = df[df["emotion"].isin(
            ["fear", "sadness", "others", "Other"])]["emotion"].index.tolist()
        indices_to_remove.extend(df[(df["subject"] == "04") & (df["material"] == "EP12_01f")].index.tolist())
        indices_to_remove.extend(df[(df["subject"] == "05") & (df["material"] == "EP12_03f")].index.tolist())
        indices_to_remove.extend(df[(df["subject"] == "24") & (df["material"] == "EP02_07")].index.tolist())
        df = df.drop(indices_to_remove).reset_index(drop=True)
        return df, indices_to_remove

    def process_data() -> CustomDataset:
        if optical_flow:
            of_frames = load_optical_flow_data(megc_datasets, resize=resize)
            return of_frames
        data_paths = [
            video_path
            for dataset in megc_datasets
            for video_path in dataset()[1].data_path
        ]
        data = CustomDataset(data_paths, color=color, resize=resize)
        return data
    df, indices_to_remove = process_df()
    data = process_data()
    indices_to_keep = [i for i in range(len(data)) if i not in indices_to_remove]
    data = data[indices_to_keep]

    return df, data


def cross_dataset(
    cropped: bool = True,
    color: bool = False,
    resize: Optional[Sequence[int]] = None,
    optical_flow: bool = False,
) -> Tuple[pd.DataFrame, Union[CustomDataset, np.ndarray]]:

    cross_datasets = [casme, casme2, samm, fourDmicro, mmew]
    cross_dataset_dfs = []
    for dataset in cross_datasets:
        df = dataset(cropped=cropped, color=color, resize=resize)[0]
        df.insert(11, "dataset", dataset.__name__)
        cross_dataset_dfs.append(df)
    df = pd.concat(cross_dataset_dfs, axis=0, sort=False)
    df = df.drop(["ApexF2", "index", "Inducement Code", "Duration", "Micro", "Objective Classes",
                  "Notes", "fold", "ME_number", "subME_number", "first_frame", "last_frame", "video_emotion",
                  "self-report emotion", "eye blink", "Positive", "Negative", "Surprise", "Repression", "Others",
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


    if optical_flow:
        of_frames = load_optical_flow_data(cross_datasets, resize=resize)
        return df, of_frames

    data_paths = [
        video_path
        for dataset in cross_datasets
        for video_path in dataset()[1].data_path
    ]
    data = CustomDataset(data_paths, color=color, resize=resize)

    return df, data


def load_optical_flow_data(cross_datasets, resize):
    if isinstance(resize, int):
        h = w = resize
    elif isinstance(resize, tuple):
        h = resize[0]
        w = resize[1]
    else:
        h = w = 64
    c = 3
    of_frames_list = []
    for dataset in cross_datasets:
        dataset_name = dataset.__name__
        of_path = getattr(config, f"{dataset_name}_optical_flow")
        of_frames = np.load(of_path)
        n_samples = len(of_frames)
        of_frames = sk_resize(of_frames, (n_samples, c, h, w))
        of_frames_list.append(of_frames)
    of_frames = np.concatenate(of_frames_list)
    return of_frames
