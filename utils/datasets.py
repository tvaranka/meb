import os
import re
from itertools import chain
from experiments.config.dataset_config import config
from typing import List, Tuple, Optional, Sequence, Union, Callable
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
        self, data_path: List[List[str]], color: bool = False, resize=None, n_sample: int = 6
    ) -> None:
        self.data_path = data_path
        self.color = color
        self.resize = resize
        self.n_sample = n_sample

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

    def get_video_sampled(self, index: int, sampling_func: Callable = None):
        if not isinstance(index, int):
            raise NotImplementedError("Currently only accepts a single integer index")
        data_path = self.data_path[index]
        if sampling_func:
            n_frames = get_number_of_frames(data_path)
            frame_inds = sampling_func(self.n_sample, n_frames)
            data_path = [path for frame_n, path in enumerate(data_path) if frame_n in frame_inds]
        return self._get_video(data_path)

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
                image = sk_resize(image, (self.resize[0], self.resize[1]), anti_aliasing=True)
                image = (image * 255.0).astype("uint8")
            video[f] = image

        return video

    @property
    def shape(self):
        return (len(self),) + self[0].shape


def get_number_of_files(folder: str):
    """
    Returns the number of files in a folder. Used for getting the number of frames
    """
    return len(os.listdir(folder))


def get_number_of_frames(subject_frame_paths: List[str]):
    """
    Returns the number of frames based on the first frame path.
    """
    subject_folder = "/".join(subject_frame_paths[0].split("/")[:-1])
    return get_number_of_files(subject_folder)


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
        file_formats = ["jpg", "bmp", "png"]
        image_paths = [image_path for image_path in image_paths if image_path[-3:] in file_formats]
        image_paths = [video_path + image_path for image_path in image_paths]
        image_paths = sort_video_path(image_paths)
        video_paths.append(image_paths)
    return video_paths


def sort_video_path(video_path: List[str]):
    """
    Video paths may be given in a random order on linux devices, where the frames are in
    a random order. Sorts the frames to the correct order.
    """
    # Get the frame information by splitting with "/" and then taking the last part.
    # Get only the digits and make it an int
    # Use the i to get an index which is then used to sort the video path
    d = {i: int(only_digit(data_path.split("/")[-1])) for i, data_path in enumerate(video_path)}
    idx = list(dict(sorted(d.items(), key=lambda x: x[1])).keys())
    # Use numpy array for convenient indexing
    sorted_video_path = list(np.array(video_path)[idx])
    return sorted_video_path


def smic(
    cropped: bool = True,
    color: bool = False,
    resize: Union[Sequence[int], int, None] = None,
    optical_flow: bool = False
) -> Tuple[pd.DataFrame, CustomDataset]:
    """Returns a pandas dataframe with the metadata and an iterable object with the data."""

    def process_df() -> pd.DataFrame:
        df = pd.read_excel(config.smic_excel_path)
        df = df.drop("Unnamed: 0", axis=1)
        df["n_frames"] = df["offset"] - df["onset"] + 1
        # Set apex at the halfway of onset and offset
        df["apex"] = df["onset"] + ((df["offset"] - df["onset"]) / 2).astype("int")
        return df

    def process_data(df: pd.DataFrame) -> CustomDataset:
        if optical_flow:
            of_frames = load_optical_flow_data("smic", resize=resize)
            return of_frames
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
    cropped: bool = True,
    color: bool = False,
    resize: Union[Sequence[int], int, None] = None,
    optical_flow: bool = False
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
        df.loc[[40, 42, 43], "onset"] = [100, 101, 57]
        df.loc[[40, 42, 43, 54, 140], "offset"] = [149, 119, 74, 40, 142]
        # Apex to middle frame
        df.loc[[40, 42, 43], "apex"] = [120, 110, 65]

        df["n_frames"] = df["offset"] - df["onset"] + 1
        df = extract_action_units(df)
        return df

    def process_data(df: pd.DataFrame) -> CustomDataset:
        if optical_flow:
            of_frames = load_optical_flow_data("casme", resize=resize)
            return of_frames
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
    cropped: bool = True,
    color: bool = False,
    resize: Union[Sequence[int], int, None] = None,
    optical_flow: bool = False
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
        df = extract_action_units(df)
        return df

    def process_data(df: pd.DataFrame) -> CustomDataset:
        if optical_flow:
            of_frames = load_optical_flow_data("casme2", resize=resize)
            return of_frames
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
    cropped: bool = True,
    color: bool = False,
    resize: Union[Sequence[int], int, None] = None,
    optical_flow: bool = False
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
        if optical_flow:
            of_frames = load_optical_flow_data("samm", resize=resize)
            return of_frames
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
    cropped: bool = True,
    color: bool = False,
    resize: Union[Sequence[int], int, None] = None,
    optical_flow: bool = False
) -> Tuple[pd.DataFrame, CustomDataset]:
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
    df.insert(10, "n_frames",  df["offset"] - df["onset"] + 1)
    #Switch 144 and 145
    tmp = df.loc[144, "AU":]
    df.loc[144, "AU":] = df.loc[145, "AU":]
    df.loc[145, "AU":] = tmp

    # Remove (k)s from data?
    for i in range(df.shape[0]):
        a = [au[:-3] for au in df.loc[i, "AU"].split("+") if "(k)" in au]
        a = [au[2:] if "AU" in au else au for au in a]
        for au in a:
            df.loc[i, "AU{}".format(au)] = 0

    dataset_path = (
        config.fourDmicro_cropped_dataset_path
        if cropped
        else config.fourDmicro_dataset_path
    )
    format_path = dataset_path + "/{subject}/{material}/"
    video_paths = get_video_paths(format_path, df)
    if optical_flow:
        of_frames = load_optical_flow_data("fourDmicro", resize=resize)
        return df, of_frames
    dataset = CustomDataset(video_paths, color=color, resize=resize)
    return df, dataset


def mmew(
    cropped: bool = True,
    color: bool = False,
    resize: Union[Sequence[int], int, None] = None,
    optical_flow: bool = False
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
        # Incorrect apex and offset, change them
        samples_missing_apex = [33, 35, 36, 37, 38, 39, 40, 41, 42, 142, 143, 145, 146, 148, 154, 156]
        estimated_apexes = [40, 15, 46, 20, 25, 20, 23, 28, 32, 56, 32, 27, 22, 29, 31, 38]
        estimated_offsets = [75, 54, 59, 98, 80, 93, 58, 61, 68, 88, 80, 85, 61, 108, 72, 106]
        df.loc[samples_missing_apex, ["apex", "offset"]] = np.array([estimated_apexes, estimated_offsets]).T
        df = extract_action_units(df)
        df = df.replace({"others": "repression"})
        df["n_frames"] = df["offset"] - df["onset"] + 1
        return df

    def process_data(df: pd.DataFrame) -> CustomDataset:
        if optical_flow:
            of_frames = load_optical_flow_data("mmew", resize=resize)
            return of_frames
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
            of_frames_list = []
            for dataset in megc_datasets:
                of_frames = dataset(cropped=cropped, color=color, resize=resize, optical_flow=optical_flow)[1]
                of_frames_list.append(of_frames)
            of_frames = np.concatenate(of_frames_list)
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


    if optical_flow:
        of_frames_list = []
        for dataset in cross_datasets:
            of_frames = dataset(cropped=cropped, color=color, resize=resize, optical_flow=optical_flow)[1]
            of_frames_list.append(of_frames)
        of_frames = np.concatenate(of_frames_list)
        return df, of_frames

    data_paths = [
        video_path
        for dataset in cross_datasets
        for video_path in dataset()[1].data_path
    ]
    data = CustomDataset(data_paths, color=color, resize=resize)

    return df, data


def load_optical_flow_data(dataset_name: str, resize: Union[Sequence[int], int, None] = None):
    if isinstance(resize, int):
        h = w = resize
    elif isinstance(resize, tuple):
        h = resize[0]
        w = resize[1]
    else:
        h = w = 64
    of_path = getattr(config, f"{dataset_name}_optical_flow")
    of_frames = np.load(of_path)
    n_samples, c, _, _ = of_frames.shape
    if resize:
        of_frames = sk_resize(of_frames, (n_samples, c, h, w))
    return of_frames


def validate_apex(df: pd.DataFrame) -> None:
    """
    Validates that the apex is not out of bounds using the dataframe.
    """
    assert (df["apex"] - df["onset"] < 0).sum() == 0, "Apex is lower than onset."
    assert (df["offset"] - df["apex"] < 0).sum() == 0, "Apex is greater than offset."


def validate_onset_offset(df: pd.DataFrame, data: CustomDataset) -> None:
    """
    Validates that the onset and offset values correspond in the
    dataset and in the dataframe.
    """
    for i, row in df.iterrows():
        onset_path_last = data.data_path[i][0].split("/")[-1]
        onset_f = int(re.findall("\d+", onset_path_last)[-1])
        assert onset_f == row["onset"], \
            f"The onset does not correspond for the sample\
            {row['subject']}_{row['material']}"
        offset_path_last = data.data_path[i][-1].split("/")[-1]
        offset_f = int(re.findall("\d+", offset_path_last)[-1])
        assert offset_f == row["offset"], \
            f"The offset does not correspond for the sample\
            {row['subject']}_{row['material']}"


def validate_n_frames(df: pd.DataFrame, data: CustomDataset) -> None:
    """
    Validates whether the number of frames in the dataframe and dataset
    correspond.
    """
    for i, row in df.iterrows():
        n_frames_df = row["n_frames"]
        n_frames_data = len(data.data_path[i])
        assert n_frames_df == n_frames_data, f"\
            The number of frames does not correspond for the sample\
             {row['subject']}_{row['material']}"


def validate_frames_ascending_order(data: CustomDataset) -> None:
    """
    Validates whether the frames are in the correct (ascending) order. The frames may be loaded in a random order
    on some Linux systems.
    """
    for data_path in data.data_path:
        frame_ns = []
        for img_path in data_path:
            last_part = img_path.split("/")[-1]
            frame_n = int(re.findall("\d+", last_part)[-1])
            frame_ns.append(frame_n)
        sorted_frame_ns = sorted(frame_ns)
        assert sorted_frame_ns == frame_ns, \
            f"The frames are not in an ascending order for sample {img_path}"


def validate_dataset(df: pd.DataFrame, data: CustomDataset) -> None:
    """
    Performs a set of validation tests to make sure that the dataframe and the dataset are consistent.
    An assertion is thrown if there is an issue.
    """
    validate_apex(df)
    validate_n_frames(df, data)
    validate_onset_offset(df, data)
    validate_frames_ascending_order(data)


