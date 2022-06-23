import os
import re
from itertools import chain
from typing import List, Tuple, Sequence, Union, Callable
from abc import ABC, abstractmethod
from tqdm import tqdm
import functools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize as sk_resize

from utils.get_image_size import get_image_size
from experiments.config.dataset_config import config


class LazyDataLoader:
    """
    Loads video data on demand.
    Examples
    # Create a data loader object from paths to the image files List[List[str]]
    loader = LazyDataLoader(data_path)
    # The video is loaded once indexed
    first_video = loader[0]
    # Supports slicing
    first_ten_videos = loader[:10]
    # And multi-indexing
    some_videos = loader[[4, 7, 8, 11, 56]]
    # Also works in a loop
    for video in loader:
        do_stuff(video)

    Examples from arguments
    LazyDataLoader(resize=64)
    -> shape (n_frames, 1, 64, 64)
    LazyDataLoader(resize=(140, 170))
    -> shape (n_frames, 1, 140, 170)
    LazyDataLoader(n_sample=10)
    -> shape (10, 1, width, height)
    """
    def __init__(
        self, data_path: List[List[str]], color: bool = False, resize=None, n_sample: int = 6,
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
        elif isinstance(index, Sequence):
            data_path = [self.data_path[i] for i in index]
            return LazyDataLoader(data_path, color=self.color, resize=self.resize)
        elif isinstance(index, slice):
            data_path = self.data_path[index]
            return LazyDataLoader(data_path, color=self.color, resize=self.resize)

    def get_video_sampled(self, index: int, sampling_func: Callable = None):
        if not isinstance(index, int):
            raise NotImplementedError("Currently only accepts a single integer index")
        data_path = self.data_path[index]
        if sampling_func:
            n_frames = get_number_of_frames(data_path)
            frame_inds = sampling_func(self.n_sample, n_frames)
            data_path = [data_path[i] for i in frame_inds]
        return self._get_video(data_path)

    def __repr__(self) -> str:
        return f"LazyDataLoader with {len(self)} items from {self.data_path[0][0]}"

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


class LoadedDataLoader:
    def __init__(self, data_in: Union[LazyDataLoader, List[np.ndarray]], n_sample: int = 6) -> None:
        if isinstance(data_in, LazyDataLoader):
            self.data_loader = data_in
            self.data = self.load_data()
            self.data_path = data_in.data_path
            self.n_sample = data_in.n_sample
        else:
            self.data = data_in
            self.n_sample = n_sample

    def load_data(self):
        return [video for video in tqdm(self.data_loader, desc="Loading data")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        if isinstance(index, int):
            return self.data[index]
        elif isinstance(index, Sequence):
            data = [self.data[i] for i in index]
            new = type(self)(data, self.n_sample)
            return new
        elif isinstance(index, slice):
            data = self.data[index]
            new = type(self)(data, self.n_sample)
            return new

    def __repr__(self) -> str:
        return f"LoadedDataLoader with {len(self)} items from {self.data_path[0][0]}"

    def get_video_sampled(self, index: int, sampling_func: Callable = None):
        if not isinstance(index, int):
            raise NotImplementedError("Currently only accepts a single integer index")
        if sampling_func:
            n_frames = len(self.data[index])
            frame_inds = sampling_func(self.n_sample, n_frames)
        return self.data[index][frame_inds]


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


class Dataset(ABC):
    """
    Base class for datasets.
    Examples
    # Create a dataset object which can then be used to extract the data frame and the data itself
    dataset = Smic()
    df = smic.data_frame
    lazy_data_loader = smic.data
    # Setting the optical_flow flag true
    of_frames = Smic(optical_flow).data
    # Multi dataset with optical flow resized to 64
    cross_dataset = CrossDataset(resize=64, optical_flow=True)
    # Use a custom face alignment technique
    cross_dataset = CrossDataset(cropped=False, color=True)
    # Indexing returns a data frame and a LazyDataLoader
    c = CrossDataset()
    ten_df, ten_data_loader = c[:10]
    """
    dataset_name: str
    dataset_path_format: str

    def __init__(
        self,
        color: bool = False,
        resize: Union[Sequence[int], int, None] = None,
        cropped: bool = True,
        optical_flow: bool = False,
        n_sample: int = 6,
        preload: bool = False,
    ) -> None:
        self.color = color
        self.resize = resize
        self.cropped = cropped
        self.optical_flow = optical_flow
        self.n_sample = n_sample
        self.preload = preload
        if not self.optical_flow and self.dataset_name:
            validate_dataset(self.data_frame, self.data)

    @property
    @abstractmethod
    def data_frame(self) -> pd.DataFrame:
        """
        Loads the excel file into a Pandas data frame and modifies possible issues
        and generates additional features.
        """

    @property
    @functools.lru_cache()
    def data(self) -> Union[np.ndarray, LazyDataLoader]:
        """
        Loads the dataset given a path. If the optical_flow flag is set to True, a numpy array is returned, else
        RGB frames are returned in the LazyDataLoader object.
        """
        if self.optical_flow:
            return load_optical_flow_data(self.dataset_name, resize=self.resize)
        crop_str = "_cropped" if self.cropped else ""
        dataset_path = getattr(config, f"{self.dataset_name}{crop_str}_dataset_path")
        format_path = dataset_path + self.dataset_path_format
        video_paths = get_video_paths(format_path, self.data_frame)
        dataset = LazyDataLoader(video_paths, color=self.color, resize=self.resize, n_sample=self.n_sample)
        if self.preload:
            dataset = LoadedDataLoader(dataset)
        return dataset

    def __getitem__(self, index: Union[int, Sequence, slice]) -> Tuple[pd.Series, Union[LazyDataLoader, np.ndarray]]:
        """
        Can be used to index the data.
        Returns the data frame and data object as a tuple.
        """
        if not isinstance(index, int):
            index = sorted(index)
        return self.data_frame.loc[index], self.data[index]

    def multi_dataset_data(self):
        if self.optical_flow:
            of_frames_list = []
            for dataset in self.datasets:
                of_frames = dataset.data
                of_frames_list.append(of_frames)
            of_frames = np.concatenate(of_frames_list)
            return of_frames

        data_paths = [video_path for dataset in self.datasets for video_path in
                      dataset.data.data_path
        ]
        return LazyDataLoader(data_paths, self.color, self.resize, self.n_sample)

    def __repr__(self):
        return f"{self.dataset_name} dataset with {len(self.data)} samples."


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
        of_frames = sk_resize(of_frames, (n_samples, c, h, w), anti_aliasing=True)
    return of_frames


def validate_apex(df: pd.DataFrame) -> None:
    """
    Validates that the apex is not out of bounds using the dataframe.
    """
    assert (df["apex"] - df["onset"] < 0).sum() == 0, "Apex is lower than onset."
    assert (df["offset"] - df["apex"] < 0).sum() == 0, "Apex is greater than offset."


def validate_onset_offset(df: pd.DataFrame, data: LazyDataLoader) -> None:
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


def validate_n_frames(df: pd.DataFrame, data: LazyDataLoader) -> None:
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


def validate_frames_ascending_order(data: LazyDataLoader) -> None:
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


def validate_dataset(df: pd.DataFrame, data: LazyDataLoader) -> None:
    """
    Performs a set of validation tests to make sure that the dataframe and the dataset are consistent.
    An assertion is thrown if there is an issue.
    """
    validate_apex(df)
    validate_n_frames(df, data)
    validate_onset_offset(df, data)
    validate_frames_ascending_order(data)