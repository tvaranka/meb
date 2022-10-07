import os
import re
from itertools import chain
from typing import List, Tuple, Sequence, Union, Callable
from abc import ABC, abstractmethod
from tqdm import tqdm
from functools import cached_property

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize as sk_resize

from get_image_size import get_image_size
from config.dataset_config import DatasetConfig
import py_evm


PandasIndex = pd.DataFrame | pd.Series

class LazyDataLoader:
    """Loads video data on demand.

    Given paths to videos, provides an iterable over all the videos that
    are loaded lazily.

    Parameters
    ---------
    data_path : 2-dimensional list
        A two-dimensional list where the outer list contains the videos
        and the inner lists contain the images. The list is used to create
        an iterable.
    color : bool, optional, default=False
        When true, videos are loaded with color information, otherwise as grayscale.
        For images with no color information, three channels are returned where
        grayscale is replicated across the channels.
    resize : int or Sequence, optional, default=None
        When None, images are returned with their original size. When int,
        images are resized to (int, int) size. When a size two Sequence is given,
        the height and width are resized accordingly.
    n_sample : int, optional, default=None
        Samples videos to include n_sample number of frames. If None no sampling
        is performed.
    magnify : bool, optional, default=False
        When True, videos are magnified using py_evm. Parameters to py_evm
        can be passed using kwargs with magnify_params.


    Examples
    --------
    >>> from meb.utils import LazyDataLoader
    >>> loader = LazyDataLoader(data_path)
    >>> loader
    LazyDataLoader with 2031 items from ...
    >>> first_video = loader[0]
    >>> first_ten_videos = loader[:10]
    >>> [video.shape for video in loader]

    >>> loader = LazyDataLoader(resize=112, color=True)
    >>> loader[0].shape
    (19, 112, 112, 3)
    >>> loader[[1, 5, 7, 9]]
    LazyDataLoader with 4 items from ...
    """

    def __init__(
        self,
        data_path: List[List[str]],
        color: bool = False,
        resize=None,
        n_sample: int = None,
        magnify: bool = False,
        **kwargs,
    ) -> None:
        self.data_path = data_path
        self.color = color
        self.resize = resize
        self.magnify = magnify
        self.magnify_params = kwargs.pop(
            "magnify_params", {"alpha": 10, "r1": 0.4, "r2": 0.05}
        )
        self.n_sample = n_sample

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, index: int | Sequence | slice | PandasIndex):
        """Get video(s) using index

        Parameters
        ----------
        index : int or Sequence or slice or PandasIndex
            When int, loads the video with the corresponding index and returns it.
            If other a new LazyDataLoader object based on the subset.

        Returns
        -------
        out : ndarray or LazyDataLoader
            A ndarray video based on the index if int or a new LazyDataLoader object
            from subset of the videos if other.
        """
        if isinstance(index, int):
            data_path = self.data_path[index]
            video = self._get_video(data_path)
            if self.magnify:
                return py_evm.magnify(video, **self.magnify_params)
            return video
        elif isinstance(index, Sequence):
            data_path = [self.data_path[i] for i in index]
        elif isinstance(index, slice):
            data_path = self.data_path[index]
        elif isinstance(index, pd.DataFrame) or isinstance(index, pd.Series):
            bool_array = np.array(index).flatten()
            data_path = [path for i, path in enumerate(self.data_path) if bool_array[i]]
        else:
            raise NotImplementedError
        return LazyDataLoader(
            data_path,
            color=self.color,
            resize=self.resize,
            magnify=self.magnify,
            magnify_params=self.magnify_params,
        )

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
        """Creates an empty array with correct size.

        Uses get_image_size function to load video size from metadata without
        loading the video. Resizes if necessary.

        Parameters
        ----------
        image_paths : List[str]
            List of the image paths for the video.

        Returns
        -------
        out : ndarray
            An empty array with the original size of the video or resized.
            uint8 data type.
        """
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
                image = sk_resize(
                    image, (self.resize[0], self.resize[1]), anti_aliasing=True
                )
                image = (image * 255.0).astype("uint8")
            video[f] = image

        return video

    @property
    def shape(self):
        return (len(self),) + self[0].shape


class LoadedDataLoader:
    def __init__(
        self, data_in: Union[LazyDataLoader, List[np.ndarray]], n_sample: int = 6
    ) -> None:
        if isinstance(data_in, LazyDataLoader):
            self.data_loader = data_in
            self.data = self.load_data()
            self.data_path = data_in.data_path
            self.n_sample = data_in.n_sample
        else:
            self.data_path = None
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
        elif isinstance(index, slice):
            data = self.data[index]
        elif isinstance(index, pd.DataFrame) or isinstance(index, pd.Series):
            bool_array = np.array(index).flatten()
            data = [path for i, path in enumerate(self.data) if bool_array[i]]
        else:
            raise NotImplementedError
        new = type(self)(data, self.n_sample)
        return new

    def __setitem__(self, index: int, new_video: np.ndarray):
        if isinstance(index, int):
            self.data[index] = new_video
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        if self.data_path:
            return (
                f"LoadedDataLoader with {len(self)} items from {self.data_path[0][0]}"
            )
        return f"LoadedDataLoader with {len(self)} items"

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
        image_paths = [
            image_path for image_path in image_paths if image_path[-3:] in file_formats
        ]
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
    d = {
        i: int(only_digit(data_path.split("/")[-1]))
        for i, data_path in enumerate(video_path)
    }
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
        magnify: bool = False,
        n_sample: int = 6,
        preload: bool = False,
        ignore_validation: bool = False,
        magnify_params: dict = {},
    ) -> None:
        self.color = color
        self.resize = resize
        self.cropped = cropped
        self.optical_flow = optical_flow
        self.magnify = magnify
        self.magnify_params = magnify_params
        self.n_sample = n_sample
        self.preload = preload
        if not self.optical_flow and self.dataset_name and not ignore_validation:
            validate_dataset(self.data_frame, self.data)

    @cached_property
    @abstractmethod
    def data_frame(self) -> pd.DataFrame:
        """
        Loads the excel file into a Pandas data frame and modifies possible issues
        and generates additional features.
        """

    @cached_property
    def data(self) -> Union[np.ndarray, LazyDataLoader]:
        """
        Loads the dataset given a path. If the optical_flow flag is set to True, a numpy array is returned, else
        RGB frames are returned in the LazyDataLoader object.
        """
        if self.optical_flow:
            return load_optical_flow_data(self.dataset_name, resize=self.resize)
        crop_str = "_cropped" if self.cropped else ""
        dataset_path = getattr(
            DatasetConfig, f"{self.dataset_name}{crop_str}_dataset_path"
        )
        format_path = dataset_path + self.dataset_path_format
        video_paths = get_video_paths(format_path, self.data_frame)

        dataset = LazyDataLoader(
            video_paths,
            color=self.color,
            resize=self.resize,
            n_sample=self.n_sample,
            magnify=self.magnify,
            magnify_params=self.magnify_params,
        )

        if self.preload:
            dataset = LoadedDataLoader(dataset)

        return dataset

    def __getitem__(
        self, index: Union[int, Sequence, slice]
    ) -> Tuple[pd.Series, Union[LazyDataLoader, np.ndarray]]:
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
        data_paths = [
            video_path
            for dataset in self.datasets
            for video_path in dataset.data.data_path
        ]

        if self.preload:
            dataset = LoadedDataLoader(
                list(chain.from_iterable(dataset.data for dataset in self.datasets))
            )
        else:
            dataset = LazyDataLoader(
                data_paths,
                self.color,
                self.resize,
                self.n_sample,
                magnify=self.magnify,
                magnify_params=self.magnify_params,
            )
        return dataset

    def __repr__(self):
        return f"{self.dataset_name} dataset with {len(self.data)} samples."


def only_digit(s: str) -> str:
    """Returns the digits of a string, e.g. AU12R -> 12"""
    return "".join(i for i in s if i.isdigit())


def extract_action_units(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts the action units from a single column to individual columns"""
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


def load_optical_flow_data(
    dataset_name: str, resize: Union[Sequence[int], int, None] = None
):
    if isinstance(resize, int):
        h = w = resize
    elif isinstance(resize, tuple):
        h = resize[0]
        w = resize[1]
    else:
        h = w = 64
    of_path = getattr(DatasetConfig, f"{dataset_name}_optical_flow")
    of_frames = np.load(of_path)
    n_samples, c, _, _ = of_frames.shape
    if resize:
        of_frames = sk_resize(of_frames, (n_samples, c, h, w), anti_aliasing=True)
    return of_frames


def validate_apex(df: pd.DataFrame) -> None:
    """
    Validates that the apex is not out of bounds using the dataframe.
    """
    if (df["apex"] - df["onset"] < 0).sum() != 0:
        print("Warning: Apex is lower than onset.")
    if (df["offset"] - df["apex"] < 0).sum() != 0:
        print("Warning: Apex is greater than offset.")


def validate_onset_offset(df: pd.DataFrame, data: LazyDataLoader) -> None:
    """
    Validates that the onset and offset values correspond in the
    dataset and in the dataframe.
    """
    for i, row in df.iterrows():
        onset_path_last = data.data_path[i][0].split("/")[-1]
        onset_f = int(re.findall("\d+", onset_path_last)[-1])
        if onset_f != row["onset"]:
            print(
                f"Warning: The onset does not correspond for the sample\
            {row['subject']}_{row['material']}"
            )
        offset_path_last = data.data_path[i][-1].split("/")[-1]
        offset_f = int(re.findall("\d+", offset_path_last)[-1])
        if offset_f != row["offset"]:
            print(
                f"Warning: The offset does not correspond for the sample\
            {row['subject']}_{row['material']}"
            )


def validate_n_frames(df: pd.DataFrame, data: LazyDataLoader) -> None:
    """
    Validates whether the number of frames in the dataframe and dataset
    correspond.
    """
    for i, row in df.iterrows():
        n_frames_df = row["n_frames"]
        n_frames_data = len(data.data_path[i])
        if n_frames_df != n_frames_data:
            print(
                f"Warning: \
            The number of frames does not correspond for the sample\
             {row['subject']}_{row['material']}"
            )


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
        if sorted_frame_ns != frame_ns:
            print(
                f"Warning: The frames are not in an ascending order for sample {img_path}"
            )


def validate_dataset(df: pd.DataFrame, data: LazyDataLoader) -> None:
    """
    Performs a set of validation tests to make sure that the dataframe and the dataset are consistent.
    An assertion is thrown if there is an issue.
    """
    validate_apex(df)
    validate_n_frames(df, data)
    validate_onset_offset(df, data)
    validate_frames_ascending_order(data)
