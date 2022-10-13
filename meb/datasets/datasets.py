from typing import Sequence
from . import dataset_utils

from functools import cached_property

import numpy as np
import pandas as pd


class Smic(dataset_utils.Dataset):
    """Smic Dataset

    Loads the Smic (Spontaneous Micro-Expression Database) dataset.

    Reference
    ---------
    :doi:`Li, X., Pfister, T., Huang, X., Zhao, G. and Pietikäinen M. (2013).
    "A Spontaneous Micro-expression Database: Inducement, collection and baseline".
    10th IEEE International Conference and Workshops on Automatic Face and Gesture
    Recognition (FG) pp. 1-6 <10.1109/FG.2013.6553717>`
    """

    __doc__ += dataset_utils.Dataset.__doc__

    def __init__(
        self,
        color: bool = False,
        resize: Sequence[int] | int = None,
        cropped: bool = True,
        optical_flow: bool = False,
        **kwargs,
    ) -> None:
        self.dataset_name = type(self).__name__.lower()
        self.dataset_path_format = "/{subject}/micro/{emotion}/{material}/"
        super().__init__(color, resize, cropped, optical_flow, **kwargs)

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(self.smic_excel_path)
        df = df.drop("Unnamed: 0", axis=1)
        df["n_frames"] = df["offset"] - df["onset"] + 1
        # Set apex at the halfway of onset and offset
        df["apex"] = df["onset"] + ((df["offset"] - df["onset"]) / 2).astype("int")
        return df


class Casme(dataset_utils.Dataset):
    """Casme Dataset

    Loads the Casme (Chinese Academy of Sciences Micro-Expression) dataset.

    Reference
    ---------
    :doi:`Yan, W., Wu, Q., Liu, Y., Wang, S., and Fu, X. (2013).
    "CASME database: A dataset of spontaneous micro-expressions collected from
    neutralized faces". 10th IEEE International Conference and Workshops on
    Automatic Face and Gesture Recognition (FG) pp. 1-7
    <10.1109/FG.2013.6553799>`
    """

    __doc__ += dataset_utils.Dataset.__doc__

    def __init__(
        self,
        color: bool = False,
        resize: Sequence[int] | int = None,
        cropped: bool = True,
        optical_flow: bool = False,
        **kwargs,
    ) -> None:
        self.dataset_name = "casme"
        self.dataset_path_format = "/sub{subject}/{material}/"
        super().__init__(color, resize, cropped, optical_flow, **kwargs)

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(self.casme_excel_path)
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
    """Casme2 Dataset

    Loads the Casme2 (Chinese Academy of Sciences Micro-Expression) dataset.

    Reference
    ---------
    :doi:`Yan, W., Li, X., Wang, S., Zhao, G., Liu, Y., Chen, Y. and Fu, X. (2014).
    "CASME II: An Improved Spontaneous Micro-Expression Database and the Baseline
    Evaluation". PLOS ONE 9(1): e86041 <10.1371/journal.pone.0086041>`
    """

    __doc__ += dataset_utils.Dataset.__doc__

    def __init__(
        self,
        color: bool = False,
        resize: Sequence[int] | int = None,
        cropped: bool = True,
        optical_flow: bool = False,
        **kwargs,
    ) -> None:
        self.dataset_name = "casme2"
        self.dataset_path_format = "/sub{subject}/{material}/"
        super().__init__(color, resize, cropped, optical_flow, **kwargs)

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(self.casme2_excel_path)
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
        samples_missing_apex = [
            29,
            35,
            43,
            45,
            51,
            53,
            60,
            117,
            118,
            126,
            136,
            147,
            155,
            168,
            170,
            177,
            202,
            203,
            234,
            237,
            238,
        ]
        estimated_apexes = [
            279,
            68,
            77,
            81,
            166,
            100,
            78,
            187,
            89,
            80,
            88,
            134,
            231,
            53,
            329,
            111,
            91,
            103,
            98,
            153,
            98,
        ]
        df.loc[samples_missing_apex, "apex"] = estimated_apexes
        df["n_frames"] = df["offset"] - df["onset"] + 1
        df = dataset_utils.extract_action_units(df)
        return df


class SAMM(dataset_utils.Dataset):
    """SAMM Dataset

    Loads the SAMM (Spontaneous micro-movement) dataset.

    Reference
    ---------
    :doi:`Davison, A. K., Lansley, C., Costen, N., Tan, K. and Yap, M. H. (2016).
    "SAMM: A Spontaneous Micro-Facial Movement Dataset". IEEE Transactions on
    Affective Computing, vol. 9, no. 1, pp. 116-129 <10.1109/TAFFC.2016.2573832>`
    """

    __doc__ += dataset_utils.Dataset.__doc__

    def __init__(
        self,
        color: bool = False,
        resize: Sequence[int] | int = None,
        cropped: bool = True,
        optical_flow: bool = False,
        **kwargs,
    ) -> None:
        self.dataset_name = "samm"
        self.dataset_path_format = "/{subject}/{material}/"
        super().__init__(color, resize, cropped, optical_flow, **kwargs)

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(self.samm_excel_path)
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


class Fourd(dataset_utils.Dataset):
    """4DME Dataset

    Loads the 4DME (4-dimensional Micro-Expression) dataset.

    Reference
    ---------
    :doi:`Li, X., Cheng, S., Li, Y., Behzad, M., Shen, J., Zafeiriou, S.,
     Pantic, M. and Zhao, G. (2022).
    "4DME: A Spontaneous 4D Micro-Expression Dataset With Multimodalities". IEEE
    Transactions on Affective Computing, <10.1109/TAFFC.2022.3182342>`
    """

    __doc__ += dataset_utils.Dataset.__doc__

    def __init__(
        self,
        color: bool = False,
        resize: Sequence[int] | int = None,
        cropped: bool = True,
        optical_flow: bool = False,
        **kwargs,
    ) -> None:
        self.dataset_name = "fourd"
        self.dataset_path_format = "/{subject}/{material}/"
        super().__init__(color, resize, cropped, optical_flow, **kwargs)

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(self.fourd_excel_path)
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
    """MMEW Dataset

    Loads the MMEW (Micro- and Macro-Expression Warehouse) dataset.

    Reference
    ---------
    :doi:`Ben, X., Ren, Y., Zhang, J., Wang, S., Kpalma, K., Meng, W. and Liu, Y.
    (2021). "Video-Based Facial Micro-Expression Analysis: A Survey of Datasets,
    Features and Algorithms". IEEE Transactions on Pattern Analysis and Machine
    Intelligence, vol. 44, no. 9, pp. 5826-5846, <10.1109/TPAMI.2021.3067464>`
    """

    __doc__ += dataset_utils.Dataset.__doc__

    def __init__(
        self,
        color: bool = False,
        resize: Sequence[int] | int = None,
        cropped: bool = True,
        optical_flow: bool = False,
        **kwargs,
    ) -> None:
        self.dataset_name = "mmew"
        self.dataset_path_format = "/{emotion}/{material}/"
        super().__init__(color, resize, cropped, optical_flow, **kwargs)

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(self.mmew_excel_path)
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
        samples_missing_apex = [
            33,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            142,
            143,
            145,
            146,
            148,
            154,
            156,
        ]
        estimated_apexes = [
            40,
            15,
            46,
            20,
            25,
            20,
            23,
            28,
            32,
            56,
            32,
            27,
            22,
            29,
            31,
            38,
        ]
        estimated_offsets = [
            75,
            54,
            59,
            98,
            80,
            93,
            58,
            61,
            68,
            88,
            80,
            85,
            61,
            108,
            72,
            106,
        ]
        df.loc[samples_missing_apex, ["apex", "offset"]] = np.array(
            [estimated_apexes, estimated_offsets]
        ).T
        df = dataset_utils.extract_action_units(df)
        df = df.replace({"others": "repression"})
        df["n_frames"] = df["offset"] - df["onset"] + 1
        return df


class Casme3A(dataset_utils.Dataset):
    """CASME3A Dataset

    Loads the part A of CAS(ME)^3 dataset.

    Reference
    ---------
    :doi:`Li, J., Dong, Z., Lu, S., Wang, S., Yan, W., Ma, Y., Liu, Y., Huang, C.
    and Fu, X. (2022). "CAS(ME)3: A Third Generation Facial Spontaneous
    Micro-Expression Database with Depth Information and High Ecological Validity".
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    <10.1109/TPAMI.2022.3174895>`
    """

    __doc__ += dataset_utils.Dataset.__doc__

    def __init__(
        self,
        color: bool = False,
        resize: Sequence[int] | int = None,
        cropped: bool = True,
        optical_flow: bool = False,
        **kwargs,
    ) -> None:
        self.dataset_name = type(self).__name__.lower()
        self.dataset_path_format = "/{subject}/{material}/color/"
        super().__init__(color, resize, cropped, optical_flow, **kwargs)

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(self.casme3a_excel_path)
        df = df.rename(
            {
                "Subject": "subject",
                "Filename": "material",
                "Onset": "onset",
                "Apex": "apex",
                "Offset": "offset",
            },
            axis=1,
        )
        df = self._separate_duplicate_materials(df)
        df.loc[128, "apex"] = 160
        df.loc[749, "onset"] = 2647
        df.loc[[708, 796, 798], "offset"] = [34, 1845, 2403]
        df.loc[750, ["onset", "apex"]] = 1
        df["n_frames"] = df["offset"] - df["onset"] + 1
        df = dataset_utils.extract_action_units(df)

        return df

    @staticmethod
    def _separate_duplicate_materials(df):
        def remove_nums(string: str) -> str:
            return "".join([s for s in string if not s.isdigit()])

        # For subjects with the same filename add number to seperate them
        num = 2
        for i in range(df.shape[0] - 1):
            # check if this and next are the same or not
            if remove_nums(df.loc[i, "material"]) == df.loc[i + 1, "material"]:
                # Check if subject is the same
                if df.loc[i, "subject"] == df.loc[i + 1, "subject"]:
                    df.loc[i + 1, "material"] = df.loc[i + 1, "material"] + str(num)
                    num += 1
                else:
                    num = 2
            else:
                num = 2
        return df


class Casme3C(dataset_utils.Dataset):
    """CASME3C Dataset

    Loads the part C of CAS(ME)^3 dataset.

    Reference
    ---------
    :doi:`Li, J., Dong, Z., Lu, S., Wang, S., Yan, W., Ma, Y., Liu, Y., Huang, C.
    and Fu, X. (2022). "CAS(ME)3: A Third Generation Facial Spontaneous
    Micro-Expression Database with Depth Information and High Ecological Validity".
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    <10.1109/TPAMI.2022.3174895>`
    """

    __doc__ += dataset_utils.Dataset.__doc__

    def __init__(
        self,
        color: bool = False,
        resize: Sequence[int] | int = None,
        cropped: bool = True,
        optical_flow: bool = False,
        **kwargs,
    ) -> None:
        self.dataset_name = type(self).__name__.lower()
        self.dataset_path_format = "/{subject}/{material}/color/"
        super().__init__(color, resize, cropped, optical_flow, **kwargs)

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        df = pd.read_excel(self.casme3c_excel_path)
        df = df.rename({"sub": "subject", "count": "material", "au": "AU"}, axis=1)
        # replace chinese comma with standard one for sample 16
        df.loc[16, "AU"] = df.loc[16, "AU"].replace("，", ",")
        df["AU"] = df["AU"].apply(lambda x: str(x).replace(",", "+"))
        df["subject"] = df["subject"].apply(lambda x: str(x).zfill(2))
        df["apex"] = ((df["offset"] + df["onset"]) / 2).astype("int")
        df["n_frames"] = df["offset"] - df["onset"] + 1
        df = dataset_utils.extract_action_units(df)
        return df


class Casme3(dataset_utils.Dataset):
    """CASME3 Dataset

    Combines part A and C together.

    Reference
    ---------
    :doi:`Li, J., Dong, Z., Lu, S., Wang, S., Yan, W., Ma, Y., Liu, Y., Huang, C.
    and Fu, X. (2022). "CAS(ME)3: A Third Generation Facial Spontaneous
    Micro-Expression Database with Depth Information and High Ecological Validity".
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    <10.1109/TPAMI.2022.3174895>`
    """

    __doc__ += dataset_utils.Dataset.__doc__

    def __init__(
        self,
        color: bool = False,
        resize: Sequence[int] | int = None,
        cropped: bool = True,
        optical_flow: bool = False,
        **kwargs,
    ) -> None:
        self.dataset_name = ""
        self.dataset_path_format = ""
        self.datasets = [
            dataset(
                cropped=cropped,
                color=color,
                resize=resize,
                optical_flow=optical_flow,
                **kwargs,
            )
            for dataset in [Casme3A, Casme3C]
        ]
        super().__init__(color, resize, cropped, optical_flow, **kwargs)

    @property
    def data(
        self,
    ) -> np.ndarray | dataset_utils.LazyDataLoader | dataset_utils.LoadedDataLoader:
        return self.multi_dataset_data()

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        dataset_dfs = []
        for dataset in self.datasets:
            df = dataset.data_frame
            df.insert(9, "dataset", dataset.dataset_name)
            dataset_dfs.append(df)
        df = pd.concat(dataset_dfs, axis=0, sort=False)
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


class CrossDataset(dataset_utils.Dataset):
    """Cross-Dataset

    Combines CASME, CASME2, SAMM, 4DME, MMEW and CASME3A together
    """

    __doc__ += dataset_utils.Dataset.__doc__

    def __init__(
        self,
        color: bool = False,
        resize: Sequence[int] | int = None,
        cropped: bool = True,
        optical_flow: bool = False,
        **kwargs,
    ) -> None:
        self.dataset_name = ""
        self.dataset_path_format = ""
        self.datasets = [
            dataset(
                cropped=cropped,
                color=color,
                resize=resize,
                optical_flow=optical_flow,
                **kwargs,
            )
            for dataset in [Casme, Casme2, SAMM, Fourd, MMEW, Casme3A]
        ]
        super().__init__(color, resize, cropped, optical_flow, **kwargs)

    @property
    def data(
        self,
    ) -> np.ndarray | dataset_utils.LazyDataLoader | dataset_utils.LoadedDataLoader:
        return self.multi_dataset_data()

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        cross_dataset_dfs = []
        for dataset in self.datasets:
            df = dataset.data_frame
            if "dataset" not in df:
                df.insert(11, "dataset", dataset.dataset_name)
            cross_dataset_dfs.append(df)
        df = pd.concat(cross_dataset_dfs, axis=0, sort=False)
        df = df.drop(
            [
                "ApexF2",
                "index",
                "Inducement Code",
                "Duration",
                "Micro",
                "Objective Classes",
                "Notes",
                "fold",
                "eye blink",
                "Positive",
                "Negative",
                "Surprise",
                "Repression",
                "Others",
                "Onset",
                "Total",
                "apexf",
                "Objective class",
            ],
            axis=1,
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


class MEGC(dataset_utils.Dataset):
    """CASME3 Dataset

    Combines part A and C together.

    Reference
    ---------
    :doi:`See, J., Yap, M. H., Li, J., Hong, X. and Wang, S. (2019).
    "MEGC 2019 – The Second Facial Micro-Expressions Grand Challenge".
    14th IEEE International Conference on Automatic Face & Gesture Recognition
    (FG 2019), 2019, pp. 1-5, <10.1109/FG.2019.8756611>`
    """

    __doc__ += dataset_utils.Dataset.__doc__

    def __init__(
        self,
        color: bool = False,
        resize: Sequence[int] | int = None,
        cropped: bool = True,
        optical_flow: bool = False,
        **kwargs,
    ) -> None:
        self.dataset_name = ""
        self.dataset_path_format = ""
        self.datasets = [
            dataset(
                cropped=cropped,
                color=color,
                resize=resize,
                optical_flow=optical_flow,
                **kwargs,
            )
            for dataset in [Smic, Casme2, SAMM]
        ]
        super().__init__(color, resize, cropped, optical_flow, **kwargs)

    @property
    def data(
        self,
    ) -> np.ndarray | dataset_utils.LazyDataLoader | dataset_utils.LoadedDataLoader:
        all_data = self.multi_dataset_data()
        _, indices_to_remove = self.get_data_frame_and_indices()
        indices_to_keep = [
            i for i in range(len(all_data)) if i not in indices_to_remove
        ]
        return all_data[indices_to_keep]

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        df, _ = self.get_data_frame_and_indices()
        return df

    def get_data_frame_and_indices(self):
        dataset_dfs = []
        for dataset in self.datasets:
            df = dataset.data_frame
            if "dataset" not in df:
                df.insert(5, "dataset", dataset.dataset_name)
            dataset_dfs.append(df)
        df = pd.concat(dataset_dfs, axis=0, sort=False)
        df = df.reset_index().drop("index", axis=1)
        # Add apex to smic as its missing, use half frame
        df.loc[df.index[df["apex"].isnull()], "apex"] = (
            (df[df["apex"].isnull()]["offset"] - df[df["apex"].isnull()]["onset"]) / 2
        ).astype(int)
        # Add frame and apex information
        df["n_frames"] = df["n_frames"].astype(int)
        df.insert(5, "apexf", df["apex"] - df["onset"])
        # Drop AUs and other information from the dataframe
        aus = df.columns[["AU" in column for column in df.columns]]
        df = df.drop(aus, axis=1)
        df = df.drop(
            [
                "Inducement Code",
                "Duration",
                "Micro",
                "Objective Classes",
                "Notes",
                "level_0",
            ],
            axis=1,
        )
        # Set correct emotions for megc
        df.loc[
            df["emotion"].isin(
                [
                    "Anger",
                    "Contempt",
                    "Disgust",
                    "Sadness",
                    "Fear",
                    "disgust",
                    "repression",
                ]
            ),
            "emotion",
        ] = "negative"
        df.loc[df["emotion"].isin(["Happiness", "happiness"]), "emotion"] = "positive"
        df.loc[df["emotion"] == "Surprise", "emotion"] = "surprise"
        # Remove samples that do not belong to megc
        indices_to_remove = df[
            df["emotion"].isin(["fear", "sadness", "others", "Other"])
        ]["emotion"].index.tolist()
        indices_to_remove.extend(
            df[(df["subject"] == "04") & (df["material"] == "EP12_01f")].index.tolist()
        )
        indices_to_remove.extend(
            df[(df["subject"] == "05") & (df["material"] == "EP12_03f")].index.tolist()
        )
        indices_to_remove.extend(
            df[(df["subject"] == "24") & (df["material"] == "EP02_07")].index.tolist()
        )
        df = df.drop(indices_to_remove).reset_index(drop=True)
        return df, indices_to_remove
