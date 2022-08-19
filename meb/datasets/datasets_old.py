import os
from itertools import chain
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from skimage.transform import resize


def add_zeros(num):
    num = str(num)
    if len(num) == 3:
        return num
    elif len(num) == 2:
        return "0" + num
    else:
        return "00" + num


def smic(
    df_path="../data/smic.xlsx",
    dataset_path="../../../../imag/development/Micro/data/[standard_datasets]/SMIC/SMIC_all_cropped/HS",
):
    """Returns the dataframe containing the metadata for the videos and
    a generator object containing the videos."""
    df = pd.read_excel(df_path)
    df["n_frames"] = df["offset"] - df["onset"] + 1
    root_path = dataset_path
    n_samples = df.shape[0]

    def load_smic():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]

            video = np.zeros((170, 140, n_frames), dtype="uint8")
            for k, j in enumerate(range(onset, onset + n_frames)):
                try:
                    img_path = "{}/{}/micro/{}/{}/reg_image{}.bmp".format(
                        root_path, subject, emotion, material, str(j).zfill(6)
                    )
                    img = plt.imread(img_path)
                except FileNotFoundError:
                    img_path = "{}/{}/micro/{}/{}/reg_image{}.bmp".format(
                        root_path, subject, emotion, material, str(j - 1).zfill(6)
                    )
                    img = plt.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = resize(img, (170, 140))
                img = np.round(img * 255).astype("uint8")
                video[..., k] = img
            yield video

    return df, load_smic()


def smic_raw(
    df_path="../data/smic.xlsx",
    dataset_path="../../../../cmv/development/Databases/SMIC database/Orig data/HS",
    color=False,
):
    """Returns the dataframe containing the metadata for the videos and
    a generator object containing the videos."""
    df = pd.read_excel(df_path)
    df["n_frames"] = df["offset"] - df["onset"] + 1
    root_path = dataset_path
    n_samples = df.shape[0]

    def load_smic():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]

            video = np.zeros((480, 640, n_frames), dtype="uint8")
            if color:
                video = np.zeros((480, 640, 3, n_frames), dtype="uint8")
            for k, j in enumerate(range(onset, onset + n_frames)):
                try:
                    img_path = "{}/{}/micro/{}/{}/image{}.bmp".format(
                        root_path, subject, emotion, material, str(j).zfill(6)
                    )
                    img = plt.imread(img_path)
                except FileNotFoundError:
                    img_path = "{}/{}/micro/{}/{}/image{}.bmp".format(
                        root_path, subject, emotion, material, str(j - 1).zfill(6)
                    )
                    img = plt.imread(img_path)
                if not color:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # img = np.round(img * 255).astype("uint8")
                video[..., k] = img
            yield video

    return df, load_smic()


def casme(
    df_path="../data/casme.xlsx",
    dataset_path="../../../../imag/development/Micro/data/[standard_datasets]/CASME/Cropped",
):
    # Casme with 189 samples
    df = pd.read_excel(df_path)
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
    df["subject"] = df["subject"].apply(
        lambda x: str(x) if x >= 10 else "0{}".format(x)
    )
    df.iloc[40, 2] = 108  # Mistake in file
    df.iloc[40, 5] = 149  # Mistake in file
    df.iloc[42, 2] = 101  # Mistake in file
    df.iloc[42, 5] = 119  # Mistake in file
    df.iloc[43, 2] = 57  # Mistake in file
    df.iloc[43, 5] = 74  # Mistake in file
    df.iloc[54, 5] = 40
    df["n_frames"] = df["offset"] - df["onset"] + 1

    n_samples = df.shape[0]

    root_path = dataset_path

    def load_casme():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]

            video = np.zeros((170, 140, n_frames))
            for k, j in enumerate(range(onset, onset + n_frames)):
                if i >= 92:
                    j = add_zeros(j)
                img_path = "{}/sub{}/{}/reg_{}-{}.jpg".format(
                    root_path, subject, material, material, j
                )
                img = plt.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = resize(img, (170, 140))
                video[..., k] = img
            yield video

    return df, load_casme()


def casme_raw(
    df_path="data/casme.xlsx", dataset_path="../../Micro expressions/CASME_raw_selected"
):
    # Casme with 189 samples
    df = pd.read_excel(df_path)
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
    df["subject"] = df["subject"].apply(
        lambda x: str(x) if x >= 10 else "0{}".format(x)
    )
    df.iloc[40, 2] = 108  # Mistake in file
    df.iloc[40, 5] = 149  # Mistake in file
    df.iloc[42, 2] = 101  # Mistake in file
    df.iloc[42, 5] = 119  # Mistake in file
    df.iloc[43, 2] = 57  # Mistake in file
    df.iloc[43, 5] = 74  # Mistake in file
    df.iloc[54, 5] = 40
    df["n_frames"] = df["offset"] - df["onset"] + 1

    n_samples = df.shape[0]

    root_path = dataset_path

    def load_casme():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]

            video = np.zeros((170, 140, n_frames), dtype="uint8")
            for k, j in enumerate(range(onset, onset + n_frames)):
                if i >= 92:
                    j = add_zeros(j)
                img_path = "{}/sub{}/{}/{}-{}.jpg".format(
                    root_path, subject, material, material, j
                )
                img = plt.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = resize(img, (170, 140))
                video[..., k] = img
            yield video

    return df, load_casme()


def casme2(
    df_path="../data/CASME2-coding-updated.xlsx",
    dataset_path="../../../../imag/public/casme2/Cropped_original",
    full=False,
    color=False,
    full_size=False,
):
    # Casme 2 path with 256/247 samples, length(fear + sadness) = 9, removed
    df = pd.read_excel(df_path)
    df = df.drop(["Unnamed: 2", "Unnamed: 6"], axis=1)
    df = df.rename(
        columns={
            "Estimated Emotion": "emotion",
            "Subject": "subject",
            "Filename": "material",
            "OnsetFrame": "onset",
            "ApexFrame": "apex",
            "OffsetFrame": "offset",
        }
    )
    df.iloc[60, 4] = 91  # Mistake in file, change offset to 91
    df["n_frames"] = df["offset"] - df["onset"] + 1
    df["subject"] = df["subject"].apply(
        lambda x: str(x) if x >= 10 else "0{}".format(x)
    )
    if not full:
        df = df[~df["emotion"].isin(["fear", "sadness"])]
        df = df.reset_index()
    if full:
        df = df.drop(222).reset_index()

    # missing apex, changed based on looking at OF
    if full:
        df.iloc[45, 4] = 81
        df.iloc[29, 4] = 279
        df.iloc[35, 4] = 68
        df.iloc[43, 4] = 77
        df.iloc[51, 4] = 166
        df.iloc[53, 4] = 100
        df.iloc[60, 4] = 78
        df.iloc[117, 4] = 187
        df.iloc[118, 4] = 89
        df.iloc[126, 4] = 80
        df.iloc[136, 4] = 88
        df.iloc[147, 4] = 134
        df.iloc[155, 4] = 231
        df.iloc[168, 4] = 53
        df.iloc[170, 4] = 329
        df.iloc[177, 4] = 111
        df.iloc[202, 4] = 91
        df.iloc[203, 4] = 103
        df.iloc[233, 4] = 98
        df.iloc[236, 4] = 153
        df.iloc[237, 4] = 98
    else:
        df.iloc[29, 4] = 279
        df.iloc[35, 4] = 68
        df.iloc[43, 4] = 77
        df.iloc[45, 4] = 82
        df.iloc[51, 4] = 166
        df.iloc[53, 4] = 100
        df.iloc[60, 4] = 78
        df.iloc[115, 4] = 187
        df.iloc[116, 4] = 89
        df.iloc[124, 4] = 80
        df.iloc[134, 4] = 88
        df.iloc[145, 4] = 134
        df.iloc[153, 4] = 231
        df.iloc[166, 4] = 53
        df.iloc[173, 4] = 111
        df.iloc[197, 4] = 91
        df.iloc[198, 4] = 103
        df.iloc[226, 4] = 98
        df.iloc[229, 4] = 153
        df.iloc[230, 4] = 98

    n_samples = df.shape[0]

    root_path = dataset_path

    def load_casme2():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]

            w, h, = (
                170,
                140,
            )
            video = np.zeros((w, h, n_frames), dtype="uint8")
            if full_size:
                img_path = "{}/sub{}/{}/reg_img{}.jpg".format(
                    root_path, subject, material, onset
                )
                img = plt.imread(img_path)
                w, h, c = img.shape
                video = np.zeros((w, h, c, n_frames), dtype="uint8")
            if color:
                video = np.zeros((w, h, 3, n_frames), dtype="uint8")
            for k, j in enumerate(range(onset, onset + n_frames)):
                img_path = "{}/sub{}/{}/reg_img{}.jpg".format(
                    root_path, subject, material, j
                )
                img = plt.imread(img_path)
                if not color:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = resize(img, (w, h), anti_aliasing=True)
                else:
                    img = resize(img, (w, h, 3), anti_aliasing=True)
                img = np.round(img * 255).astype("uint8")
                video[..., k] = img
            yield video

    return df, load_casme2()


def casme2_raw(
    df_path="../data/CASME2-coding-updated.xlsx",
    dataset_path="../../../../imag/development/Micro/data/[standard_datasets]/CASME2/CASME2_RAW_selected.zip",
    color=False,
    full=False,
):
    # Casme 2 path with 256/247 samples, length(fear + sadness) = 9, removed
    df = pd.read_excel(df_path)
    df = df.drop(["Unnamed: 2", "Unnamed: 6"], axis=1)
    df = df.rename(
        columns={
            "Estimated Emotion": "emotion",
            "Subject": "subject",
            "Filename": "material",
            "OnsetFrame": "onset",
            "ApexFrame": "apex",
            "OffsetFrame": "offset",
        }
    )
    df.iloc[60, 4] = 91  # Mistake in file, change offset to 91
    df["n_frames"] = df["offset"] - df["onset"] + 1
    df["subject"] = df["subject"].apply(
        lambda x: str(x) if x >= 10 else "0{}".format(x)
    )
    if not full:
        df = df[~df["emotion"].isin(["fear", "sadness"])]
        df = df.reset_index()
    if full:
        df = df.drop(222).reset_index()

    # missing apex, changed based on looking at OF
    if full:
        df.iloc[45, 4] = 81
        df.iloc[29, 4] = 279
        df.iloc[35, 4] = 68
        df.iloc[43, 4] = 77
        df.iloc[51, 4] = 166
        df.iloc[53, 4] = 100
        df.iloc[60, 4] = 78
        df.iloc[117, 4] = 187
        df.iloc[118, 4] = 89
        df.iloc[126, 4] = 80
        df.iloc[136, 4] = 88
        df.iloc[147, 4] = 134
        df.iloc[155, 4] = 231
        df.iloc[168, 4] = 53
        df.iloc[170, 4] = 329
        df.iloc[177, 4] = 111
        df.iloc[202, 4] = 91
        df.iloc[203, 4] = 103
        df.iloc[233, 4] = 98
        df.iloc[236, 4] = 153
        df.iloc[237, 4] = 98
    else:
        df.iloc[29, 4] = 279
        df.iloc[35, 4] = 68
        df.iloc[43, 4] = 77
        df.iloc[45, 4] = 82
        df.iloc[51, 4] = 166
        df.iloc[53, 4] = 100
        df.iloc[60, 4] = 78
        df.iloc[115, 4] = 187
        df.iloc[116, 4] = 89
        df.iloc[124, 4] = 80
        df.iloc[134, 4] = 88
        df.iloc[145, 4] = 134
        df.iloc[153, 4] = 231
        df.iloc[166, 4] = 53
        df.iloc[173, 4] = 111
        df.iloc[197, 4] = 91
        df.iloc[198, 4] = 103
        df.iloc[226, 4] = 98
        df.iloc[229, 4] = 153
        df.iloc[230, 4] = 98

    n_samples = df.shape[0]

    root_path = dataset_path
    archive = zipfile.ZipFile(root_path, "r")

    def load_casme2():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]

            video = np.zeros((480, 640, n_frames), dtype="uint8")
            if color:
                video = np.zeros((480, 640, 3, n_frames), dtype="uint8")
            for k, j in enumerate(range(onset, onset + n_frames)):
                img_path = "CASME2_RAW_selected/sub{}/{}/img{}.jpg".format(
                    subject, material, j
                )
                img = plt.imread(archive.open(img_path))
                if not color:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                video[..., k] = img
            yield video

    return df, load_casme2()


def samm(
    df_path="../data/SAMM_Micro_FACS_Codes_v2.xlsx",
    dataset_path="../../ytli/dataset/SAMM/SAMM_CROP",
    full_size=False,
):
    # SAMM with 159 samples
    df = pd.read_excel(df_path)
    # preprocess the dataframe as it contains some text
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
        }
    )
    df.iloc[56, 6] = 5793  # mistake in file
    # df.loc[56, "offset"] = 5739 #mistake in file
    df.loc[125, "apex"] = 1105  # mistake in file, set arbitrarily
    df.loc[132, "apex"] = 4945
    df.loc[133, "apex"] = 5130
    df["n_frames"] = df["offset"] - df["onset"] + 1
    root_path = dataset_path
    n_samples = df.shape[0]

    def load_samm():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]

            files = list(os.walk("{}/{}/{}".format(root_path, subject, material)))[0][2]
            w, h, = (
                170,
                140,
            )
            video = np.zeros((w, h, n_frames), dtype="uint8")
            if full_size:
                img_path = "{}/{}/{}/{}".format(root_path, subject, material, files[0])
                img = plt.imread(img_path)
                w, h = img.shape
                video = np.zeros((w, h, n_frames), dtype="uint8")
            for k, f in enumerate(files):
                img_path = "{}/{}/{}/{}".format(root_path, subject, material, f)
                img = plt.imread(img_path)
                if not full_size:
                    img = resize(img, (170, 140))
                video[..., k] = img
            yield video

    return df, load_samm()


def samm_raw(
    df_path="../data/SAMM_Micro_FACS_Codes_v2.xlsx",
    dataset_path="../../ytli/dataset/SAMM/SAMM",
    color=False,
):
    # SAMM with 159 samples
    df = pd.read_excel(df_path)
    # preprocess the dataframe as it contains some text
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
        }
    )
    # df.iloc[56, 6] = 5793
    df.loc[125, "apex"] = 1105  # mistake in file, set arbitrarily
    df.loc[132, "apex"] = 4945
    df.loc[133, "apex"] = 5130
    df.loc[56, "offset"] = 5739  # mistake in file
    df["n_frames"] = df["offset"] - df["onset"] + 1
    root_path = dataset_path
    n_samples = df.shape[0]

    def load_samm():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]

            files = list(os.walk("{}/{}/{}".format(root_path, subject, material)))[0][2]
            video = np.zeros((650, 960, n_frames), dtype="uint8")
            if color:
                video = np.zeros((650, 960, 3, n_frames), dtype="uint8")
            for k, f in enumerate(files):
                img_path = "{}/{}/{}/{}".format(root_path, subject, material, f)
                img = plt.imread(img_path)
                if color:
                    img = np.array([img, img, img]).transpose(1, 2, 0)
                video[..., k] = img
            yield video

    return df, load_samm()


def fourDmicro(
    df_path="../../Xiaobai/4Dmicro/4DME_Labelling_Micro-expression_emotion_eyeblink_final_remove_unsuefull_codes.xlsx",
    dataset_path="../../Xiaobai/4Dmicro/gray_video_ytli/gray_micro_register_final/",
):
    root_path = dataset_path
    subjects = sorted(os.listdir(root_path))
    subjects.remove("S13_1st_001_01_1")

    df = pd.read_excel(df_path)
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
        }
    )
    df.loc[df["self-report emotion"] == 1, "self-report emotion"] = "1'"
    df.loc[df["self-report emotion"] == 7, "self-report emotion"] = "7'"
    df.loc[df["self-report emotion"] == 2, "self-report emotion"] = "2'"
    df.loc[df["self-report emotion"] == 1.3, "self-report emotion"] = "1.3'"
    df.loc[df["self-report emotion"] == 1.7, "self-report emotion"] = "1.7'"
    # df = df.drop(["ME_number'", "subME_number'", "first_frame'", "last_frame'"], axis=1)
    df["onset"] = df["onset"].astype("int")
    df["apex"] = df["apex"].astype("int")
    df["offset"] = df["offset"].astype("int")
    df.loc[26, "apex"] = 2490
    df["apexf"] = df["apex"] - df["onset"]
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
    df["subME_number'"] = df["subME_number'"].astype("str")
    df.loc[263, "emotion"] = "Surprise+Positive"
    tmp = df.loc[145]
    df.loc[145] = df.loc[144]
    df.loc[144] = tmp
    df.loc[245, "n_frames"] = 24

    # Remove static(marked as (k) in the file) AUs
    df.loc[77, "AUs"] = df.loc[77, "AUs"][:-1]
    df.loc[124, "AUs"] = "AU4(k)+L7"
    df.loc[171, "AUs"] = "AUR2+12(k)+45"

    for i in range(267):
        a = [au[:-3] for au in df.loc[i, "AUs"].split("+") if "(k)" in au]
        a = [au[2:] if "AU" in au else au for au in a]
        for au in a:
            df.loc[i, "AU{}".format(au)] = 0

    def load_4Dmicro():
        for i, subject in enumerate(subjects):
            frames = sorted(os.listdir(root_path + "/" + subject))
            n_frames = int(df["n_frames"][i])
            # load first image to know the size
            img = plt.imread(root_path + "/" + subject + "/" + frames[0])
            video = np.zeros((170, 140, n_frames), dtype="uint8")
            for j, frame in enumerate(frames):
                img = plt.imread(root_path + "/" + subject + "/" + frame)
                img = resize(img, (170, 140))
                img = np.round(img * 255).astype("uint8")
                video[..., j] = img
            yield video

    return df, load_4Dmicro()


def fourDmicro_raw(
    df_path="",
    dataset_path="../Xiaobai/4Dmicro/gray_video_ytli/gray_micro_register_final/",
):
    root_path = dataset_path
    subjects = sorted(os.listdir(root_path))
    subjects.remove("S13_1st_001_01_1")

    df = pd.read_excel(
        "../Xiaobai/4Dmicro/4DME_Labelling_Micro-expression_emotion_eyeblink_final_remove_unsuefull_codes.xlsx"
    )
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
        }
    )
    df.loc[df["self-report emotion"] == 1, "self-report emotion"] = "1'"
    df.loc[df["self-report emotion"] == 7, "self-report emotion"] = "7'"
    df.loc[df["self-report emotion"] == 2, "self-report emotion"] = "2'"
    df.loc[df["self-report emotion"] == 1.3, "self-report emotion"] = "1.3'"
    df.loc[df["self-report emotion"] == 1.7, "self-report emotion"] = "1.7'"
    # df = df.drop(["ME_number'", "subME_number'", "first_frame'", "last_frame'"], axis=1)
    df["onset"] = df["onset"].astype("int")
    df["apex"] = df["apex"].astype("int")
    df["offset"] = df["offset"].astype("int")
    df.loc[26, "apex"] = 2490
    df["apexf"] = df["apex"] - df["onset"]
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
    df["subME_number'"] = df["subME_number'"].astype("str")
    df.loc[263, "emotion"] = "Surprise+Positive"
    tmp = df.loc[145]
    df.loc[145] = df.loc[144]
    df.loc[144] = tmp
    df.loc[245, "n_frames"] = 24

    # Remove static(marked as (k) in the file) AUs
    df.loc[77, "AUs"] = df.loc[77, "AUs"][:-1]
    df.loc[124, "AUs"] = "AU4(k)+L7"
    df.loc[171, "AUs"] = "AUR2+12(k)+45"

    for i in range(267):
        a = [au[:-3] for au in df.loc[i, "AUs"].split("+") if "(k)" in au]
        a = [au[2:] if "AU" in au else au for au in a]
        for au in a:
            df.loc[i, "AU{}".format(au)] = 0

    def load_4Dmicro():
        for i, subject in enumerate(subjects):
            frames = sorted(os.listdir(root_path + "/" + subject))
            # load first image to know the size
            img = plt.imread(root_path + "/" + subject + "/" + frames[0])
            video = np.zeros((img.shape[0], img.shape[1], len(frames)), dtype="uint8")
            for j, frame in enumerate(frames):
                img = plt.imread(root_path + "/" + subject + "/" + frame)
                video[..., j] = img
            yield video

    return df, load_4Dmicro()


def only_digit(s):
    return "".join(i for i in s if i.isdigit())


def extract_action_units(df):
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


def mmew(
    df_path="../../../../imag/development/Micro/data/[standard_datasets]/MMEW/MMEW_Micro_Exp.xlsx",
    dataset_path="../../../../imag/development/Micro/data/[standard_datasets]/MMEW/MMEW/Micro_Expression",
    color=False,
):
    df = pd.read_excel(df_path)
    df = df.drop("remarks", axis=1)
    df = df.rename(columns={"Action Units": "AU", "Subject": "subject"})
    df = extract_action_units(df)
    df = df.replace({"others": "repression"})

    def load_mmew():
        for i, sample_df in df.iterrows():
            emotion = sample_df["Estimated Emotion"]
            filename = sample_df["Filename"]
            img_path_root = os.path.join(dataset_path, emotion, filename)
            offset_frame = sample_df["OffsetFrame"]
            if color:
                video = np.zeros((231, 231, 3, offset_frame), dtype="uint8")
            else:
                video = np.zeros((231, 231, offset_frame), dtype="uint8")
            for f in range(1, offset_frame):
                img_path = img_path_root + "/" + str(f) + ".jpg"
                img = plt.imread(img_path)
                if not color:
                    img = img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                video[..., f - 1] = img

            yield video

    return df, load_mmew()


def megc(dataset="raw", color=False):
    if dataset == "raw":
        df_samm, load_samm = samm_raw(color=color)
        df_casme2, load_casme2 = casme2_raw(color=color)
        df_smic, load_smic = smic_raw(color=color)
    else:
        df_samm, load_samm = samm()
        df_casme2, load_casme2 = casme2()
        df_smic, load_smic = smic()
    # Remove "others" from casme2
    indices = df_casme2[df_casme2["emotion"] != "others"]["emotion"].index.tolist()
    load_casme2 = (video for i, video in enumerate(load_casme2) if i in indices)
    df_casme2 = df_casme2.loc[df_casme2["emotion"] != "others"]
    # Set the correct emotions
    df_casme2.loc[
        df_casme2["emotion"].isin(["disgust", "repression"]), "emotion"
    ] = "negative"
    df_casme2.loc[df_casme2["emotion"] == "happiness", "emotion"] = "positive"

    # remove "others" from samm
    indices2 = df_samm[df_samm["emotion"] != "Other"]["emotion"].index.tolist()
    load_samm = (video for i, video in enumerate(load_samm) if i in indices2)
    df_samm = df_samm[df_samm["emotion"] != "Other"]
    # Set the correct emotions
    df_samm.loc[
        df_samm["emotion"].isin(["Anger", "Contempt", "Disgust", "Sadness", "Fear"]),
        "emotion",
    ] = "negative"
    df_samm.loc[df_samm["emotion"] == "Happiness", "emotion"] = "positive"
    df_samm.loc[df_samm["emotion"] == "Surprise", "emotion"] = "surprise"

    # merge dataframes and iterators
    df_smic = df_smic.rename(columns={"Unnamed: 0": "index"})
    df = pd.concat([df_casme2, df_smic, df_samm], sort=True)
    df = df.reset_index()
    df = df.drop(
        ["Duration", "Inducement Code", "Micro", "Notes", "Objective Classes"], axis=1
    )

    # add column for dataset information
    df["dataset"] = "casme2"
    df.loc[148:312, "dataset"] = "smic"
    df.loc[312:, "dataset"] = "samm"

    # apex is the apex frame based on file number and apexf for image number in sequence
    df.loc[df.index[df["apex"].isnull()], "apex"] = (
        df[df["apex"].isnull()]["offset"] - df[df["apex"].isnull()]["onset"]
    ) / 2
    df["apexf"] = df["apex"] - df["onset"]
    df.loc[df["dataset"] == "smic", "apexf"] = df.loc[df["dataset"] == "smic", "apex"]
    df.loc[:, "apex"] = round(df["apex"].astype("float")).astype("int")
    df.loc[:, "apexf"] = round(df["apexf"].astype("float")).astype("int")

    df = df.drop([17, 26, 130]).drop("level_0", axis=1).reset_index()

    load_data = chain(load_casme2, load_smic, load_samm)
    # remove 17, 26, 130
    load_data = (video for i, video in enumerate(load_data) if i not in [17, 26, 130])

    return df, load_data
