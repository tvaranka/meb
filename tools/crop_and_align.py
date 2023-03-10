import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from retinaface import RetinaFace
from retinaface.commons.postprocess import alignment_procedure
from tqdm import tqdm

from meb import datasets

parser = argparse.ArgumentParser(description="Face alignment and cropping.")
parser.add_argument(
    "--dataset_name",
    help="Dataset name from Smic/Casme/Casme2/Samm/Mmew/Fourd/Casme3A/Casme3C",
)
parser.add_argument(
    "--save_location",
    help="Save location of cropped folder, e.g., data/casme3_cropped/",
)
parser.add_argument(
    "--n_folders",
    default=3,
    type=int,
    help=(
        "The number of folders from root to frame.jpg. For example:"
        "root=data/Casme2, then n_folders should be set to 3 because"
        "sub01/EP02_01f/img46.jpg is three folders away."
    ),
)


def align_img(img: np.ndarray, landmarks: dict) -> np.ndarray:
    aligned = alignment_procedure(
        img, landmarks["right_eye"], landmarks["left_eye"], landmarks["nose"]
    )
    return aligned


def crop_img(img: np.ndarray, facial_area: List[int]) -> np.ndarray:
    fa = facial_area
    # Add extra width of around 2% for both sides
    extra_width = int(img.shape[1] * 0.02)
    return img[fa[1] : fa[3], fa[0] - extra_width : fa[2] + extra_width]


def extract_facial_area_and_landmarks(
    onset: np.ndarray, threshold: float = 0.9
) -> Tuple[dict, dict]:
    obj = RetinaFace.detect_faces(onset, threshold=threshold)
    # Error handling if face not found, try with lower threshold
    if not isinstance(obj, dict):  # Returns dict if face found
        obj = RetinaFace.detect_faces(onset, threshold=0.5)
        # If face found, set new threshold to 0.5 for alignment
        if isinstance(obj, dict):
            threshold = 0.5
    # Extract 5 facial landmarks
    landmarks = obj["face_1"]["landmarks"]
    aligned_onset = align_img(onset, landmarks)
    obj_aligned = RetinaFace.detect_faces(aligned_onset, threshold=threshold)
    # Extract borders of face
    facial_area = obj_aligned["face_1"]["facial_area"]
    return landmarks, facial_area


def crop_align_video(video: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    onset = video[0]
    landmarks, facial_area = extract_facial_area_and_landmarks(onset, threshold)
    new_video = []
    for i, frame in enumerate(video):
        aligned_frame = align_img(frame, landmarks)
        cropped_frame = crop_img(aligned_frame, facial_area)
        new_video.append(cropped_frame)
    return np.array(new_video)


def save_video(video: np.ndarray, data_path: List[str], root_path: str) -> None:
    subject_path = "/".join(data_path[0].split("/")[-n_folders:-1])
    subject_path = root_path + subject_path
    if not os.path.exists(subject_path):
        os.makedirs(subject_path)
    for i, frame in enumerate(video):
        sample_path = "/".join(data_path[i].split("/")[-n_folders:])
        frame_path = root_path + sample_path
        plt.imsave(frame_path, frame)


args = parser.parse_args()

n_folders = args.n_folders

# Setup dataset
dataset = getattr(datasets, args.dataset_name)
c = dataset(cropped=False, color=True)
df = c.data_frame
data = c.data

data_paths = data.data_path
root_path = args.save_location
if not os.path.exists(root_path):
    os.mkdir(root_path)

for i, video in tqdm(enumerate(data)):
    n_samples = video.shape[0]
    cropped_video = crop_align_video(video)
    save_video(cropped_video, data_paths[i], root_path)
