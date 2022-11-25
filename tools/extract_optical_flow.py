import argparse
from tqdm import tqdm

import matlab.engine
import numpy as np
from scipy import ndimage

from meb import datasets

parser = argparse.ArgumentParser(description="Test dataset setup.")
parser.add_argument(
    "--dataset_name",
    type=str,
    help=(
        "Dataset name from Smic/Casme/Casme2/Samm/Mmew/"
        "Fourd/Casme3A/Casme3C/Megc/CrossDataset"
    ),
    required=True,
)
parser.add_argument(
    "--save_location",
    type=str,
    help="Save location of .npy file, e.g., data/casme3",
    required=True,
)

args = parser.parse_args()
dataset = getattr(datasets, args.dataset_name)
c = dataset(cropped=True, color=False)

df = c.data_frame
data = c.data

eng = matlab.engine.start_matlab()


def optical_strain(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute optical strain from u and v components"""
    ux = ndimage.sobel(u, axis=0, mode="constant")
    uy = ndimage.sobel(u, axis=1, mode="constant")
    vx = ndimage.sobel(v, axis=0, mode="constant")
    vy = ndimage.sobel(v, axis=1, mode="constant")
    grad = np.sqrt(ux**2 + vy**2 + 0.5 * (uy**2 + vx**2))
    return grad


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalizes the image between [0, 1]"""
    img_min = img.min((0, 1))
    img_max = img.max((0, 1))
    normalized_img = (img - img_min) / (img_max - img_min)
    return normalized_img


uv_frames = []
for i in tqdm(range(len(data))):
    onset_f = data[i][..., 0]
    apex_f = data[i][..., df.loc[i, "apexf"]]
    ret = eng.estimate_flow_interface(
        matlab.uint8(onset_f.tolist()), matlab.uint8(apex_f.tolist()), "classic+nl-fast"
    )
    flow = np.array(ret)
    strain = optical_strain(flow[..., 0], flow[..., 1])
    frames = np.concatenate([flow, np.expand_dims(strain, 2)], 2)
    frames = normalize_img(frames)
    uv_frames.append(frames.transpose(2, 0, 1))

save_file_name = (
    args.save_location + "/" + dataset.__name__ + "uv_frames_secrets_of_OF.npy"
)
np.save(save_file_name, uv_frames)
