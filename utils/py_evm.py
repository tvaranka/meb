import numpy as np
import torch
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
import os


def magnify(video, alpha=10, r1=0.4, r2=0.05, n_levels=6, lambda_c=16, overwrite=False):
    """
    Magnify a given video.

    Magnify the motion of a video using the Eulerian Video Magnification from the paper
    Eulerian Video Magnification for Revealing Subtle Changes in the World, Hao-Yu Wu and
    Michael Rubinstein and Eugene Shih and John Guttag and Fr\'{e}do Durand and
    William T. Freeman, ACM Transactions on Graphics (Proc. SIGGRAPH 2012).

    If a location of a file is given load_video is called first which loads the video,
    after which the magnification is done, next the video is saved with the same
    format as specified.

    If a numpy/pytorch array is given the motion magnified video is returned as a numpy
    array.

    Parameters
    ----------
    video : numpy/pytorch array or a string for .mp4 file or folder of images
        The video to be magnified. Image should be in the format of (F, W, H, C)
            where
            F = number of frames,
            H = height,
            W = width,
            C = channels.
        Image should be in RGB.
    alpha : int or float, optional
        A parameter that specifies the amount of magnification, a typical value is
        between 5 and 150, see Table 1 from paper. Defaults to 10.
    r1 : int or float, optional
        Upper bound cutoff frequency of the bandpass filter. For normalized frequencies
        should be between 0-1Hz.
    r2 : int or float, optinal
        Lower bound cutoff frequency of the bandpass filter. For normalized frequencies
        should be between 0-1Hz.
    lambda_c : int or float, optinal
        Spatial frequency cutoff after which the alpha value is used for magnification.
    overwrite : boolean, optional
        Only used if video parameter is a string. Determines whether the saved magnified
        video is overwitten, or a new one is created.


    Returns
    -------
    mm_video : numpy array or nothing
        If the given parameter video was a numpy/pytorch array a numpy array in the same
        format is returned. If the given parameter video was a string to a video or a
        folder of images nothing is returned. Instead the image is saved.

    Examples
    --------
    >>> magnify("data/baby.mp4")
    >>> magnify("data/video", alpha=5, r1=0.1, r2=0.01)

    >>> video = load_video("data/baby.mp4")
    >>> mm_video = magnify(video)
    >>> save_video(mm_video, "mm_baby")

    """
    if isinstance(video, str):
        video_loaded = load_video(video)
        mm_video = _magnify(video_loaded, alpha, lambda_c, r1, r2, n_levels)
        save_video(mm_video, video, overwrite=overwrite)
    else:
        mm_video = _magnify(video, alpha, lambda_c, r1, r2, n_levels)
        return mm_video


def load_video(video_path):
    """ """
    assert os.path.exists(
        video_path
    ), "File/folder not found, make sure it exists and is in the current directory."

    if video_path[-3:] == "mp4":
        video = _load_video(video_path)
    else:
        video = _load_directory(video_path)
    return video


def save_video(video, video_path, overwrite=False):
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    if os.path.exists(video_path):
        video_path = "/".join(video_path.split("/")[1:])

    video_path = results_folder + "/" + video_path
    if not overwrite:
        video_path = _check_file_and_return_name(video_path)
    if video_path[-3:] == "mp4":
        _save_video(video, video_path)
    else:
        _save_directory(video, video_path)


def _magnify(video, alpha, lambda_c, r1, r2, n_levels):
    video = _preprocess(video)
    F, H, W, C = _shape(video)
    # change size to closest divisible to n_levels^2 if not divisible
    divisor = 2 ** (n_levels - 1)
    if H % divisor != 0 or W % divisor != 0:
        new_H = H + divisor - H % divisor
        new_W = W + divisor - W % divisor
        video = (
            resize(video, (F, new_H, new_W, C))
            if C == 3
            else resize(video, (F, new_H, new_W))
        )

    frame = video[0]
    output_video = np.zeros_like(video)

    pyr = _laplacian_pyramid(frame, n_levels)
    lowpass1 = pyr
    lowpass2 = pyr

    for i in range(1, F):
        pyr = _laplacian_pyramid(video[i], n_levels)
        lowpass1 = [r1 * pyr[k] + (1 - r1) * lowpass1[k] for k in range(n_levels + 1)]
        lowpass2 = [r2 * pyr[k] + (1 - r2) * lowpass2[k] for k in range(n_levels + 1)]
        filtered = [lowpass1[k] - lowpass2[k] for k in range(n_levels + 1)]

        delta = lambda_c / 8 / (1 + alpha)
        lmbd = (H**2 + W**2) ** 0.5 / 3

        for j in reversed(range(len(pyr))):
            cur_alpha = (lmbd / delta / 8 - 1) * 2
            if j == len(pyr) - 1 or j == 0:
                filtered[j] = np.zeros_like(filtered[j])
            elif cur_alpha > alpha:
                filtered[j] = alpha * filtered[j]
            else:
                filtered[j] = cur_alpha * filtered[j]

            lmbd /= 2
        output = _reconstruct_laplacian(filtered)
        output_video[i] = frame + output
    output_video[0] = frame
    output_video[output_video < 0] = 0
    output_video[output_video > 1] = 1

    # change video back to original size
    _, out_H, out_W, _ = _shape(output_video)
    if out_H != H or out_W != W:
        output_video = resize(output_video, (F, H, W, C))
    if output_video.shape[-1] == 1:
        output_video = output_video[..., 0]
    return output_video


def _preprocess(video):
    assert isinstance(
        video, (np.ndarray, torch.Tensor)
    ), "The video should be either a numpy array or a torch tensor"
    assert (
        len(video.shape) == 3 or len(video.shape) == 4
    ), "The shape of the video should be either (F, H, W) or \
                                                            (F, H, W, C)"
    F, H, W, C = _shape(video)
    assert (
        C == 3 or C == 1
    ), "The shape of the video should be either (F, H, W) or \
                                                            (F, H, W, C)"
    video = np.array(video) if isinstance(video, torch.Tensor) else video
    video = (
        video
        if video.dtype == np.float32 or video.dtype == np.float64
        else video.astype(np.float32)
    )
    video = video / 255.0 if video.max() > 1 else video
    return video


def _check_file_and_return_name(file):
    """
    Check whether a file exists and if it does, accumulate file number so that we don't overwrite previous files.
    """

    i = 2
    if file[-3:] == "mp4":
        if os.path.exists(file):
            file = file[:-4] + str(i) + ".mp4"
            i += 1
        while os.path.exists(file):
            file = file[:-5] + str(i) + ".mp4"
            i += 1
        return file
    else:
        while os.path.exists(file):
            file = file + str(i)
        return file


def _load_directory(video_path):
    image_paths = os.listdir(video_path)
    assert len(image_paths) != 0, "No images found in the folder"

    # load a single image to get its properties
    img = plt.imread(video_path + "/" + image_paths[0])
    W, H, C = img.shape
    images = np.empty((len(image_paths), W, H, C), dtype="uint8")
    for i in range(len(image_paths)):
        img = plt.imread(video_path + "/" + image_paths[i])
        images[i] = img.astype("uint8")

    return images


def _load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    video = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.append(frame)

    video = np.stack(video)
    return video


def _save_video(video, video_name=None):
    if not video_name:
        video_name = "evm.mp4"
    """A numpy/torch array in (F, H, W, C) format,
       where
       F = number of frames,
       H = height,
       W = width,
       C = channels
    """
    video = np.array(video)
    F, H, W, C = _shape(video)
    # convert to uint8 and BGR first for opencv
    vid = np.zeros_like(video, dtype="uint8")
    if video.dtype != "uint8":
        video *= 255
    for i in range(F):
        vid[i] = cv2.cvtColor(video[i].astype("uint8"), cv2.COLOR_RGB2BGR)

    # define video writer and save video
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    video_writer = cv2.VideoWriter(video_name, fourcc, 30, (W, H))
    for image in vid:
        video_writer.write(image)
    video_writer.release()
    return video_name


def _save_directory(video, folder_path):
    """
    Takes in a numpy/tensor array in RGB and a folder path
    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    F, H, W, C = _shape(video)
    for i in range(F):
        save_name = folder_path + "/" + folder_path.split("/")[-1] + str(i) + ".jpg"
        plt.imsave(save_name, video[i])


def _shape(frame):
    if frame.shape[-1] == 3 and len(frame.shape) >= 3:
        return frame.shape
    elif frame.shape[-1] == 1:
        return frame.shape
    else:
        return frame.shape + (1,)


def _laplacian_pyramid(frame, n_levels=6):
    lower = frame.copy()
    gaussian_pyr = [lower]
    for i in range(n_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(lower)

    laplacian_top = gaussian_pyr[-1]
    laplacian_pyr = [laplacian_top]
    for i in range(n_levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr


def _reconstruct_laplacian(pyr):
    n_levels = len(pyr)
    W, H, C = _shape(pyr[1])
    frame = (
        resize(pyr[0], (W, H, C)) + pyr[1]
        if C == 3
        else resize(pyr[0], (W, H)) + pyr[1]
    )
    for i in range(1, n_levels - 1):
        frame = cv2.pyrUp(frame) + pyr[i + 1]
    return frame
