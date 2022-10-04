import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import dlib

from numba import jit
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
from skimage.transform import resize


def bitget(n, i):
    if isinstance(i, int):
        return 1 if n & (1 << i) > 0 else 0
    else:
        return [1 if n & (1 << j) > 0 else 0 for j in i]


def bitset(n, i, st):
    if st:
        return n | (1 << i)
    return n & ~(1 << i)


# @jit
def get_mapping(samples, mapping_type="u2"):
    """Currently only for uniform2"""
    table = np.arange(2 ** samples)
    new_max = 0
    index = 0
    if mapping_type == "u2":
        new_max = samples * (samples - 1) + 3
        for i in table:
            j = bitset(i << 1, 0, bitget(i, samples - 1))
            numt = sum(bitget(i ^ j, np.arange(samples)))

            if numt <= 2:
                table[i] = index
                index += 1
            else:
                table[i] = new_max - 1

    return table, new_max


# @jit
def LBPTOP(
    video,
    first_f,
    last_f,
    fx_radius,
    fy_radius,
    t_interval,
    neighbor_points,
    time_length,
    border_length,
    rows,
    cols,
    overratio,
    time_part,
    mapping_type,
):
    xy_neighbor_points, xt_neighbor_points, yt_neighbor_points = neighbor_points

    if mapping_type == "u2":
        code_xy, _ = get_mapping(xy_neighbor_points, mapping_type)
        code_xt, _ = get_mapping(xt_neighbor_points, mapping_type)
        code_yt, _ = get_mapping(yt_neighbor_points, mapping_type)

    # The number of neighbors in three planes may be different
    bin_xy = np.unique(code_xy).shape[0]
    bin_xt = np.unique(code_xt).shape[0]
    bin_yt = np.unique(code_yt).shape[0]
    bin_count = max(bin_xy, bin_xt, bin_yt)

    histogram = np.zeros((rows * cols * time_part * 3, bin_count))
    ts, pixels, t_over_size = 1, 1, 1
    height, width, n_frames = video.shape

    for i in range(first_f + time_length * ts, last_f - time_length * ts + 1, ts):
        p_frame = np.zeros((height, width, 2 * t_interval + 1))
        for j in range(-t_interval * ts, t_interval * ts + 1, ts):
            p_frame[..., j + t_interval] = video[..., i + j]
            if i == first_f + time_length * ts and j == -t_interval * ts:
                if 1 - overratio < 1e-5:
                    x_over_size, y_over_size = 10, 10
                else:
                    noover_block_w = np.ceil((width - border_length * 2) / cols) - 1
                    noover_block_h = np.ceil((height - border_length * 2) / rows) - 1
                    x_over_size = np.ceil(noover_block_w * overratio) - 1
                    y_over_size = np.ceil(noover_block_h * overratio) - 1
                    block_w = np.floor(
                        (width - border_length * 2 + x_over_size * (cols - 1)) / cols
                    )
                    block_h = np.floor(
                        (height - border_length * 2 + y_over_size * (rows - 1)) / rows
                    )
                    block_t = np.floor(
                        (n_frames - border_length * 2 + t_over_size * (time_part - 1))
                        / time_part
                    )

        if i - first_f - time_length - block_t < 0:
            cur_t = 0
        else:
            cur_t = (
                np.ceil(
                    (i - first_f - time_length - block_t) / (block_t - t_over_size) + 1
                )
                - 1
            )

        if cur_t >= time_part:
            cur_t = time_part - 1

        new_t = (
            np.ceil((i - first_f - time_length + (cur_t + 1) * t_over_size) / block_t)
            - 1
        )
        if new_t >= time_part:
            new_t = time_part - 1

        for yc in range(border_length + 1, height - border_length + 1):
            for xc in range(border_length + 1, width - border_length + 1):
                if yc - border_length - block_h - 1 < 0:
                    cur_rows = 0
                else:
                    cur_rows = np.floor(
                        (yc - border_length - block_h - 1) / (block_h - y_over_size) + 1
                    )

                if xc - border_length - block_w - 1 < 0:
                    cur_cols = 0
                else:
                    cur_cols = np.floor(
                        (xc - border_length - block_w - 1) / (block_w - x_over_size) + 1
                    )

                if cur_rows >= rows:
                    cur_rows = rows - 1

                if cur_cols >= cols:
                    cur_cols = cols - 1

                new_rows = (
                    np.ceil(
                        (yc - border_length + (cur_rows + 1) * y_over_size) / block_h
                    )
                    - 1
                )
                new_cols = (
                    np.ceil(
                        (xc - border_length + (cur_cols + 1) * x_over_size) / block_w
                    )
                    - 1
                )

                if new_rows >= rows:
                    new_rows = rows - 1

                if new_cols >= cols:
                    new_cols = cols - 1

                # XY plane
                basic_lbp = 1
                center_value = p_frame[yc - 1, xc - 1, t_interval]

                for p in range(xy_neighbor_points):
                    x = np.floor(
                        xc
                        + fx_radius * np.cos((2 * np.pi * p) / xy_neighbor_points)
                        + 0.5
                    )
                    y = np.floor(
                        yc
                        - fy_radius * np.sin((2 * np.pi * p) / xy_neighbor_points)
                        + 0.5
                    )
                    current_value = p_frame[int(y) - 1, int(x) - 1, t_interval]
                    if current_value >= center_value:
                        basic_lbp = basic_lbp + 2 ** p

                histogram[
                    int((cur_t * (rows * cols) + cur_rows * cols + cur_cols) * 3),
                    code_xy[basic_lbp - 1],
                ] += 1

                if (new_rows != cur_rows) | (new_cols != cur_cols) | (new_t != cur_t):
                    histogram[
                        int((new_t * (rows * cols) + new_rows * cols + new_cols) * 3),
                        code_xy[basic_lbp - 1],
                    ] += 1
                # XT plane
                basic_lbp = 1
                for p in range(xt_neighbor_points):
                    x = np.floor(
                        xc
                        + fx_radius * np.cos((2 * np.pi * p) / xt_neighbor_points)
                        + 0.5
                    )
                    z = np.floor(
                        i
                        - t_interval * np.sin((2 * np.pi * p) / xt_neighbor_points)
                        + 0.5
                    )
                    current_value = p_frame[
                        int(yc) - 1, int(x) - 1, int(z - i + t_interval)
                    ]
                    if current_value >= center_value:
                        basic_lbp = basic_lbp + 2 ** p

                histogram[
                    int((cur_t * (rows * cols) + cur_rows * cols + cur_cols) * 3 + 1),
                    code_xt[basic_lbp - 1],
                ] += 1

                if (new_rows != cur_rows) | (new_cols != cur_cols) | (new_t != cur_t):
                    histogram[
                        int(
                            (new_t * (rows * cols) + new_rows * cols + new_cols) * 3 + 1
                        ),
                        code_xt[basic_lbp - 1],
                    ] += 1
                # Yt plane
                basic_lbp = 1
                for p in range(yt_neighbor_points):
                    y = np.floor(
                        yc
                        - fx_radius * np.cos((2 * np.pi * p) / yt_neighbor_points)
                        + 0.5
                    )
                    z = np.floor(
                        i
                        - t_interval * np.sin((2 * np.pi * p) / yt_neighbor_points)
                        + 0.5
                    )
                    current_value = p_frame[
                        int(y) - 1, int(xc) - 1, int(z - i + t_interval)
                    ]
                    if current_value >= center_value:
                        basic_lbp = basic_lbp + 2 ** p

                histogram[
                    int((cur_t * (rows * cols) + cur_rows * cols + cur_cols) * 3 + 2),
                    code_xt[basic_lbp - 1],
                ] += 1

                if (new_rows != cur_rows) | (new_cols != cur_cols) | (new_t != cur_t):
                    histogram[
                        int(
                            (new_t * (rows * cols) + new_rows * cols + new_cols) * 3 + 2
                        ),
                        code_xt[basic_lbp - 1],
                    ] += 1

    return histogram.reshape(1, -1)


def get_path_weight_matrix(n):
    G = np.zeros((n, n))
    for i in range(n - 1):
        G[i, i + 1] = 1
        G[i + 1, i] = 1
    return G


def get_laplacian(G):
    n = G.shape[0]

    D = np.zeros((n, n))
    D[: n - 1]
    idx = np.arange(n) * n + np.arange(1, n + 1) - 1
    D.reshape(-1)[idx] = G.sum(0)
    D = D.reshape(n, n)
    L = D - G
    return L


def tim(video, n_frames):
    n = video.shape[-1]
    color = len(video.shape) == 4
    X = video.reshape(-1, n).copy()
    X = X.astype("float64")
    # check linear independency
    if np.linalg.matrix_rank(X) < n:
        assert "TIM: Invalid input."

    mu = X.mean(1)
    X -= np.tile(mu, (n, 1)).T
    U, S, V = np.linalg.svd(X, full_matrices=False)
    S = S[:-1]
    V = V.T[:, :-1]
    U = U[:, :-1]
    Q = S.reshape(S.shape[0], 1) * V.T
    G = get_path_weight_matrix(n)
    L = get_laplacian(G)

    D, V = np.linalg.eig(L)
    # Sort eigenvalues as numpy does not do this automatically
    idx = D.argsort()
    D = D[idx]
    V = V[:, idx]

    V = V[:, 1:]
    W, _, _, _ = np.linalg.lstsq(Q @ Q.T, Q @ V, rcond=None)
    m = np.zeros((n - 1, 1))
    for i in range(n - 1):
        m[i] = (Q[:, 0].T @ W[:, i]) / np.sin(
            1 / n * (i + 1) * np.pi + np.pi * (n - (i + 1)) / (2 * n)
        )

    pos = (np.arange(n_frames) / (n_frames - 1)) * (1 - 1 / n) + 1 / n
    X = np.zeros((U.shape[0], n_frames))
    ndim = W.shape[0]
    for i in range(n_frames):
        v = np.zeros((ndim, 1))
        for j in range(ndim):
            v[j] = np.sin(pos[i] * (j + 1) * np.pi + np.pi * (n - (j + 1)) / (2 * n))
        ls, _, _, _ = np.linalg.lstsq(W.T, v * m, rcond=None)
        X[:, i] = (U @ ls).squeeze() + mu
    if color:
        X = X.reshape(video.shape[0], video.shape[1], 3, -1)
    else:
        X = X.reshape(video.shape[0], video.shape[1], -1)
    X = np.round(X)
    return X


def load_roi_36(feature_points):
    """This function calculates the 36 regions of interest (ROI)"""
    m, n = feature_points.shape
    if m != 2 and n != 68:
        return

    ROI36X = np.zeros((4, 36))  # resultX
    ROI36Y = np.zeros((4, 36))  # resultY
    tp = np.zeros((2, 22))  # middle point
    fp = feature_points  # feature point
    delta = 20  # middle value

    # get the middle point
    tp[0, 0] = fp[0, 17]
    tp[1, 0] = max(fp[1, 17] - delta, 0)
    tp[0, 1] = fp[0, 19]
    tp[1, 1] = max(fp[1, 19] - delta, 0)

    tp[0, 2] = fp[0, 21]
    tp[1, 2] = max(fp[1, 21] - delta, 0)

    tp[0, 4] = fp[0, 22]
    tp[1, 4] = max(fp[1, 22] - delta, 0)

    tp[:, 3] = (tp[:, 2] + tp[:, 4]) * 0.5

    tp[0, 5] = fp[0, 24]
    tp[1, 5] = max(fp[1, 24] - delta, 0)

    tp[0, 6] = fp[0, 26]
    tp[1, 6] = max(fp[1, 26] - delta, 0)

    tp[:, 7] = (fp[:, 3] + fp[:, 29]) * 0.5

    tp[:, 8] = (fp[:, 13] + fp[:, 29]) * 0.5

    tp[:, 9] = (fp[:, 4] + fp[:, 33]) * 0.5

    tp[:, 10] = (fp[:, 12] + fp[:, 33]) * 0.5

    tp[:, 11] = (fp[:, 4] + fp[:, 33]) * 0.5

    tp[:, 12] = (fp[:, 12] + fp[:, 33]) * 0.5

    tp[:, 13] = (fp[:, 5] + fp[:, 59]) * 0.5

    tp[:, 14] = (fp[:, 11] + fp[:, 55]) * 0.5

    tp[:, 15] = (fp[:, 2] + fp[:, 41]) * 0.5

    tp[:, 16] = (fp[:, 14] + fp[:, 46]) * 0.5

    tp[:, 17] = (fp[:, 5] + fp[:, 59]) * 0.5

    tp[:, 18] = (fp[:, 6] + fp[:, 58]) * 0.5

    tp[:, 19] = (fp[:, 8] + fp[:, 57]) * 0.5

    tp[:, 20] = (fp[:, 10] + fp[:, 56]) * 0.5

    tp[:, 21] = (fp[:, 11] + fp[:, 55]) * 0.5

    # get ROI
    ROI36X[:, 0] = [fp[0, 17], tp[0, 0], tp[0, 1], fp[0, 19]]
    ROI36Y[:, 0] = [fp[1, 17], tp[1, 0], tp[1, 1], fp[1, 19]]
    ROI36X[:, 1] = [fp[0, 19], tp[0, 1], tp[0, 2], fp[0, 21]]
    ROI36Y[:, 1] = [fp[1, 19], tp[1, 1], tp[1, 2], fp[1, 21]]
    ROI36X[:, 2] = [fp[0, 21], tp[0, 2], tp[0, 3], fp[0, 27]]
    ROI36Y[:, 2] = [fp[1, 21], tp[1, 2], tp[1, 3], fp[1, 27]]
    ROI36X[:, 3] = [fp[0, 27], tp[0, 3], tp[0, 4], fp[0, 22]]
    ROI36Y[:, 3] = [fp[1, 27], tp[1, 3], tp[1, 4], fp[1, 22]]
    ROI36X[:, 4] = [fp[0, 22], tp[0, 4], tp[0, 5], fp[0, 24]]
    ROI36Y[:, 4] = [fp[1, 22], tp[1, 4], tp[1, 5], fp[1, 24]]
    ROI36X[:, 5] = [fp[0, 24], tp[0, 5], tp[0, 6], fp[0, 26]]
    ROI36Y[:, 5] = [fp[1, 24], tp[1, 5], tp[1, 6], fp[1, 26]]
    ROI36X[:, 6] = [fp[0, 1], fp[0, 0], fp[0, 17], fp[0, 36]]
    ROI36Y[:, 6] = [fp[1, 1], fp[1, 0], fp[1, 17], fp[1, 36]]
    ROI36X[:, 7] = [fp[0, 36], fp[0, 17], fp[0, 19], fp[0, 38]]
    ROI36Y[:, 7] = [fp[1, 36], fp[1, 17], fp[1, 19], fp[1, 38]]
    ROI36X[:, 8] = [fp[0, 38], fp[0, 19], fp[0, 21], fp[0, 39]]
    ROI36Y[:, 8] = [fp[1, 38], fp[1, 19], fp[1, 21], fp[1, 39]]
    ROI36X[:, 9] = [fp[0, 39], fp[0, 21], fp[0, 27], fp[0, 28]]
    ROI36Y[:, 9] = [fp[1, 39], fp[1, 21], fp[1, 27], fp[1, 28]]
    ROI36X[:, 10] = [fp[0, 28], fp[0, 27], fp[0, 22], fp[0, 42]]
    ROI36Y[:, 10] = [fp[1, 28], fp[1, 27], fp[1, 22], fp[1, 42]]
    ROI36X[:, 11] = [fp[0, 42], fp[0, 22], fp[0, 24], fp[0, 43]]
    ROI36Y[:, 11] = [fp[1, 42], fp[1, 22], fp[1, 24], fp[1, 43]]
    ROI36X[:, 12] = [fp[0, 43], fp[0, 24], fp[0, 26], fp[0, 45]]
    ROI36Y[:, 12] = [fp[1, 43], fp[1, 24], fp[1, 26], fp[1, 45]]
    ROI36X[:, 13] = [fp[0, 45], fp[0, 26], fp[0, 16], fp[0, 15]]
    ROI36Y[:, 13] = [fp[1, 45], fp[1, 26], fp[1, 16], fp[1, 15]]
    ROI36X[:, 14] = [fp[0, 3], fp[0, 2], tp[0, 15], tp[0, 7]]
    ROI36Y[:, 14] = [fp[1, 3], fp[1, 2], tp[1, 15], tp[1, 7]]
    ROI36X[:, 15] = [tp[0, 7], tp[0, 15], fp[0, 41], fp[0, 39]]
    ROI36Y[:, 15] = [tp[1, 7], tp[1, 15], fp[1, 41], fp[1, 39]]
    ROI36X[:, 16] = [tp[0, 7], fp[0, 39], fp[0, 28], fp[0, 29]]
    ROI36Y[:, 16] = [tp[1, 7], fp[1, 39], fp[1, 28], fp[1, 29]]
    ROI36X[:, 17] = [fp[0, 29], fp[0, 28], fp[0, 42], tp[0, 8]]
    ROI36Y[:, 17] = [fp[1, 29], fp[1, 28], fp[1, 42], tp[1, 8]]
    ROI36X[:, 18] = [tp[0, 8], fp[0, 42], fp[0, 46], tp[0, 16]]
    ROI36Y[:, 18] = [tp[1, 8], fp[1, 42], fp[1, 46], tp[1, 16]]
    ROI36X[:, 19] = [tp[0, 8], tp[0, 16], fp[0, 14], fp[0, 13]]
    ROI36Y[:, 19] = [tp[1, 8], tp[1, 16], fp[1, 14], fp[1, 13]]
    ROI36X[:, 20] = [tp[0, 11], tp[0, 7], fp[0, 29], fp[0, 33]]
    ROI36Y[:, 20] = [tp[1, 11], tp[1, 7], fp[1, 29], fp[1, 33]]
    ROI36X[:, 21] = [fp[0, 33], fp[0, 29], tp[0, 8], tp[0, 12]]
    ROI36Y[:, 21] = [fp[1, 33], fp[1, 29], tp[1, 8], tp[1, 12]]
    ROI36X[:, 22] = [tp[0, 13], tp[0, 9], fp[0, 33], fp[0, 59]]
    ROI36Y[:, 22] = [tp[1, 13], tp[1, 9], fp[1, 33], fp[1, 59]]
    ROI36X[:, 23] = [fp[0, 55], fp[0, 33], tp[0, 10], tp[0, 14]]
    ROI36Y[:, 23] = [fp[1, 55], fp[1, 33], tp[1, 10], tp[1, 14]]
    ROI36X[:, 24] = [tp[0, 18], tp[0, 17], fp[0, 59], fp[0, 58]]
    ROI36Y[:, 24] = [tp[1, 18], tp[1, 17], fp[1, 59], fp[1, 58]]
    ROI36X[:, 25] = [tp[0, 19], tp[0, 18], fp[0, 58], fp[0, 57]]
    ROI36Y[:, 25] = [tp[1, 19], tp[1, 18], fp[1, 58], fp[1, 57]]
    ROI36X[:, 26] = [tp[0, 20], tp[0, 19], fp[0, 57], fp[0, 56]]
    ROI36Y[:, 26] = [tp[1, 20], tp[1, 19], fp[1, 57], fp[1, 56]]
    ROI36X[:, 27] = [tp[0, 21], tp[0, 20], fp[0, 56], fp[0, 55]]
    ROI36Y[:, 27] = [tp[1, 21], tp[1, 20], fp[1, 56], fp[1, 55]]
    ROI36X[:, 28] = [fp[0, 4], fp[0, 3], tp[0, 7], tp[0, 11]]
    ROI36Y[:, 28] = [fp[1, 4], fp[1, 3], tp[1, 7], tp[1, 11]]
    ROI36X[:, 29] = [tp[0, 12], tp[0, 8], fp[0, 13], fp[0, 12]]
    ROI36Y[:, 29] = [tp[1, 12], tp[1, 8], fp[1, 13], fp[1, 12]]
    ROI36X[:, 30] = [fp[0, 5], fp[0, 4], tp[0, 9], tp[0, 13]]
    ROI36Y[:, 30] = [fp[1, 5], fp[1, 4], tp[1, 9], tp[1, 13]]
    ROI36X[:, 31] = [tp[0, 14], tp[0, 10], fp[0, 12], fp[0, 11]]
    ROI36Y[:, 31] = [tp[1, 14], tp[1, 10], fp[1, 12], fp[1, 11]]
    ROI36X[:, 32] = [fp[0, 6], fp[0, 5], tp[0, 17], tp[0, 18]]
    ROI36Y[:, 32] = [fp[1, 6], fp[1, 5], tp[1, 17], tp[1, 18]]
    ROI36X[:, 33] = [fp[0, 8], fp[0, 6], tp[0, 18], tp[0, 19]]
    ROI36Y[:, 33] = [fp[1, 8], fp[1, 6], tp[1, 18], tp[1, 19]]
    ROI36X[:, 34] = [fp[0, 10], fp[0, 8], tp[0, 19], tp[0, 20]]
    ROI36Y[:, 34] = [fp[1, 10], fp[1, 8], tp[1, 19], tp[1, 20]]
    ROI36X[:, 35] = [fp[0, 11], fp[0, 10], tp[0, 20], tp[0, 21]]
    ROI36Y[:, 35] = [fp[1, 11], fp[1, 10], tp[1, 20], tp[1, 21]]

    au_region = []
    for i in range(36):
        au_region.append(np.array((ROI36X[:, i], ROI36Y[:, i])).T)
    return np.array(au_region)


def calculate_lbptop(df, videos, tim_frames=0):
    lbptop_features = np.zeros((df.shape[0], 11328))
    for i, video in enumerate(videos):
        if i % 30 == 0:
            print(i, "/", df.shape[0])
        if tim_frames != 0:
            video = tim(video, tim_frames)
        lbptop_features[i, :] = LBPTOP(
            video, 0, 9, 3, 3, 3, [8, 8, 8], 3, 3, 8, 8, 0.7, 1, "u2"
        )
    return lbptop_features


def shape2points(shape, dtype="int", pointNum=68):
    coords = np.zeros((pointNum, 2), dtype=dtype)
    for i in range(0, pointNum):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def polyroi(img, c, r):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    rc = np.array((c, r)).T
    cv2.drawContours(mask, [rc], 0, 255, -1)
    return mask


@jit
def face_alignment(u, v, p):
    """Aligns the face in the optical flow domain.

    Parameters
    ----------
      u : optical flow in the x-direction, array of size m x n
      v : optical flow in the y-direction, array of size m x n
      p : facial feature points of the first frame of the sample, array of size 2 x 68

    Returns
    -------
    u, v : the optical flow components aligned in the optical flow domain
    """
    m, n = u.shape
    p = p.T
    # Use 17 feature points from the DRMF model
    M, N = 13, 2
    Pi = np.zeros((M, N))
    P1 = np.zeros((M, N))
    points = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 30]
    for i, point in enumerate(points):
        px = max(p[0, point], 0)
        py = max(p[1, point], 0)
        P1[i, 0] = px + 1
        P1[i, 1] = py + 1
        Pi[i, 0] = px + u[py, px]
        Pi[i, 1] = py + v[py, px]
    # Solve the least squares problem for T
    T, _, _, _ = np.linalg.lstsq(Pi, P1, rcond=None)
    print(T)
    uv = np.zeros((m, n))
    vv = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            x = u[i, j] + j
            y = v[i, j] + i

            uv[i, j] = x * T[0, 0] + y * T[1, 0] - j
            vv[i, j] = x * T[0, 1] + y * T[1, 1] - i

    return uv, vv


def MDMO_video(video, feature_points):
    # Calculate roi
    height, width, n_frames = video.shape
    au_region = load_roi_36(feature_points.T)
    n_region = au_region.shape[0]
    img = video[:, :, 0]
    template = img
    au_region_mask = []
    for r in range(n_region):
        mask = polyroi(
            template, au_region[r][:, 0].astype(int), au_region[r][:, 1].astype(int)
        )
        au_region_mask.append(mask)
    au_region_mask = np.array(au_region_mask)
    U = np.zeros((height, width, n_frames - 1))
    V = np.zeros((height, width, n_frames - 1))

    # Calculate the optical flow
    for f in range(1, n_frames):
        img_f = video[:, :, f]
        flow = cv2.calcOpticalFlowFarneback(
            template, img_f, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        u = flow[..., 0]
        v = flow[..., 1]
        U[:, :, f - 1] = u
        V[:, :, f - 1] = v

    # Calculate MDMO
    bin_ranges = np.array(
        [
            0,
            np.pi / 8,
            np.pi / 8 * 3,
            np.pi / 8 * 5,
            np.pi / 8 * 7,
            np.pi / 8 * 9,
            np.pi / 8 * 11,
            np.pi / 8 * 13,
            np.pi / 8 * 15,
            np.pi * 2,
        ]
    )
    features = np.zeros((2, n_region, n_frames - 1))
    for f in range(n_frames - 1):
        # Face alignment in the OF domain
        # u, v = face_alignment(U[:, :, f], V[:, :, f], feature_points)
        u = U[:, :, f]
        v = V[:, :, f]
        # Translate to polar coordinates from cartesian
        rho, theta = cv2.cartToPolar(u, v)
        # Calculate the main direction for each of the ROIs
        for r in range(n_region):
            mask = au_region_mask[r]
            theta_masked = theta[mask != 0]
            rho_masked = rho[mask != 0]
            bincounts, _ = np.histogram(theta_masked, bin_ranges)
            inds = np.digitize(theta_masked, bin_ranges)
            tmp = bincounts
            tmp[0] = tmp[0] + tmp[-1] + tmp[-2]
            bincounts = tmp
            inds[inds == 9] = 1
            inds[inds == 8] = 1
            max_number = np.max(bincounts)
            max_index = np.argmax(bincounts)
            theta_max = theta_masked[inds == max_index + 1]
            rho_max = rho_masked[inds == max_index + 1]
            theta_mean = theta_max.mean()
            rho_mean = rho_max.mean()
            features[0, r, f] = rho_mean
            features[1, r, f] = theta_mean

    # Weights
    weight_features = np.zeros((2, n_region))
    rho_sum = features[0].sum().sum()

    rho_t = features[0].copy().T
    theta_t = features[1].copy().T
    uf, vf = cv2.polarToCart(rho_t, theta_t)
    wuf = np.zeros((1, n_region))
    wvf = np.zeros((1, n_region))
    for f in range(n_frames - 1):
        alpha = features[1, :, f].sum() / rho_sum
        wuf += alpha * uf[f]
        wvf += alpha * vf[f]

    feature_rho, feature_theta = cv2.cartToPolar(wuf, wvf)
    tmp = feature_rho.max()
    feature_rho = feature_rho / tmp
    feature_theta -= np.pi

    return feature_rho, feature_theta


def calculate_mdmo(df, load_data):
    n_samples = df.shape[0]
    # 36 is the number of ROIs. 2 comes from rho and theta
    MDMO_features = np.zeros((n_samples, 36 * 2))
    # Setup detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../../data/shape_predictor_68_face_landmarks.dat")
    prev_shape = None  # May brake if not able to detect face on first video
    for i, video in enumerate(load_data):
        if i % 10 == 0:
            print(i, "/", df.shape[0])
        # calculate the feature_points for the first frame of the video
        img = video[..., 0]
        rects = detector(img, 2) if detector(img, 2) else detector(img, 3)
        if rects:
            shape = predictor(img, rects[0])
        else:
            shape = prev_shape
        prev_shape = shape
        feature_points = shape2points(shape)
        # MDMO
        feature_rho, feature_theta = MDMO_video(video, feature_points)
        MDMO_features[i, :36] = feature_rho
        MDMO_features[i, 36:] = feature_theta
    return MDMO_features
