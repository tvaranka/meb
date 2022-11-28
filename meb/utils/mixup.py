import torch
import pytorchvideo.transforms

# Uses CutMix from pytorchvideo, but _mix_labels function has been modified
# to work with multi-label inputs


def _mix_labels(
    labels: torch.Tensor,
    num_classes: int = None,
    lam: float = 1.0,
    label_smoothing: float = 0.0,
):
    """
    This function converts class indices to one-hot vectors and mix labels, given the
    number of classes.

    Args:
        labels (torch.Tensor): Class labels.
        num_classes (int): Total number of classes.
        lam (float): lamba value for mixing labels.
        label_smoothing (float): Label smoothing value.
    """
    off_value = label_smoothing / labels.shape[-1]
    on_value = 1 - label_smoothing + off_value
    y = labels.clone()
    y = torch.where(y == 1, on_value, off_value)
    return lam * y + (1.0 - lam) * y.flip(0)


pytorchvideo.transforms.mix._mix_labels = _mix_labels
CutMix = pytorchvideo.transforms.CutMix
MixUp = pytorchvideo.transforms.MixUp
MixVideo = pytorchvideo.transforms.MixVideo
