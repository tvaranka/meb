import torch

from meb import utils


def test_mix_labels():
    labels = torch.tensor([[1, 0], [0, 1]])
    out = utils.mixup._mix_labels(labels, lam=1, label_smoothing=0)
    assert torch.equal(out, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    out = utils.mixup._mix_labels(labels, lam=0, label_smoothing=0)
    assert torch.equal(out, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    out = utils.mixup._mix_labels(labels, lam=1, label_smoothing=0.1)
    assert torch.equal(out, torch.tensor([[0.95, 0.05], [0.05, 0.95]]))
