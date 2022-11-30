import copy

import numpy as np
import torch

from meb import models, utils


def test_mix_labels():
    labels = torch.tensor([[1, 0], [0, 1]])
    out = utils.mixup._mix_labels(labels, lam=1, label_smoothing=0)
    assert torch.equal(out, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    out = utils.mixup._mix_labels(labels, lam=0, label_smoothing=0)
    assert torch.equal(out, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    out = utils.mixup._mix_labels(labels, lam=1, label_smoothing=0.1)
    assert torch.equal(out, torch.tensor([[0.95, 0.05], [0.05, 0.95]]))


def test_multi_label_f1_shape():
    labels = torch.tensor([[1, 0], [0, 1]])
    predictions = torch.tensor([[0.95, -0.4], [0.1, 0.6]])
    metric1 = utils.MultiLabelF1Score(average="binary")
    metric2 = utils.MultiLabelF1Score()
    assert len(metric1(labels, predictions)) == 2
    assert len(metric2(labels, predictions)) == 2


def test_reset_weights():
    m = models.SSSNet()
    pc = copy.deepcopy(list(m.parameters()))
    assert np.all([torch.equal(pc[i], list(m.parameters())[i]) for i in range(len(pc))])
    utils.reset_weights(m)
    assert np.all(
        [~torch.equal(pc[i], list(m.parameters())[i]) for i in range(len(pc))]
    )
