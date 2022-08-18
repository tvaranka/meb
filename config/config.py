import torch
import torch.optim as optim
from functools import partial
from utils.models import models as models


class config:
    device = torch.device("cuda:0")
    epochs = 200
    # Optimizer
    learning_rate = 1e-2
    weight_decay = 1e-3
    optimizer = partial(optim.SGD, momentum=0.9)
    scheduler = None
    # Dataloader
    batch_size = 128
    train_transform = None
    test_transform = None
    model = models.SSSNet(h_dims=[16, 32, 128])