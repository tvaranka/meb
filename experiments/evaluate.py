"""
This script provides use of command line arguments and a python script for MEB.
The script can be modified and the command line arguments ignored.
"""
import argparse
from functools import partial

import torch
from torch import optim

from meb import core, datasets, models, utils

# Inspiration from TIMM
# https://github.com/rwightman/pytorch-image-models/blob/main/train.py
parser = argparse.ArgumentParser(description="Micro-expression training and testing")

# Dataset
group = parser.add_argument_group("Dataset parameters")
group.add_argument(
    "--dataset",
    "-d",
    default="CrossDataset",
    help="Dataset type: see meb/datasets/datasets.py",
)
group.add_argument(
    "--protocol",
    "-p",
    default="CrossDataset",
    help="Protocol type: see meb/core/train_eval.py",
)
group.add_argument(
    "--optical_flow", default=False, help="Load data as rgb or optical flow."
)
group.add_argument(
    "--resize", default=None, help="Resize dataset before training. int or tuple."
)
group.add_argument("--color", default=False, help="Load data in RGB or grayscale.")
group.add_argument("--cropped", default=True, help="Load data as cropped or not.")

# Model parameters
group = parser.add_argument_group("Model parameters")
group.add_argument(
    "--model", "-m", default="", help="Model used for training: see meb/models."
)
group.add_argument(
    "--pretrained", default=True, help="Use pretrained model if available."
)
group.add_argument(
    "--num_classes",
    default=None,
    help="Number of classes or number of labels in multi-label.",
)


# Optimizer parameters
group = parser.add_argument_group("Optimizer parameters")
group.add_argument("--optimizer", "-o", default="Adam", help="Optimizer to be used.")
group.add_argument(
    "--lr", default=1e-4, help="Learning rate of the optimizer at start."
)
group.add_argument("--weight_decay", default=1e-3, help="Weight decay of optimizer.")

# Miscellaneous parameters
group = parser.add_argument_group("Miscellaneous parameters")
group.add_argument(
    "--device", default="cuda:0", help="Device to be used, default cuda:0."
)
group.add_argument("--batch_size", default=64, help="Batch size of training samples.")
group.add_argument("--num_workers", default=2, help="Num workers used in data loading")
group.add_argument("--epochs", default=200, help="Number of epochs to train for.")
group.add_argument(
    "--loss",
    default="MultiLabelBCELoss",
    help="Loss function used in training: see meb/utils",
)


args = parser.parse_args()


def main():
    num_classes = (
        args.num_classes if args.num_classes else len(core.Config.action_units)
    )

    class Config(core.Config):
        device = torch.device(args.device)
        batch_size = args.batch_size
        num_workers = args.num_workers
        epochs = args.epochs
        criterion = getattr(utils, args.loss)
        optimizer = partial(
            getattr(optim, args.optimizer), lr=args.lr, weight_decay=args.weight_decay
        )
        model = partial(
            getattr(models, args.model),
            num_classes=num_classes,
            pretrained=args.pretrained,
        )

    dataset = getattr(datasets, args.dataset)
    c = dataset(
        optical_flow=args.optical_flow,
        color=args.color,
        resize=args.resize,
        cropped=args.cropped,
    )
    validator = getattr(core, args.protocol + "Validator")(Config)
    validator.validate_n_times(c.data_frame, c.data)


if __name__ == "__main__":
    main()
