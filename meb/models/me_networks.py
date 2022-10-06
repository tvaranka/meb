from typing import List

import torch.nn as nn
import torch.nn.functional as F
import torch


class SSSNet(nn.Module):
    """Shallow Single Stream network (SSSNet)

    Neural network model designed for micro-expression recognition.
    Uses two convolutional and two fully connected layers.

    Parameters
    ----------
    num_classes : int, default=3
        Size of the final fully connected layer, number of classes
        to be predicted.
    h_dims : sequence, default=[32, 64, 256]
        The dimensions of hidden layers.
    dropout : float, default=0.5
        Determines dropout value for both convolutional and fully
        connected layers.
    softmax: bool, default=False
        When true, softmax is applied after the first layer. Can
        Increase performance in some cases.

    References
    ----------
    :doi:`Varanka, T., Peng, W., and Zhao, G. (2021).
    "Micro-expression recognition with noisy labels".
    IS&T Int’l. Symp. on Electronic Imaging: Human Vision and
    Electronic Imaging, 33, 157-1 - 157-8.
    <10.2352/ISSN.2470-1173.2021.11.HVEI-157>`
    """
    def __init__(
        self,
        num_classes: int = 3,
        h_dims: List[int] = [32, 64, 256],
        dropout: float = 0.5,
        softmax=False,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        h1 = h_dims[0]
        h2 = h_dims[1]
        h3 = h_dims[2]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=h1, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bn1 = nn.BatchNorm2d(h1)
        self.drop1 = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(in_channels=h1, out_channels=h2, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(h2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(dropout)

        self.fc1 = nn.Linear(9 ** 2 * h2, h3)
        self.drop3 = nn.Dropout(dropout)
        self.fc = nn.Linear(h3, num_classes)
        self.softmax = None
        if softmax:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.drop1(self.bn1(self.pool(F.relu(self.conv1(x)))))
        x = self.drop2(self.bn2(self.pool2(F.relu(self.conv2(x)))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc(self.drop3(x))
        if self.softmax:
            x = self.softmax(x)
        return x


class STSTNet(nn.Module):
    """Shallow Triple Stream Three-dimensional network (SSSNet)

    Neural network model designed for micro-expression recognition.
    Uses three convolutional and one fully connected layers.

    Parameters
    ----------
    in_channels : int, default=3
        The number of channels of the input tensor.
    num_classes : int, default=3
        Size of the final fully connected layer, number of classes
        to be predicted.

    References
    ----------
    :doi:`Liong, S., Gan, Y.S. , See, J., Khor, H. , Huang, Y. (2019).
    "Shallow Triple Stream Three-dimensional CNN (STSTNet) for
    Micro-expression Recognition".
    IEEE International Conference on Automatic Face & Gesture Recognition, 14, 1-5.
    <10.1109/FG.2019.8756567>`
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 3, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=3, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels=5, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(5)
        self.bn3 = nn.BatchNorm2d(8)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=5 * 5 * 16, out_features=num_classes)

    def forward(self, x):
        x1 = self.dropout(self.maxpool(self.bn1(self.relu(self.conv1(x)))))
        x2 = self.dropout(self.maxpool(self.bn2(self.relu(self.conv2(x)))))
        x3 = self.dropout(self.maxpool(self.bn3(self.relu(self.conv3(x)))))
        x = torch.cat((x1, x2, x3), 1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class OffApexNet(nn.Module):
    """Off-ApexNet

    Neural network model designed for micro-expression recognition
    using optical flow of apex and onset. Consists of two convolutional
    layers and two fully connected layers.

    Parameters
    ----------
    num_classes : int, default=3
        Size of the final fully connected layer, number of classes
        to be predicted.
    dropout : float, default=0.5
        Determines dropout value for both convolutional and fully
        connected layers.

    References
    ----------
    :doi:`Liong, S., Gan, Y.S. , Yau, W., Huang, Y.,Ken, T.L. (2019).
    "OFF-ApexNet on micro-expression recognition system".
    Signal Processing: Image Communication, 74, 129-139.
    <10.1016/j.image.2019.02.005>`
    """
    def __init__(self, num_classes: int = 3, dropout: float = 0.5, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=6, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(7 ** 2 * 16, 1024)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(self.drop(x)))
        x = self.fc(self.drop2(x))
        return x
