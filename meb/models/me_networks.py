from typing import List

import torch.nn as nn
import torch.nn.functional as F
import torch


class SSSNet(nn.Module):
    def __init__(self, output_size: int = 3, h_dims: List[int] = [32, 64, 256],
                 dropout: float = 0.5, softmax=False, **kwargs):
        super().__init__()
        self.output_size = output_size
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
        self.fc = nn.Linear(h3, output_size)
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
    def __init__(self, in_channels=3, out_channels=3, **kwargs):
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
        self.fc = nn.Linear(in_features=5 * 5 * 16, out_features=out_channels)

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
    def __init__(self, output_size: int = 3, dropout: float = 0.5, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(7 ** 2 * 16, 1024)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, output_size)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(self.drop(x)))
        x = self.fc(self.drop2(x))
        return x