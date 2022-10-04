from typing import Any, Union

import torch.nn as nn


def multi_task(model: nn.Module) -> nn.Module:
    class MultiTaskModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.model = model(*args, **kwargs)
            MultiTaskModel.__name__ = model.__name__
            self.task_num = kwargs.pop("task_num", "")
            if self.task_num:
                MultiTaskModel.__name__ = f"Multi-task {model.__name__}"
                self.task_num = self.task_num
                in_features = self.model.fc.in_features
                self.model.fc = nn.Identity()
                self.fcs = nn.ModuleList(
                    [nn.Linear(in_features, 2) for _ in range(self.task_num)]
                )

        def forward(self, x):
            if self.task_num:
                x = self.model(x)
                x = [fc(x) for fc in self.fcs]
            else:
                x = self.model(x)
            return x

    return MultiTaskModel
