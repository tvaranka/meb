# Micro-expression-cross-dataset
![example workflow](https://github.com/tvaranka/Cross-dataset-micro-expression/workflows/Python%20application/badge.svg)
[![contributions elcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/tvaranka/Cross-dataset-micro-expression/issues)

## Installing

## Getting started

The following example shows how to run a pretrained ResNet18 on the cross dataset protocol using optical flow as input.

```python
from meb import core
from meb import datasets
from meb import utils
from functools import partial
from timm import models

cross_dataset = datasets.CrossDataset(resize=112, optical_flow=True)

class ResNetConfig(core.Config):
    action_units = utils.dataset_aus["cross"]
    model = partial(models.resnet18, num_classes=len(action_units), pretrained=True)
  
validator = core.CrossDatasetValidator(ResNetConfig)

validator.validate(cross_dataset.data_frame, cross_dataset.data)
```

Citation
