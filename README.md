# MEB
![example workflow](https://github.com/tvaranka/meb/workflows/Python%20application/badge.svg)
[![contributions elcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/tvaranka/Cross-dataset-micro-expression/issues)

MEB aims to simplify, standardize and streamline the process of micro-expression analysis. It provides tools for data loading and training micro-expression models.

[ArXiv] [Data Leakage and Evaluation Issues in Micro-Expression Analysis](https://arxiv.org/abs/2211.11425)

## Getting started

The following example shows how to run a pretrained ResNet18 on the cross dataset protocol using optical flow as input.

```python
from meb import core, datasets
from functools import partial
from timm import models

# Load in the data
cross_dataset = datasets.CrossDataset(resize=112, optical_flow=True)

# Define configurations. Inherit core.Config for default configs
class ResNetConfig(core.Config):
    model = partial(models.resnet18, num_classes=len(core.Config.action_units), pretrained=True)

# Create a validator for cross-dataset protocol based on config
validator = core.CrossDatasetValidator(ResNetConfig)

# Train and test with the cross-dataset protocol
validator.validate(cross_dataset.data_frame, cross_dataset.data)
```
[Experiments](experiments) for jupyter notebook demos of using the library.

See [Getting started](docs/getting_started.md) for installing and adding datasets. See [Config](docs/config.md), [Datasets](docs/datasets.md) and [Validation](docs/validation.md) for understanding the pipeline.

## Installing
```shell
git clone https://github.com/tvaranka/meb
cd meb
pip install -e .
```

## Citation
```
@article{meb,
  author = {Varanka, Tuomas and Li, Yante and Peng, Wei and Zhao, Guoying},
  title = {Data Leakage and Evaluation Issues in Micro-Expression Analysis},
  journal={arXiv preprint arXiv:2211.11425},
  year = {2022},
}
```
