1. [Getting started](getting_started.md)
2. [Datasets](datasets.md)
3. [Config](config.md)
4. [Validator](validator.md)

# Config
Config is used to gather all information regarding the configuration of the experiment. Configs are python classes with the configuration details given as class attributes.

```python
class Config:
    batch_size = 64
```

Inheritance of classes is used to avoid writing commonly used configurations

```python
# Also contains all the attributes from Config
class ExtendedConfig(Config):
    epochs = 100
```
#### Passing objects to config
To avoid storing large objects in the `Config` object, the objects are initialized later. To pass parameters we use the `partial` function from functools. `partial` allows us to pass parameters to a function or object in multiple instances. This comes particularly handy when we want to define parameters for our optimizer in the config file, but the construction of the optimizer requires parameters of the network, which are not available at this stage. 

```python
from functools import partial
from torch import optim
optimizer_fn = partial(optim.Adam, lr=1e-4, weight_decay=1e-3)
...
# Later we can construct the optimizer with the network parameters
optimizer = optimizer_fn(model.parameters())
```

For certain objects, like the optimizer, this is needed as the parameters of the network are needed, which are only available after the model is created.

#### Default config
The default config to be inherited is defined in meb.core.

```python
class Config:
    action_units = utils.dataset_aus["cross"]
    print_loss_interval = None
    validation_interval = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 200
    criterion = utils.MultiLabelBCELoss
    evaluation_fn = partial(utils.MultiLabelF1Score, average="macro")
    # Optimizer
    optimizer = partial(optim.Adam, lr=1e-4, weight_decay=1e-3)
    scheduler = None
    # Dataloader
    batch_size = 32
    train_transform = {"spatial": None, "temporal": None}
    test_transform = {"spatial": None, "temporal": None}
    mixup_fn = None
    num_workers = 2
    model = None
```
#### Practical use
Let's create a `Config` using Resnet18 for action unit detection on SAMM. We define the `action_units` to be used. The number of `epochs` is overwritten to 50 and `batch_size` to 64. A transform is added and the model is added a partial, with the `pretrained` and `num_classes` parameters

```python
class ResNetConfig(core.Config):
    # My config
    action_units = utils.dataset_aus["samm"]
    epochs = 50
    batch_size = 64
    train_transform = {
      "spatial": transforms.Compose([
          transforms.RandomHorizontalFlip(0.5)
      ])
      "temporal": None
    }
    model = partial(models.resnet18, num_classes=len(action_units), pretrained=True)
```

1. [Getting started](getting_started.md)
2. [Datasets](datasets.md)
3. [Config](config.md)
4. [Validator](validator.md)
