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
    evaluation_fn = partial(utils.MultiLabelF1Score, average="binary")
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
    channels_last = torch.contiguous_format
    loss_scaler = None
```

Attributes
----------

##### action_units : List[str], default=utils.dataset_aus["cross"]
Defaults to the AUs used in cross-dataset evaluation. See
utils.dataset_aus for the values. Should be set to None when using
emotional labels.
    
##### print_loss_interval : int, default=None
How often loss is printed in terms of batch size. E.g., for a batch_size
of 64 and a print_loss_interval value of 5, loss is printed every
64 * 5 = 320 samples. None refers to not printed.
    
##### validation_interval : int, default=None
How often validation, both training and testing, is performed and printed
in terms of epochs. None refers to not printed.
    
##### device : torch.device, default=cuda:0 else cpu
Defaults cuda:0 if available, otherwise to cpu.
    
##### criterion : class, default=utils.MultiLabelBCELoss
Criterion, i.e., the loss function. Should be multi-label if using AUs and
multi-class if using emotions. Only the class or partial should be given, not
an instance of it. The object needs to be callable.
    
##### evaluation_fn : class, default=partial(utils.MultiLabelF1Score, average="macro")
Evaluation function is used for calculating performance from predictions.
Should be multi-label if using AUs and multi-class if using emotions. Only the
class or partial should be given, not an instance of it. The object needs to
be callable.
    
##### optimizer : class, default=partial(optim.Adam, lr=1e-4, weight_decay=1e-3)
Optimizer of the model. Only the class or partial should be given, not an
instance of it. The object needs to be callable.
    
##### scheduler : class, default=None
Learning rate scheduler of the optimizer. Only the class or partial should
be given, not an instance of it. The object needs to be callable.
    
##### batch_size : int, default=32
Batch size determines how many samples are included in a single batch.
    
##### train_transform : dict, default={"spatial": None, "temporal": None}
A dictionary with the spatial and temporal keys. If images, e.g., optical flow
is used, only the spatial component needs to be defined. Transforms should be
from the torchvision.transforms library. Temporal transforms determine
sampling. See datasets.sampling for examples.
    
##### test_transform : dict, default={"spatial": None, "temporal": None}
A dictionary with the spatial and temporal keys. If images, e.g., optical flow
is used, only the spatial component needs to be defined. Transforms should be
from the torchvision.transforms library. Temporal transforms determine
sampling. See datasets.sampling for examples.
    
##### mixup_fn : class, default=None
Mixup (or MixCut or VideoMix) uses a linear combination of samples and their
labels to create new samples. See utils.mixup.
    
##### num_workers : int, default=2
Parameter for the torch DataLoader object. For cases with no transformations,
0 may result in faster times.
    
##### model : torch.nn.module, default=None
A torch model used for training and evaluation. Only the class or partial
should be given, not an instance of it. The object needs to be callable.
    
##### channels_last : torch.memory_format, default=torch.contiguous_format
Can be used to improve speed. Use torch.channels_last or torch.channels_last_3d
for improved performance.
    
##### loss_scaler : LossScaler, default=None
When a scaler is provided AMP (automated mixed precision) is applied. Use
for example torch.cuda.amp.GradScaler
        
        
#### Practical use
Let's create a `Config` using Resnet18 for action unit detection on SAMM. We define the `action_units` to be used. The number of `epochs` is overwritten to 50 and `batch_size` to 64. A transform is added and the model is added as a partial, with the `pretrained` and `num_classes` parameters

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
