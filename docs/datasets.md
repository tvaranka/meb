1. [Getting started](getting_started.md)
2. [Datasets](datasets.md)
3. [Config](config.md)
4. [Validator](validator.md)

# Data loading
See [Getting started](getting_started.md) to first add datasets. Datasets are loaded as *dataset* objects. For example 

```python
from meb import datasets
casme2_dataset = datasets.Casme2()
```


The `casme2_dataset` is a dataset object with mainly two important methods: *data_frame* and *data*.

#### data_frame
By calling `casme2_dataset.data_frame` we are calling a method by the same name, which returns a pandas data frame. For convenience we are using a python wrapper `@property`, such that the *data_frame* can be accessed as a variable rather than a method. Further, `@cached_property` is used to cache the data frame, to avoid loading the data from memory multiple times.

```python
# Pandas data frame object
df = casme2_dataset.data_frame
```

#### data
Similar to *data_frame*, `casme2_dataset.data` is a using a cached property wrapper for convenience. *data* can return a few different things, depending on parameters passed to the class constructor. Without any arguments the default *data* returns a *LazyDataLoader*, which is an iterator over the requested dataset, but the videos are loaded to the memory upon request. Another option is the *LoadedDataLoader*, which preloads the data to the memory when creating the *dataset* object. For optical flow, the *data* simply returns a numpy array.

```python
# Iterable dataloader object
data_loader = casme2_dataset.data
```


## Parameters
We can pass a few different parameters to the *dataset* object.

```python
color: bool = False,
resize: Union[Sequence[int], int, None] = None,
cropped: bool = True,
optical_flow: bool = False,
magnify: bool = False,
n_sample: int = 6,
preload: bool = False,
ignore_validation: bool = False,
magnify_params: dict = {}
```

###### color
Boolean, whether to return the dataset in color or not. For SAMM, which is only available in grayscale, the videos are duplicated to match the 3 channels.

###### resize
Resizes the images accordingly. `resize=None` does nothing and images are returned in their original size. By passing an int `resize=128` resizes the images to (128, 128). By passing in a sequence (list or tuple) `resize=(140, 170)` the image is is resized accordingly.

###### cropped
Whether the cropped images or uncropped images are used. The location of the cropped dataset needs to be defined in dataset_config.py for this to work.

###### optical_flow
Returns optical flow. Similar to cropped, the location of the optical flow dataset needs to be defined in dataset_config.py. *data* now returns a numpy array.

###### magnify
Uses Eulerian motion magnification on the images. Default parameters to magnification $\alpha = 20, r_1 = 0.4, r_2 = 0.05$.

###### n_sample
Subsamples the videos uniformly to `n_sample` amount.

###### preload
Preloads the data to the memory during the dataset object generation. *data* now returns a *LoadedDataLoader*.

###### ignore_validation
Dataset is validated when creating an instance to see whether the data frame and the loaded data match. Setting this to True ignores it the procedure.


###### magnify_params
Pass magnification parameters to the motion magnification.


## Examples
```python
# MEGC optical flow resized to 64
megc = datasets.MEGC(resize=64, optical_flow)

# Cross-dataset RGB resized to 112 and preloaded for fast training
cross_dataset = datasets.CrossDataset(resize=112, color=True, preload=True)

# Casme2 Gray magnified and not cropped or resized
casme2 = datasets.Casme2(
    magnify=True,
    magnify_params={"a": 10, "r_1": 0.5, "r_2": 0.2"}
)
```

1. [Getting started](getting_started.md)
2. [Datasets](datasets.md)
3. [Config](config.md)
4. [Validator](validator.md)

