# Getting started

##### Requires Python >= 3.10

#### Create a virtual environment
```shell
conda create --name meb_env python=3.10
```

#### Clone the library

```shell
git clone https://github.com/tvaranka/meb
```

#### Build

```shell
cd meb
pip install -e .
```

#### Adding datasets
Due to private data (faces) appearing in micro-expression datasets they are only available after accepting the release agreement. See [Micro-expression datasets](micro_expression_datasets.md) for more information on where to obtain the datasets. 

The datasets can be added to the framework by two ways.
##### 1. Modifying the path in `config/dataset_config.py`

The `DatasetConfig` consists of all the paths to the datasets. Each dataset has four different paths: `excel_path`, `cropped_dataset_path`, `dataset_path` and `optical_flow` paths. (TODO: create scripts that create the cropped and optical flow datasets) The variable names in the `DatasetConfig` should not be changed, but the paths can be changed freely.

```python
class DatasetConfig:
    """Used as a config file for storing dataset information"""

    smic_excel_path = "data/SMIC/smic.xlsx"
    smic_cropped_dataset_path = "data/SMIC/SMIC_all_cropped/HS"
    smic_dataset_path = "data/SMIC/HS"
    smic_optical_flow = "data/SMIC/smic_uv_frames_secrets_of_OF.npy"
  
    ...
```
**OR**
##### 2. Using symbolic links
Create symbolic links to match the ones in `config/dataset_config.py`. Change the paths starting with "my_path" to match yours. This example uses linux, use `mklink /D {my_path} {link_path}` for windows.
```shell
ln -sfn "my_path/micro_expressions/SMIC/Original data/smic.xlsx" "data/SMIC/smic.xlsx"
ln -sfn "my_path/micro_expressions/SMIC/Preprocessed data/SMIC_all_cropped/HS" "data/SMIC/SMIC_all_cropped/HS"
ln -sfn "my_path/micro_expressions/SMIC/Original data/HS" "data/SMIC/HS"
ln -sfn "my_path/micro_expressions/SMIC/Preprocessed data/smic_uv_frames_secrets_of_OF.npy" "data/SMIC/smic_uv_frames_secrets_of_OF.npy"
```
See `create_sym_links.sh` for the whole script.

#### Run the getting_started.ipynb/py
```shell
python experiments/getting_started.py
```

Browse the `experiments` folder to find more examples and see documentation on [datasets](datasets.md), [configs](config.md) and [validation](validation.md).

