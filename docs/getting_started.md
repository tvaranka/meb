# Getting started

#### Clone the library

```shell
git clone https://github.com/tvaranka/Cross-dataset-micro-expression
```

#### Build

```shell
cd Cross-dataset-micro-expression
pip install .
```

#### Adding datasets
Add datasets, this is the tedious part.  There are two ways to achieve this
1. Modifying the path in `config/dataset_config.py`
2. Create symbolic links to match the ones in `config/dataset_config.py`. See `create_sym_links.sh` example script for this

#### Run the getting_started.ipynb/py
```shell
python experiments/getting_started.py
```

Browse the `experiments` folder to find more examples and see documentation on [datasets](datasets.md), [configs](config.md) and [validation](validation.md).

