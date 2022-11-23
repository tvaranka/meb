1. [Getting started](getting_started.md)
2. [Datasets](datasets.md)
3. [Config](config.md)
4. [Validator](validator.md)

# Training and validation
The training and validation uses the `core` of the library. A *validator* is created based on the validation style

```python
from meb import core

validator = core.MEGCValidator(core.Config)
```

As parameter the *validator* takes an instance of the Config class. Typically it is created for each specific case, see [config](config.md).

Next the data and data frame are passed to the validator. These are obtained from the dataset object, see (datasets)[datasets.md).

```python
model_outputs = validator.validate(df, data)
```

You may also want to validate for $n$ times to avoid choice of seed to affect results.

```python
model_outputs = validator.validate_n_times(df, data, n_times=5)
```

#### Validator
The *validator* uses a base class `Validator` that set ups methods such as `get_data_loader`,  `train_model` and `evaluate_model`, that work regardless whether you are using multi-class or multi-label. The `Validator` class is inherited by *validators* to specify the validation strategy (leave-one-subject-out, leave-one-dataset-out, multi-label, multi-class).


#### Custom validator
You can create custom *validators* to change the validation strategy or change the training if it doesn't fit your method. This can be done by inheriting a *validator* and modifying it.

```python
class CustomValidator(core.Validator):
    def __init__(self, config):
        super().__init__(config)
    def train_one_epoch(self, epoch: int, dataloader: torch.utils.data.Dataloader):
         # Custom implementation
         pass
```
Here is an example of creating a custom validator for hand-crafted methods for the cross-dataset validation protocol. As hand-crafted methods use pre-extracted features, we can use those as the input to our validator. Then we are only left with replacing a neural network model with a classifier. This example uses a classifier compatible with the Scikit-learn API. Only three functions need to modified.

```python 
class HandCraftedCrossValidator(core.CrossDatasetValidator):
    def __init__(self, config: core.Config, verbose: bool = True):
        super().__init__(config)
        self.verbose = verbose
        
    def train_model(
        self, train_data: np.ndarray, train_labels: np.ndarray
        ) -> None:
        self.model.fit(train_data, train_labels)
        
    def evaluate_model(
        self, data: np.ndarray, labels: np.ndarray, test: bool = False
        ) -> List[float] |Tuple[List[float] | torch.tensor]:
        predictions = self.model.predict(data)
        results = self.evaluation_fn(torch.tensor(labels), torch.tensor(predictions))
        if test:
            return results, torch.tensor(predictions)
        return results
        
    def validate_split(self, df: pd.DataFrame, input_data: np.ndarray, labels: np.ndarray, split_name: str):
        """Main setup of each split. Should be called by the overriden validate method."""
        train_data, train_labels, test_data, test_labels = self.split_data(
            df[self.split_column], input_data, labels, split_name
        )
        self.model = self.cf.model()

        self.train_model(train_data, train_labels)
        train_metrics = self.evaluate_model(train_data, train_labels)
        test_metrics, outputs_test = self.evaluate_model(test_data, test_labels, test=True)
        return train_metrics, test_metrics, outputs_test
```
#### What's next?
See [Experiments](../experiments) to find examples on how to use the library.

1. [Getting started](getting_started.md)
2. [Datasets](datasets.md)
3. [Config](config.md)
4. [Validator](validator.md)
