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
Different validation styles exists depending whether leave-one-subject-out, leave-one-dataset-out or action units is used.

As parameter the *validator* takes an instance of the Config class. Typically it is created for each specific case, see [config](config.md).

Next the data and data frame are passed to the validator. These are obtained from the dataset object, see [datasets](datasets.md).

```python
model_outputs = validator.validate(df, data)
```
The `model_outputs` contains outputs of the model which can be used for computing metrics. For convenience, results are also printed out. For example:
```
Dataset: casme, n=189 | train_mean: 0.9866 | test_mean: 0.2145
Test per AU: [('AU1', 51.43), ('AU2', 71.43), ('AU4', 68.7), ('AU5', 0.0), ('AU6', 0.0), ('AU7', 0.0), ('AU9', 4.76), ('AU10', 0.0), ('AU12', 13.33), ('AU14', 18.75), ('AU15', 28.95), ('AU17', 0.0)]

Dataset: casme2, n=256 | train_mean: 0.9992 | test_mean: 0.415
Test per AU: [('AU1', 85.25), ('AU2', 70.37), ('AU4', 94.82), ('AU5', 0.0), ('AU6', 0.0), ('AU7', 56.14), ('AU9', 0.0), ('AU10', 11.76), ('AU12', 38.1), ('AU14', 54.84), ('AU15', 11.76), ('AU17', 75.0)]

.
.
.
Final results

All AUs:  [('AU1', 69.94), ('AU2', 64.86), ('AU4', 82.52), ('AU5', 6.13), ('AU6', 5.41), ('AU7', 33.26), ('AU9', 6.41), ('AU10', 4.94), ('AU12', 25.29), ('AU14', 33.82), ('AU15', 19.2), ('AU17', 48.89)]
Mean:  33.39
```

You may also want to validate for $n$ times to avoid choice of seed to affect results.

```python
validator.validate_n_times(df, data, n_times=5)
```
Outputs:
```
AUS: ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'Average']
68.7 & 65.2 & 83.8 & 8.3 & 6.4 & 39.6 & 11.0 & 6.8 & 31.0 & 29.6 & 8.8 & 39.6 & 33.2

Datasets:  ['casme', 'casme2', 'samm', 'fourd', 'mmew', 'casme3a', 'Average']
21.2 & 39.8 & 36.7 & 33.8 & 32.6 & 27.7 & 32.0
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
