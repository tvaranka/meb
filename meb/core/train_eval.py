from abc import ABC, abstractmethod
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import optim
from tqdm import tqdm

from meb import utils
from meb.datasets.dataset_utils import InputData
from meb.utils.utils import NullScaler


class Config:
    """Stores configuration settings.

    The class is used to store configuration settings for running experiments.
    Various objects require that they are passed as a class, rather than as an
    instance. This is required for optimizers and schedulers, but the convention
    is kept across other objects.

    Attributes
    ----------
    action_units : List[str], default=utils.dataset_aus["cross"]
        Defaults to the AUs used in cross-dataset evaluation. See
        utils.dataset_aus for the values. Should be set to None when using
        emotional labels.
    print_loss_interval : int, default=None
        How often loss is printed in terms of batch size. E.g., for a batch_size
        of 64 and a print_loss_interval value of 5, loss is printed every
        64 * 5 = 320 samples. None refers to not printed.
    validation_interval : int, default=None
        How often validation, both training and testing, is performed and printed
        in terms of epochs. None refers to not printed.
    device : torch.device, default=cuda:0 else cpu
        Defaults cuda:0 if available, otherwise to cpu.
    criterion : class, default=utils.MultiLabelBCELoss
        Criterion, i.e., the loss function. Should be multi-label if using AUs and
        multi-class if using emotions. Only the class or partial should be given, not
        an instance of it. The object needs to be callable.
    evaluation_fn : class, default=partial(utils.MultiLabelF1Score, average="macro")
        Evaluation function is used for calculating performance from predictions.
        Should be multi-label if using AUs and multi-class if using emotions. Only the
        class or partial should be given, not an instance of it. The object needs to
        be callable.
    optimizer : class, default=partial(optim.Adam, lr=1e-4, weight_decay=1e-3)
        Optimizer of the model. Only the class or partial should be given, not an
        instance of it. The object needs to be callable.
    scheduler : class, default=None
        Learning rate scheduler of the optimizer. Only the class or partial should
        be given, not an instance of it. The object needs to be callable.
    batch_size : int, default=32
        Batch size determines how many samples are included in a single batch.
    train_transform : dict, default={"spatial": None, "temporal": None}
        A dictionary with the spatial and temporal keys. If images, e.g., optical flow
        is used, only the spatial component needs to be defined. Transforms should be
        from the torchvision.transforms library. Temporal transforms determine
        sampling. See datasets.sampling for examples.
    test_transform : dict, default={"spatial": None, "temporal": None}
        A dictionary with the spatial and temporal keys. If images, e.g., optical flow
        is used, only the spatial component needs to be defined. Transforms should be
        from the torchvision.transforms library. Temporal transforms determine
        sampling. See datasets.sampling for examples.
    mixup_fn : class, default=None
        Mixup (or MixCut or VideoMix) uses a linear combination of samples and their
        labels to create new samples. See utils.mixup.
    num_workers : int, default=2
        Parameter for the torch DataLoader object. For cases with no transformations,
        0 may result in faster times.
    model : torch.nn.module, default=None
        A torch model used for training and evaluation. Only the class or partial
        should be given, not an instance of it. The object needs to be callable.
    channels_last : torch.memory_format, default=torch.contiguous_format
        Can be used to improve speed. Use torch.channels_last or torch.channels_last_3d
        for improved performance.
    loss_scaler : LossScaler, default=None
        When a scaler is provided AMP (automated mixed precision) is applied. Use
        for example torch.cuda.amp.GradScaler

    Examples
    --------
    >>> from meb import core
    >>> from functools import partial
    >>> import timm
    >>> class ResNetConfig(core.Config):
    ...     model = partial(
    ...         timm.models.resnet18,
    ...         num_classes=len(core.Config.action_units),
    ...         pretrained=True
    ...     )
    """

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


class Validator(ABC):
    """Abstract class for validator.

    Performs the training and evaluation.

    Parameters
    ----------
    config : Config
        Determines the configuration that is used during training.
    split_column : str
        Determines what to dataframe column to split on. Is typically either
        "subject" or "dataset".
    """

    def __init__(self, config: Config, split_column: str):
        self.cf = config
        self.disable_tqdm = False
        self.split_column = split_column
        # Create evaluation_fn here as it is used outside validate_split function
        if not isinstance(self.cf.evaluation_fn, Sequence):
            self.cf.evaluation_fn = [self.cf.evaluation_fn]

        self.evaluation_fn = utils.MultiMetric(self.cf.evaluation_fn)

        self.printer = utils.Printer(self.cf, split_column=self.split_column)

        self.amp_autocast = nullcontext
        if self.cf.loss_scaler:
            self.amp_autocast = torch.cuda.amp.autocast

        # Validate config
        utils.validate_config(self.cf, Config)

    @abstractmethod
    def validate(self, df: pd.DataFrame, input_data: InputData, seed_n: int = 1):
        """Main validation function

        This method should be overwritten based on the validation strategy.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe from the datasets module. Should contain at least the split
            column and emotional labels or AUs.
        input_data : InputData
            This is the data received from datasets. Should be an iterable.
        seed_n : int
            Determines what seed is used for randomness.

        Returns
        -------
        out : List[torch.tensor]
            List contains the outputs of each split from the model.
        """

    def get_data_loader(
        self, data: InputData, labels: np.ndarray, train: bool
    ) -> torch.utils.data.DataLoader:
        """Constructs pytorch dataloader

        Uses InputData to construct a pytorch dataloader and applies transforms.

        Parameters
        ----------
        data : InputData
            The raw data (RGB or optical flow)
        labels : np.ndarray
            Labels corresponding to the input data.
        train : bool
            When set to true applies train transforms, otherwise uses test transforms

        Returns
        -------
        out : torch.utils.data.Dataloader
            Returns a torch dataloader object for training
        """
        if train:
            transform = self.cf.train_transform
            batch_size = self.cf.batch_size
        else:
            transform = self.cf.test_transform
            # Divide by four to ensure not using too much memory
            batch_size = self.cf.batch_size // 4
        dataset = utils.MEData(
            data,
            labels,
            spatial_transform=transform["spatial"],
            temporal_transform=transform["temporal"],
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=self.cf.num_workers,
            pin_memory=True,
        )
        return dataloader

    @staticmethod
    def split_data(
        split_column: pd.Series, data: InputData, labels: np.ndarray, split_name: str
    ) -> Tuple[InputData, np.ndarray, InputData, np.ndarray]:
        """
        Splits data based on the split_column and split_name.
        E.g., df["dataset"] == "smic"
        """
        train_idx = split_column[split_column != split_name].index.tolist()
        test_idx = split_column[split_column == split_name].index.tolist()
        return data[train_idx], labels[train_idx], data[test_idx], labels[test_idx]

    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> None:
        """Main training loop

        The main training loop that controls the scheduler and printing.
        This function can be overwritten for custom schedulers, printing,
        etc.
        """
        for epoch in tqdm(range(self.cf.epochs), disable=self.disable_tqdm):
            self.train_one_epoch(epoch, train_loader)
            if self.scheduler:
                self.scheduler.step(epoch + 1)
            if self.cf.validation_interval:
                if (epoch + 1) % self.cf.validation_interval == 0:
                    train_metrics = self.evaluate_model(train_loader)
                    test_metrics, outputs_test = self.evaluate_model(
                        test_loader, test=True
                    )
                    self.printer.print_train_test_validation(
                        train_metrics, test_metrics, epoch
                    )

    def train_one_epoch(self, epoch: int, dataloader: torch.utils.data.DataLoader):
        """Train model for single epoch

        Given the epoch number, trains the model for a single epoch. This function
        can be overwritten to create custom training loops.
        """
        num_updates = epoch * len(dataloader)
        for i, (X, y) in enumerate(dataloader):
            X = X.to(self.cf.device, memory_format=self.cf.channels_last)
            y = y.to(self.cf.device)
            self.optimizer.zero_grad()
            if self.mixup_fn:
                X, y = self.mixup_fn(X.float(), y.float())
            with self.amp_autocast():
                outputs = self.model(X.float())
                loss = self.criterion(outputs, y)
            self.loss_scaler.scale(loss).backward()
            self.loss_scaler.step(self.optimizer)
            self.loss_scaler.update()
            num_updates += 1
            if self.scheduler:
                self.scheduler.step_update(num_updates=num_updates)
            if self.cf.print_loss_interval:
                if i % self.cf.print_loss_interval == 0:
                    print(
                        f"{datetime.now()} - INFO - Epoch"
                        f" [{epoch + 1}/{self.cf.epochs}][{i + 1}/{len(dataloader)}]"
                        f" lr: {self.optimizer.param_groups[0]['lr']:>6f}, loss:"
                        f" {loss.item():>7f}"
                    )

    def setup_training(self) -> None:
        """Setup training components

        Sets up the training modules, including model, criterion, optimizer, scheduler
        and mixup. This function can be overwritten to setup custom objects.
        """
        self.model = self.cf.model()
        self.criterion = self.cf.criterion()
        self.model.to(self.cf.device, memory_format=self.cf.channels_last)
        self.optimizer = self.cf.optimizer(self.model.parameters())
        self.scheduler = (
            self.cf.scheduler(self.optimizer) if self.cf.scheduler else None
        )
        self.mixup_fn = self.cf.mixup_fn() if self.cf.mixup_fn else None
        self.loss_scaler = (
            self.cf.loss_scaler() if self.cf.loss_scaler else NullScaler()
        )

    def validate_split(
        self,
        df: pd.DataFrame,
        input_data: InputData,
        labels: np.ndarray,
        split_name: str,
    ):
        """Splits data and begins training

        Splits data according to the split and starts the evaluation process.
        """

        # Load data
        train_data, train_labels, test_data, test_labels = self.split_data(
            df[self.split_column], input_data, labels, split_name
        )
        train_loader = self.get_data_loader(train_data, train_labels, train=True)
        test_loader = self.get_data_loader(test_data, test_labels, train=False)

        # Setup model
        self.setup_training()

        # Train and evaluation
        self.train_model(train_loader, test_loader)
        train_metrics = self.evaluate_model(train_loader)
        test_metrics, outputs_test = self.evaluate_model(test_loader, test=True)
        return train_metrics, test_metrics, outputs_test

    def evaluate_model(
        self, dataloader: torch.utils.data.DataLoader, test: bool = False
    ) -> Union[List[float], Tuple[List[float], torch.tensor]]:
        """
        Evaluates the model given a dataloader and an evaluation function. Returns
        the evaluation result and if boolean test is set to true also the
        predictions.
        """
        self.model.eval()
        outputs_list = []
        labels_list = []
        with torch.no_grad():
            for batch in dataloader:
                data_batch = batch[0].to(self.cf.device)
                labels_batch = batch[1]
                outputs = self.model(data_batch.float())
                outputs_list.append(outputs.detach().cpu())
                labels_list.append(labels_batch)
        self.model.train()
        predictions = torch.cat(outputs_list)
        labels = torch.cat(labels_list)
        results = self.evaluation_fn(labels, predictions)
        if test:
            return results, predictions
        return results


class CrossDatasetValidator(Validator):
    """Validator for cross-dataset protocol

    Expects multiple datasets with 'dataset' column in the dataframe and action units
    as labels. A leave-one-dataset-out protocol is performed, where a single dataset
    is used as the testing and the rest as training data. This is performed such that
    all datasets are the testing data once.
    """

    __doc__ += Validator.__doc__

    def __init__(self, config: Config, verbose: bool = True):
        super().__init__(config, split_column="dataset")
        self.verbose = verbose

    def validate_n_times(
        self, df: pd.DataFrame, input_data: InputData, n_times: int = 5
    ) -> None:
        self.disable_tqdm = True
        au_results = []
        dataset_results = []
        for n in tqdm(range(n_times)):
            outputs_list = self.validate(df, input_data, seed_n=n + 45)
            au_result, dataset_result = self.printer.results_to_list(outputs_list, df)
            au_results.append(au_result)
            dataset_results.append(dataset_result)

        aus = [i for i in self.cf.action_units]
        dataset_names = df["dataset"].unique().tolist()
        aus.append("Average")
        dataset_names.append("Average")
        au_results = np.array(au_results)
        dataset_results = np.array(dataset_results)
        for i in range(len(self.cf.evaluation_fn)):
            if len(self.cf.evaluation_fn) > 1:
                print(self.printer.metric_name(self.cf.evaluation_fn[i]))
            au_result = self.printer.list_to_latex(list(au_results[:, i].mean(axis=0)))
            dataset_result = self.printer.list_to_latex(
                list(dataset_results[:, i].mean(axis=0))
            )
            print("AUS:", aus)
            print(au_result)
            print("\nDatasets: ", dataset_names)
            print(dataset_result)

    def validate(
        self, df: pd.DataFrame, input_data: InputData, seed_n: int = 1
    ) -> List[torch.tensor]:
        utils.set_random_seeds(seed_n)
        dataset_names = df["dataset"].unique()
        # Create a boolean array with the AUs
        labels = np.array(df[self.cf.action_units])
        outputs_list = []
        for dataset_name in dataset_names:
            train_metrics, test_metrics, outputs_test = self.validate_split(
                df, input_data, labels, dataset_name
            )
            outputs_list.append(outputs_test)
            if self.verbose:
                self.printer.print_train_test_evaluation(
                    train_metrics, test_metrics, dataset_name, outputs_test.shape[0]
                )

        # Calculate total f1-scores
        predictions = torch.cat(outputs_list)
        metrics = self.evaluation_fn(labels, predictions)
        if self.verbose:
            self.printer.print_test_validation(metrics)
        return outputs_list


class IndividualDatasetAUValidator(Validator):
    """Validator for single dataset AU"""

    __doc__ += Validator.__doc__

    def __init__(self, config: Config, verbose: bool = True):
        super().__init__(config, split_column="subject")
        self.verbose = verbose
        self.disable_tqdm = False

    def validate_n_times(
        self, df: pd.DataFrame, input_data: InputData, n_times: int = 5
    ) -> None:
        self.verbose = False
        au_results = []
        subject_results = []
        for n in tqdm(range(n_times)):
            outputs_list = self.validate(df, input_data, seed_n=n + 45)
            au_result, subject_result = self.printer.results_to_list(outputs_list, df)
            au_results.append(au_result)
            subject_results.append(subject_result)

        aus = [i for i in self.cf.action_units]
        subject_names = df[self.split_column].unique().tolist()
        aus.append("Average")
        subject_names.append("Average")
        au_results = np.array(au_results)
        subject_results = np.array(subject_results)
        for i in range(len(self.cf.evaluation_fn)):
            if len(self.cf.evaluation_fn) > 1:
                print(self.printer.metric_name(self.cf.evaluation_fn[i]))
            au_result = self.printer.list_to_latex(list(au_results[:, i].mean(axis=0)))
            subject_result = self.printer.list_to_latex(
                list(subject_results[:, i].mean(axis=0))
            )
            print("AUS:", aus)
            print(au_result)
            print("\nSubjects: ", subject_names)
            print(subject_result)

    def validate(
        self, df: pd.DataFrame, input_data: InputData, seed_n: int = 1
    ) -> List[torch.tensor]:
        utils.set_random_seeds(seed_n)
        subject_names = df["subject"].unique()
        labels = np.concatenate(
            [np.expand_dims(df[au], 1) for au in self.cf.action_units], axis=1
        )
        outputs_list = []
        for subject_name in subject_names:
            train_metrics, test_metrics, outputs_test = self.validate_split(
                df, input_data, labels, subject_name
            )
            outputs_list.append(outputs_test)
            if self.verbose:
                self.printer.print_train_test_evaluation(
                    train_metrics, test_metrics, subject_name, outputs_test.shape[0]
                )

        # Calculate total f1-scores
        predictions = torch.cat(outputs_list)
        metrics = self.evaluation_fn(labels, predictions)
        if self.verbose:
            self.printer.print_test_validation(metrics)
        return outputs_list


class MEGCValidator(Validator):
    """MEGC composite dataset validator"""

    __doc__ += Validator.__doc__

    def __init__(self, config: Config, verbose: bool = True):
        super().__init__(config, split_column="subject")
        self.verbose = verbose
        self.disable_tqdm = False

    def validate(
        self, df: pd.DataFrame, input_data: InputData, seed_n: int = 1
    ) -> List[torch.tensor]:
        utils.set_random_seeds(seed_n)
        subject_names = df["subject"].unique()
        le = LabelEncoder()
        labels = le.fit_transform(df["emotion"])
        outputs_list = []
        for subject_name in subject_names:
            train_metrics, test_metrics, outputs_test = self.validate_split(
                df, input_data, labels, subject_name
            )
            outputs_list.append(outputs_test)
            if self.verbose:
                self.printer.print_train_test_evaluation(
                    train_metrics, test_metrics, subject_name, outputs_test.shape[0]
                )

        # Calculate total f1-scores
        predictions = torch.cat(outputs_list)
        f1_total = self.evaluation_fn(labels, predictions)
        idx = df["dataset"] == "smic"
        f1_smic = self.evaluation_fn(labels[idx], predictions[idx])
        idx = df["dataset"] == "casme2"
        f1_casme2 = self.evaluation_fn(labels[idx], predictions[idx])
        idx = df["dataset"] == "samm"
        f1_samm = self.evaluation_fn(labels[idx], predictions[idx])
        if self.verbose:
            print(
                "Total f1: {}, SMIC: {}, CASME2: {}, SAMM: {}".format(
                    f1_total, f1_smic, f1_casme2, f1_samm
                )
            )
        return outputs_list
