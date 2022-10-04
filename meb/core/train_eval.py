from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Tuple, List, Union, Sequence
from datetime import datetime

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from functools import partial
from torch import optim
from timm.scheduler.cosine_lr import CosineLRScheduler

from meb import utils


class Config:
    """
    Used to store config values.
    """

    action_units = utils.dataset_aus["cross"]
    print_loss_interval = None
    validation_interval = None
    device = torch.device("cuda:0")
    epochs = 200
    criterion = utils.MultiLabelBCELoss
    evaluation_fn = partial(utils.MultiLabelF1Score, average="macro")
    # Optimizer
    optimizer = partial(optim.AdamW, lr=5e-4, weight_decay=5e-2)
    scheduler = partial(
        CosineLRScheduler,
        t_initial=epochs,
        warmup_lr_init=1e-6,
        warmup_t=20,
        lr_min=1e-6,
    )
    # Dataloader
    batch_size = 32
    train_transform = {"spatial": None, "temporal": None}
    test_transform = {"spatial": None, "temporal": None}
    mixup_fn = None
    num_workers = 2


class Validation(ABC):
    """
    Abstract class for validation.
    """

    def __init__(self, config: Config, split_column: str):
        self.cf = config
        self.disable_tqdm = False
        self.split_column = split_column
        # Create evaluation_fn here as it is used outside validate_split function
        if not isinstance(self.cf.evaluation_fn, Sequence):
            self.cf.evaluation_fn = [self.cf.evaluation_fn]

        self.evaluation_fn = utils.MultiMetric(self.cf.evaluation_fn)

        if self.cf.action_units:
            label_type = "au"
        else:
            label_type = "emotion"
        self.printer = utils.Printer(
            self.cf, label_type=label_type, split_column=self.split_column
        )

    @abstractmethod
    def validate(self, df: pd.DataFrame, input_data: np.ndarray, seed_n: int = 1):
        """Overridable method that defines how the validation is done."""

    def get_data_loader(
        self, data: np.ndarray, labels: np.ndarray, train: bool
    ) -> torch.utils.data.DataLoader:
        """Constructs pytorch dataloader from numpy data."""
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
        split_column: pd.Series, data: np.ndarray, labels: np.ndarray, split_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits data based on the split_column and split_name. E.g., df["dataset"] == "smic"
        """
        train_idx = split_column[split_column != split_name].index.tolist()
        test_idx = split_column[split_column == split_name].index.tolist()
        return data[train_idx], labels[train_idx], data[test_idx], labels[test_idx]

    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> None:
        """Main training loop. Can be overriden for custom training loops."""
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
        num_updates = epoch * len(dataloader)
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(self.cf.device), y.to(self.cf.device)
            self.optimizer.zero_grad()
            if self.mixup_fn:
                X, y = self.mixup_fn(X.float(), y.float())
            outputs = self.model(X.float())
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            num_updates += 1
            if self.scheduler:
                self.scheduler.step_update(num_updates=num_updates)
            if self.cf.print_loss_interval:
                if i % self.cf.print_loss_interval == 0:
                    print(
                        f"{datetime.now()} - INFO - Epoch "
                        f"[{epoch + 1}/{self.cf.epochs}][{i + 1}/{len(dataloader)}] "
                        f"lr: {self.optimizer.param_groups[0]['lr']:>6f}, loss: {loss.item():>7f}"
                    )

    def setup_training(self) -> None:
        """
        Sets up the training modules, including model, criterion, optimizer, scheduler and mixup.
        """
        self.model = self.cf.model()
        self.criterion = self.cf.criterion()
        self.model.to(self.cf.device)
        self.optimizer = self.cf.optimizer(self.model.parameters())
        self.scheduler = (
            self.cf.scheduler(self.optimizer) if self.cf.scheduler else None
        )
        self.mixup_fn = self.cf.mixup_fn() if self.cf.mixup_fn else None

    def validate_split(
        self,
        df: pd.DataFrame,
        input_data: np.ndarray,
        labels: np.ndarray,
        split_name: str,
    ):
        """Main setup of each split. Should be called by the overriden validate method."""

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


class CrossDatasetValidation(Validation):
    """
    Performs cross-dataset validation.
    """

    def __init__(self, config: Config, verbose: bool = True):
        super().__init__(config, split_column="dataset")
        self.verbose = verbose

    def validate_n_times(
        self, df: pd.DataFrame, input_data: np.ndarray, n_times: int = 5
    ) -> None:
        self.verbose = False
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

    def validate(self, df: pd.DataFrame, input_data: np.ndarray, seed_n: int = 1):
        utils.set_random_seeds(seed_n)
        dataset_names = df["dataset"].unique()
        # Create a boolean array with the AUs
        labels = np.concatenate(
            [np.expand_dims(df[au], 1) for au in self.cf.action_units], axis=1
        )
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


class IndividualDatasetAUValidation(Validation):
    def __init__(self, config: Config, verbose: bool = True):
        super().__init__(config, split_column="subject")
        self.verbose = verbose
        self.disable_tqdm = True

    def validate_n_times(
        self, df: pd.DataFrame, input_data: np.ndarray, n_times: int = 5
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
            subject_results = self.printer.list_to_latex(
                list(subject_results[:, i].mean(axis=0))
            )
            print("AUS:", aus)
            print(au_results)
            print("\nSubjects: ", subject_names)
            print(subject_results)

    def validate(self, df: pd.DataFrame, input_data: np.ndarray, seed_n: int = 1):
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


class MEGCValidation(Validation):
    def __init__(self, config: Config, verbose: bool = True):
        super().__init__(config, split_column="subject")
        self.verbose = verbose
        self.disable_tqdm = False

    def validate(self, df: pd.DataFrame, input_data: np.ndarray, seed_n: int = 1):
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
