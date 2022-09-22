from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Tuple, List, Union
from datetime import datetime

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from functools import partial
from torch import optim
from timm.scheduler.cosine_lr import CosineLRScheduler

from ..utils import utils
from ..datasets import latex_tools as lt


class Config:
    """
    Used to store config values.
    """
    print_loss_interval = None
    validation_interval = None
    device = torch.device("cuda:0")
    epochs = 200
    criterion = utils.MultiLabelBCELoss
    evaluation_fn = partial(utils.MultiLabelF1Score, average="macro")
    # Optimizer
    optimizer = partial(optim.AdamW, lr=5e-4, weight_decay=5e-2)
    scheduler = partial(CosineLRScheduler, t_initial=epochs, warmup_lr_init=1e-6, warmup_t=20, lr_min=1e-6)
    # Dataloader
    batch_size = 32
    train_transform = {"spatial": None, "temporal": None}
    test_transform = {"spatial": None, "temporal": None}
    mixup_fn = None


class Validation(ABC):
    """
    Abstract class for validation.
    """
    split_column: str

    def __init__(self, config: Config):
        self.cf = config
        self.disable_tqdm = False
        # Create evaluation_fn here as it is used outside validate_split function
        self.evaluation_fn = self.cf.evaluation_fn()

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
        dataset = utils.MEData(data, labels,
                               spatial_transform=transform["spatial"],
                               temporal_transform=transform["temporal"])
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=train, num_workers=0, pin_memory=True
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
        test_loader: torch.utils.data.DataLoader
    ) -> None:
        """Main training loop. Can be overriden for custom training loops."""
        for epoch in tqdm(range(self.cf.epochs), disable=self.disable_tqdm):
            self.train_one_epoch(epoch, train_loader)
            if self.scheduler:
                self.scheduler.step(epoch + 1)
            if self.cf.validation_interval:
                if (epoch + 1) % self.cf.validation_interval == 0:
                    train_f1 = self.evaluate_model(train_loader)
                    test_f1, outputs_test = self.evaluate_model(test_loader, test=True)
                    print("-" * 80)
                    print(
                        f"Validating at epoch {epoch + 1}\n"
                        f"Training F1-Score mean: {np.mean(train_f1):>6f}\n"
                        f"Test F1 per AU: {list(zip(self.cf.action_units, np.around(np.array(test_f1) * 100, 2)))}\n"
                        f"Testing F1-Score mean: {np.mean(test_f1):>6f}"
                    )
                    print("-" * 80)

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

    def validate_split(self, df: pd.DataFrame, input_data: np.ndarray, labels: np.ndarray, split_name: str):
        """Main setup of each split. Should be called by the overriden validate method."""
        train_data, train_labels, test_data, test_labels = self.split_data(
            df[self.split_column], input_data, labels, split_name
        )
        train_loader = self.get_data_loader(train_data, train_labels, train=True)
        test_loader = self.get_data_loader(test_data, test_labels, train=False)
        self.model = self.cf.model()
        self.criterion = self.cf.criterion()
        self.model.to(self.cf.device)
        self.optimizer = self.cf.optimizer(self.model.parameters())
        self.scheduler = self.cf.scheduler(self.optimizer) if self.cf.scheduler else None
        self.mixup_fn = self.cf.mixup_fn() if self.cf.mixup_fn else None

        self.train_model(train_loader, test_loader)
        train_f1 = self.evaluate_model(train_loader)
        test_f1, outputs_test = self.evaluate_model(test_loader, test=True)
        return train_f1, test_f1, outputs_test

    def evaluate_model(self, dataloader: torch.utils.data.DataLoader, test: bool = False
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
        result = self.evaluation_fn(labels, predictions)
        if test:
            return result, predictions
        return result


class CrossDatasetValidation(Validation):
    """
    Performs cross-dataset validation.
    """

    def __init__(self, config: Config, verbose: bool = True):
        super().__init__(config)
        self.verbose = verbose
        self.split_column = "dataset"

    def validate_n_times(self, df: pd.DataFrame, input_data: np.ndarray, n_times: int = 5) -> None:
        self.verbose = False
        self.disable_tqdm = True
        au_results = []
        dataset_results = []
        for n in tqdm(range(n_times)):
            outputs_list = self.validate(df, input_data, seed_n=n + 45)
            au_result, dataset_result = lt.results_to_list(
                outputs_list, df, self.cf.action_units, split=self.split_column
            )
            au_results.append(au_result)
            dataset_results.append(dataset_result)
        au_results = lt.list_to_latex(np.mean(au_results, axis=0))
        dataset_results = lt.list_to_latex(np.mean(dataset_results, axis=0))
        aus = [i for i in self.cf.action_units]
        dataset_names = df["dataset"].unique().tolist()
        aus.append("Average")
        dataset_names.append("Average")
        print("AUS:", aus)
        print(au_results)
        print("\nDatasets: ", dataset_names)
        print(dataset_results)

    def validate(self, df: pd.DataFrame, input_data: np.ndarray, seed_n: int = 1):
        utils.set_random_seeds(seed_n)
        dataset_names = df["dataset"].unique()
        # Create a boolean array with the AUs
        labels = np.concatenate([np.expand_dims(df[au], 1) for au in self.cf.action_units], axis=1)
        outputs_list = []
        for dataset_name in dataset_names:
            train_f1, test_f1, outputs_test = self.validate_split(df, input_data, labels, dataset_name)
            outputs_list.append(outputs_test)
            if self.verbose:
                print(
                    f"Dataset: {dataset_name}, n={outputs_test.shape[0]} | "
                    f"train_f1: {np.mean(train_f1):.4} | "
                    f"test_f1: {np.mean(test_f1):.4}"
                )
                print(f"Test F1 per AU: {list(zip(self.cf.action_units, np.around(np.array(test_f1) * 100, 2)))}\n")

        # Calculate total f1-scores
        predictions = torch.cat(outputs_list)
        f1_aus = self.evaluation_fn(labels, predictions)
        if self.verbose:
            print("All AUs: ", list(zip(self.cf.action_units, np.around(np.array(f1_aus) * 100, 2))))
            print("Mean f1: ", np.around(np.mean(f1_aus) * 100, 2))
        return outputs_list


class IndividualDatasetAUValidation(Validation):
    def __init__(self, config: Config, verbose: bool = True):
        super().__init__(config)
        self.verbose = verbose
        self.split_column = "subject"
        self.disable_tqdm = True

    def validate_n_times(self, df: pd.DataFrame, input_data: np.ndarray, n_times: int = 5) -> None:
        self.verbose = False
        au_results = []
        subject_results = []
        for n in tqdm(range(n_times)):
            outputs_list = self.validate(df, input_data, seed_n=n + 45)
            au_result, subject_result = lt.results_to_list(
                outputs_list, df, self.cf.action_units, split=self.split_column
            )
            au_results.append(au_result)
            subject_results.append(subject_result)
        au_results = lt.list_to_latex(np.mean(au_results, axis=0))
        subject_results = lt.list_to_latex(np.mean(subject_results, axis=0))
        aus = [i for i in self.cf.action_units]
        subject_names = df[self.split_column].unique().tolist()
        aus.append("Average")
        subject_names.append("Average")
        print("AUS:", aus)
        print(au_results)
        print("\nSubjects: ", subject_names)
        print(subject_results)

    def validate(self, df: pd.DataFrame, input_data: np.ndarray, seed_n: int = 1):
        utils.set_random_seeds(seed_n)
        subject_names = df["subject"].unique()
        labels = np.concatenate([np.expand_dims(df[au], 1) for au in self.cf.action_units], axis=1)
        outputs_list = []
        for subject_name in subject_names:
            train_f1, test_f1, outputs_test = self.validate_split(df, input_data, labels, subject_name)
            outputs_list.append(outputs_test)
            if self.verbose:
                print(
                    f"Subject: {subject_name}, n={outputs_test.shape[0]} | "
                    f"train_f1: {np.mean(train_f1):.4} | "
                    f"test_f1: {np.mean(test_f1):.4}"
                )
                print(f"Test F1 per AU: {list(zip(self.cf.action_units, np.around(np.array(test_f1) * 100, 2)))}\n")

        # Calculate total f1-scores
        predictions = torch.cat(outputs_list)
        f1_aus = self.evaluation_fn(labels, predictions)
        if self.verbose:
            print("All AUs: ", list(zip(self.cf.action_units, np.around(np.array(f1_aus) * 100, 2))))
            print("Mean f1: ", np.around(np.mean(f1_aus) * 100, 2))
        return outputs_list


class MEGCValidation(Validation):
    def __init__(self, config: Config, verbose: bool = True):
        super().__init__(config)
        self.verbose = verbose
        self.split_column = "subject"
        self.disable_tqdm = True

    def validate(self, df: pd.DataFrame, input_data: np.ndarray, seed_n: int = 1):
        utils.set_random_seeds(seed_n)
        subject_names = df["subject"].unique()
        le = LabelEncoder()
        labels = le.fit_transform(df["emotion"])
        outputs_list = []
        for subject_name in subject_names:
            train_f1, test_f1, outputs_test = self.validate_split(df, input_data, labels, subject_name)
            outputs_list.append(outputs_test)
            if self.verbose:
                print(
                    f"Subject: {subject_name}, n={outputs_test.shape[0]} | "
                    f"train_f1: {np.mean(train_f1):.4} | "
                    f"test_f1: {np.mean(test_f1):.4}"
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
