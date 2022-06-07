from functools import partial
from abc import ABC, abstractmethod
from tqdm import tqdm

import numpy as np
import torch.nn as nn
import torch
import pandas as pd
import utils.utils as utils
import utils.latex_tools as lt
from typing import Callable, Tuple, List, Union
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


def megc_validation(
    input_data: np.ndarray,
    df: pd.DataFrame,
    cf
) -> List[np.ndarray]:
    utils.set_random_seeds()
    le = LabelEncoder()
    labels = le.fit_transform(df["emotion"])
    criterion = nn.CrossEntropyLoss()
    evaluation = partial(f1_score, average="macro")
    model = cf.model.to(cf.device)
    outputs_list = []
    for subject in df["subject"].unique():
        train_data, train_labels, test_data, test_labels = split_data(
            df["subject"], input_data, labels, subject
        )
        train_loader = get_data_loader(
            train_data, train_labels, transform=cf.train_transform, train=True, batch_size=cf.batch_size
        )
        test_loader = get_data_loader(
            test_data, test_labels, transform=cf.test_transform, train=False, batch_size=cf.batch_size
        )
        model.apply(utils.reset_weights)
        optimizer = cf.optimizer(model.parameters(), lr=cf.learning_rate, weight_decay=cf.weight_decay)

        train_model(model, train_loader, criterion, optimizer, cf.epochs, cf.device)
        train_f1 = evaluate_model(model, train_loader, evaluation_func=evaluation, device=cf.device)
        test_f1, outputs_test = evaluate_model(
            model, test_loader, evaluation_func=evaluation, test=True, device=cf.device
        )
        outputs_list.append(outputs_test)
        print(
            f"Subject: {subject}, n={test_data.shape[0]} | "
            f"train_f1: {np.mean(train_f1):.5} | "
            f"test_f1: {np.mean(test_f1):.5}"
        )
    # Calculate total f1-scores
    predictions = torch.cat(outputs_list)
    f1_total = evaluation(labels, predictions)
    idx = df["dataset"] == "smic"
    f1_smic = evaluation(labels[idx], predictions[idx])
    idx = df["dataset"] == "casme2"
    f1_casme2 = evaluation(labels[idx], predictions[idx])
    idx = df["dataset"] == "samm"
    f1_samm = evaluation(labels[idx], predictions[idx])
    print(
        "Total f1: {}, SMIC: {}, CASME2: {}, SAMM: {}".format(
            f1_total, f1_smic, f1_casme2, f1_samm
        )
    )
    return outputs_list


class BasicConfig:
    """
    Used to store config values.
    """


class Validation(ABC):
    """
    Abstract class for validation.
    """

    def __init__(self, config: BasicConfig):
        self.cf = config

    @abstractmethod
    def validate(self, df: pd.DataFrame, input_data: np.ndarray, seed_n: int = 1):
        pass

    def get_data_loader(
            self, data: np.ndarray, labels: np.ndarray, train: bool
    ) -> torch.utils.data.DataLoader:
        transform = self.cf.train_transform if train else self.cf.test_transform
        dataset = utils.MEData(data, labels, transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cf.batch_size, shuffle=train, num_workers=0, pin_memory=True
        )
        return dataloader

    @staticmethod
    def split_data(
            split_column: pd.Series, data: np.ndarray, labels: np.ndarray, split_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits data based on the split_column and split_name. E.g., df["dataset"] == "smic"
        """
        train_idx = split_column[split_column != split_name].index
        test_idx = split_column[split_column == split_name].index
        return data[train_idx], labels[train_idx], data[test_idx], labels[test_idx]

    def train_model(self, dataloader: torch.utils.data.DataLoader) -> None:
        for epoch in range(self.cf.epochs):
            for batch in dataloader:
                data_batch, labels_batch = batch[0].to(self.cf.device), batch[1].to(self.cf.device)
                self.optimizer.zero_grad()

                outputs = self.cf.model(data_batch.float())
                loss = self.cf.criterion(outputs, labels_batch.long())
                loss.backward()
                self.optimizer.step()

    @abstractmethod
    def evaluate_model(self, dataloader: torch.utils.data.DataLoader, test: bool = False):
        pass


class CrossDatasetValidation(Validation):
    """
    Performs cross-dataset validation.
    """

    def __init__(self, config: BasicConfig, verbose: bool = True):
        number_of_tasks = len(config.action_units)
        super().__init__(config)
        self.verbose = True

    def validate_n_times(self, df: pd.DataFrame, input_data: np.ndarray, n_times: int = 5) -> None:
        self.verbose = False
        au_results = []
        dataset_results = []
        for n in tqdm(range(n_times)):
            outputs_list = self.validate(df, input_data, seed_n=n + 45)
            au_result, dataset_result = lt.results_to_list(outputs_list, df, self.config.action_units)
            au_results.append(au_result)
            dataset_results.append(dataset_result)
        au_results = lt.list_to_latex(np.mean(au_results, axis=0))
        dataset_results = lt.list_to_latex(np.mean(dataset_results, axis=0))
        aus = [i for i in self.config.action_units]
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
        number_of_tasks = labels.shape[1]
        outputs_list = []
        model = self.cf.model.to(self.cf.device)
        for dataset_name in dataset_names:
            train_f1, test_f1, outputs_test = self.validate_dataset(df, input_data, labels, dataset_name)
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
        f1_aus = self.cf.evaluation(labels, predictions)
        if self.verbose:
            print("All AUs: ", list(zip(self.cf.action_units, np.around(np.array(f1_aus) * 100, 2))))
            print("Mean f1: ", np.around(np.mean(f1_aus) * 100, 2))
        return outputs_list

    def validate_dataset(self, df: pd.DataFrame, input_data: np.ndarray, labels: np.ndarray, dataset_name: str):
        train_data, train_labels, test_data, test_labels = self.split_data(
            df["dataset"], input_data, labels, dataset_name
        )
        train_loader = self.get_data_loader(train_data, train_labels, train=True)
        test_loader = self.get_data_loader(test_data, test_labels, train=False)
        self.cf.model.apply(utils.reset_weights)
        self.optimizer = self.cf.optimizer(
            self.cf.model.parameters(), lr=self.cf.learning_rate, weight_decay=self.cf.weight_decay
        )

        self.train_model(train_loader)

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
        self.cf.model.eval()
        outputs_list = []
        labels_list = []
        for batch in dataloader:
            data_batch = batch[0].to(self.cf.device)
            labels_batch = batch[1]
            outputs = self.cf.model(data_batch.float())
            outputs_list.append([output.detach().cpu() for output in outputs])
            labels_list.append(labels_batch)
        self.cf.model.train()
        predictions = self.outputs_list_to_predictions(outputs_list)
        labels = torch.cat(labels_list)
        result = self.cf.evaluation(labels, predictions)
        if test:
            return result, predictions
        return result

    @staticmethod
    def outputs_list_to_predictions(outputs_list: List[List[torch.tensor]]) -> torch.tensor:
        """
        Turns the output lists from multi-task format to a single tensor predictions
        """
        predictions = torch.cat(
            [
                torch.tensor(
                    [torch.max(au_output, 1)[1].tolist() for au_output in split_output]
                ).T
                for split_output in outputs_list
            ]
        )
        return predictions


def cross_dataset_validation(
    input_data: np.ndarray,
    df: pd.DataFrame,
    cf
) -> List[np.ndarray]:
    """
    Cross dataset evaluation
    """
    utils.set_random_seeds(4)
    dataset_names = df["dataset"].unique()
    # Create a boolean array with the AUs
    labels = np.concatenate([np.expand_dims(df[au], 1) for au in cf.action_units], axis=1)
    number_of_tasks = labels.shape[1]
    outputs_list = []
    criterion = utils.MultiTaskLoss(number_of_tasks)
    evaluation = utils.MultiTaskF1(number_of_tasks)
    model = cf.model.to(cf.device)
    for dataset_name in dataset_names:
        train_data, train_labels, test_data, test_labels = split_data(
            df["dataset"], input_data, labels, dataset_name
        )
        train_loader = get_data_loader(
            train_data, train_labels, transform=cf.train_transform, train=True, batch_size=cf.batch_size
        )
        test_loader = get_data_loader(
            test_data, test_labels, transform=cf.test_transform, train=False, batch_size=cf.batch_size
        )
        model.apply(utils.reset_weights)
        optimizer = cf.optimizer(model.parameters(), lr=cf.learning_rate, weight_decay=cf.weight_decay)

        train_model(model, train_loader, criterion, optimizer, epochs=cf.epochs, device=cf.device)

        train_f1 = evaluate_model_multi_task(model, train_loader, evaluation, cf.device)
        test_f1, outputs_test = evaluate_model_multi_task(
            model, test_loader, evaluation, test=True, device=cf.device
        )
        outputs_list.append(outputs_test)
        print(
            f"Dataset: {dataset_name}, n={test_data.shape[0]} | "
            f"train_f1: {np.mean(train_f1):.4} | "
            f"test_f1: {np.mean(test_f1):.4}"
        )
        print(f"Test F1 per AU: {list(zip(cf.action_units, np.around(np.array(test_f1) * 100, 2)))}\n")
    # Calculate total f1-scores
    predictions = torch.cat(outputs_list)
    f1_aus = evaluation(labels, predictions)
    print("All AUs: ", list(zip(cf.action_units, np.around(np.array(f1_aus) * 100, 2))))
    print("Mean f1: ", np.around(np.mean(f1_aus) * 100, 2))
    return outputs_list


def get_data_loader(
    data: np.ndarray, labels: np.ndarray, transform, train: bool, batch_size: int
) -> torch.utils.data.DataLoader:
    dataset = utils.MEData(data, labels, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=0, pin_memory=True
    )
    return dataloader


def train_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion,
    optimizer,
    epochs: int,
    device
) -> None:
    for epoch in range(epochs):
        for batch in dataloader:
            data_batch, labels_batch = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            outputs = model(data_batch.float())
            loss = criterion(outputs, labels_batch.long())
            loss.backward()
            optimizer.step()


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    evaluation_func: Callable[[torch.tensor, torch.tensor], Union[List[float], float]],
    device: torch.device,
    test: bool = False,
) -> Union[float, Tuple[float, torch.tensor]]:
    """
    Evaluates the model given a dataloader and an evaluation function. Returns
    the evaluation result and if boolean test is set to true also the
    predictions.
    """
    model.eval()
    outputs_list = []
    labels_list = []
    for batch in dataloader:
        data_batch = batch[0].to(device)
        labels_batch = batch[1]
        outputs = model(data_batch.float())
        outputs_list.append(outputs)
        labels_list.append(labels_batch)
    model.train()
    outputs = torch.cat(outputs_list).cpu().detach()
    _, predictions = outputs.max(1)
    labels = torch.cat(labels_list)
    result = evaluation_func(labels, predictions)
    if test:
        return result, predictions
    return result


def evaluate_model_multi_task(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    evaluation_func: Callable[[torch.tensor, torch.tensor], Union[List[float], float]],
    device: torch.device,
    test: bool = False,
) -> Union[List[float], Tuple[List[float], torch.tensor]]:
    """
    Evaluates the model given a dataloader and an evaluation function. Returns
    the evaluation result and if boolean test is set to true also the
    predictions.
    """
    model.eval()
    outputs_list = []
    labels_list = []
    for batch in dataloader:
        data_batch = batch[0].to(device)
        labels_batch = batch[1]
        outputs = model(data_batch.float())
        outputs_list.append([output.detach().cpu() for output in outputs])
        labels_list.append(labels_batch)
    model.train()
    predictions = outputs_list_to_predictions(outputs_list)
    labels = torch.cat(labels_list)
    result = evaluation_func(labels, predictions)
    if test:
        return result, predictions
    return result


def split_data(
    split_column: pd.Series, data: np.ndarray, labels: np.ndarray, split_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data based on the split_column and split_name. E.g., df["dataset"] == "smic"
    """
    train_idx = split_column[split_column != split_name].index
    test_idx = split_column[split_column == split_name].index
    return data[train_idx], labels[train_idx], data[test_idx], labels[test_idx]


def outputs_list_to_predictions(outputs_list: List[List[torch.tensor]]) -> torch.tensor:
    predictions = torch.cat(
        [
            torch.tensor(
                [torch.max(au_output, 1)[1].tolist() for au_output in split_output]
            ).T
            for split_output in outputs_list
        ]
    )
    return predictions
