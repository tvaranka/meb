from typing import List
import numpy as np
import pandas as pd
from utils.utils import MultiTaskF1
import torch


def list_to_latex(values: list) -> str:
    string_list = [str(round(v, 1)) for v in values]
    return " & ".join(string_list)


def results_to_latex(outputs_list: List[np.ndarray], df: pd.DataFrame, action_units: np.ndarray) -> None:
    # Create a copy of action units
    aus = [i for i in action_units]
    evaluation = MultiTaskF1(len(action_units))
    # Per action units
    labels = np.concatenate([np.expand_dims(df[au], 1) for au in aus], axis=1)
    f1_aus = evaluation(torch.cat(outputs_list), labels)
    f1_aus = [au_f1 * 100 for au_f1 in f1_aus]
    # Per dataset
    f1_datasets = []
    for i, dataset_name in enumerate(df["dataset"].unique()):
        df_dataset = df[df["dataset"] == dataset_name]
        labels = np.concatenate(
            [np.expand_dims(df_dataset[au], 1) for au in aus], axis=1
        )
        f1_dataset = np.mean(evaluation(outputs_list[i], labels))
        f1_datasets.append(f1_dataset * 100)
    # Add average at the end of both
    aus.append("Average")
    f1_aus.append(np.mean(f1_aus))
    dataset_names = df["dataset"].unique().tolist()
    dataset_names.append("Average")
    f1_datasets.append(np.mean(f1_datasets))
    # Turn list to latex
    f1_aus = list_to_latex(f1_aus)
    f1_datasets = list_to_latex(f1_datasets)
    print("AUS:", aus)
    print(f1_aus)
    print("\nDatasets: ", dataset_names)
    print(f1_datasets)
