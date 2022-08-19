from typing import List
import numpy as np
import pandas as pd
from ..utils.utils import MultiLabelF1Score
import torch


def list_to_latex(values: list, round_decimals: int = 1) -> str:
    string_list = [str(round(v, round_decimals)) for v in values]
    return " & ".join(string_list)


def results_to_list(
    outputs_list,
    df: pd.DataFrame,
    action_units: np.ndarray,
    split: str = "dataset",
    evaluation=MultiLabelF1Score(average="macro")
):
    # Create a copy of action units
    aus = [i for i in action_units]
    # Per action units
    labels = np.array(df[action_units])
    f1_aus = evaluation(labels, torch.cat(outputs_list))
    f1_aus = [au_f1 * 100 for au_f1 in f1_aus]
    # Per split
    f1_splits = []
    for i, split_name in enumerate(df[split].unique()):
        df_split = df[df[split] == split_name]
        labels = np.array(df_split[action_units])
        f1_split = np.mean(evaluation(labels, outputs_list[i]))
        f1_splits.append(f1_split * 100)
    # Add average at the end of both
    f1_aus.append(np.mean(f1_aus))
    f1_splits.append(np.mean(f1_splits))
    return f1_aus, f1_splits


def results_to_latex(
    outputs_list: List[np.ndarray],
    df: pd.DataFrame,
    action_units: np.ndarray,
    round_decimals: int = 1
) -> None:
    aus = [i for i in action_units]
    aus.append("Average")
    dataset_names = df["dataset"].unique().tolist()
    dataset_names.append("Average")
    f1_aus, f1_datasets = results_to_list(outputs_list, df, action_units)
    # Turn list to latex
    f1_aus = list_to_latex(f1_aus, round_decimals=round_decimals)
    f1_datasets = list_to_latex(f1_datasets, round_decimals=round_decimals)
    print("AUS:", aus)
    print(f1_aus)
    print("\nDatasets: ", dataset_names)
    print(f1_datasets)
