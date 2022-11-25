import os
from tqdm import tqdm
import shutil
import argparse

import pandas as pd

parser = argparse.ArgumentParser(description="Casme3 extraction.")
parser.add_argument(
    "--excel",
    type=str,
    help="Path to excel file, e.g., data/casme3/CAS(ME)3_part_A_v2.xls",
    required=True,
)
parser.add_argument(
    "--casme3_original",
    type=str,
    help=(
        "Path to the root of the dataset, e.g.,"
        " data/casme3/part_a/data/part_A_split/part_A/"
    ),
    required=True,
)
parser.add_argument(
    "--casme3_me",
    type=str,
    help="Path to the root of the new extracted me dataset, e.g., data/casme3_me_A/",
    required=True,
)

# Extract excel file and root location
args = parser.parse_args()
casme3_excel = args.excel
df = pd.read_excel(casme3_excel)
root = args.casme3_original
root_new = args.casme3_me

# Only consider micro-expressions
df = df[df["Expression type"] == "Micro-expression"].reset_index()


def remove_nums(string: str) -> str:
    return "".join([s for s in string if not s.isdigit()])


# For subjects with the same filename add number to seperate them
num = 2  # Start with 2 and leave 1 empty
for i in range(df.shape[0] - 1):
    # Check if this and next are the same or not
    if remove_nums(df.loc[i, "Filename"]) == df.loc[i + 1, "Filename"]:
        # Check if subject is the same
        if df.loc[i, "Subject"] == df.loc[i + 1, "Subject"]:
            df.loc[i + 1, "Filename"] = df.loc[i + 1, "Filename"] + str(num)
            num += 1
        else:
            num = 2
    else:
        num = 2


# Main loop for extracting samples
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    root_row = f"{row['Subject']}/{row['Filename']}/color"
    root_row_src = f"{row['Subject']}/{remove_nums(row['Filename'])}/color"
    missing_n = 0
    for f in range(row["Onset"], row["Offset"] + 1):
        fsrc = root + root_row_src + f"/{f}.jpg"
        fdst = root_new + root_row + f"/{f}.jpg"
        os.makedirs(os.path.dirname(fdst), exist_ok=True)
        # Try except block as there are frame drops
        try:
            shutil.copyfile(fsrc, fdst)
        except FileNotFoundError:
            missing_n += 1
    print(fdst, missing_n) if missing_n > 0 else None
