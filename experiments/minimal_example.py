import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score

from meb import datasets
from meb import models

# This scripts provides a minimal example of running a cross-dataset evaluation using
# mostly standard pytorch. MEB is used for data loading and exporting a standard
# pytorch model.

# Note: This example runs the experiment once. To ensure reproducibility the experiments
# should be run 5 times with different seeds and then taking the mean of the results

# Define constants
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 1

# Ensure reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data loading still uses MEB for convenience
c = datasets.CrossDataset(optical_flow=True, resize=64)
# df is a pandas data frame and data is a numpy array
df, data = c.data_frame, c.data
# >>> df.shape
# (2031, 46)
# >>> data.shape
# (2031, 3, 64, 64)

# Model still uses MEB for convenience (can be easily replaced, as this is just
# a standard pytorch model).
# Custom models can be placed in here or a separate custom file

# Action units to be used
action_units = [
    "AU1",
    "AU2",
    "AU4",
    "AU5",
    "AU6",
    "AU7",
    "AU9",
    "AU10",
    "AU12",
    "AU14",
    "AU15",
    "AU17",
]


class MEData(Dataset):
    def __init__(
        self,
        frames: np.ndarray,
        labels: np.ndarray,
        transform=None,
    ):
        self.frames = frames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        sample = self.frames[idx]
        if self.transform:
            sample = torch.tensor(sample)
            sample = self.transform(sample)

        label = self.labels[idx]
        return sample, label


labels = np.array(df.loc[:, action_units])  # Extract labels from the dataframe
criterion = nn.BCEWithLogitsLoss()  # Define loss function
outputs_list = []  # Collect outputs from each fold

# Main cross-validation loop
# >>> df["dataset"].unique()
# ["Casme", "CASME2", "SAMM", "MMEW", "FOURD", "CASME3A"]
for dataset_name in tqdm(df["dataset"].unique()):
    # Split data based on the dataset name to training and testing
    train_idx = df["dataset"] != dataset_name
    test_idx = df["dataset"] == dataset_name
    train_X = data[train_idx]
    train_y = labels[train_idx]
    test_X = data[test_idx]
    test_y = labels[test_idx]

    train_data = MEData(train_X, train_y, transform=None)
    test_data = MEData(test_X, test_y, transform=None)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE // 4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Setup training
    model = models.SSSNet(num_classes=len(action_units))
    model.to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Training
    for epoch in range(EPOCHS):
        for X, y in train_dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X.float())
            loss = criterion(outputs, y.float())
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    split_outputs_list = []
    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(DEVICE)
            outputs = model(X.float())
            split_outputs_list.append(outputs.detach().cpu())

    # Get metrics for each dataset
    outputs = torch.cat(split_outputs_list)
    predictions = torch.where(outputs > 0, 1, 0)
    split_f1 = [
        f1_score(test_y[:, i], np.array(predictions[:, i]), average="binary")
        for i in range(outputs.shape[1])
    ]
    print(split_f1)
    print(np.mean(split_f1))
    # Store outputs from each fold for final F1-score computation
    outputs_list.append(outputs)

# Use the outputs from all folds to compute the total F1-Score
outputs = torch.cat(outputs_list)
# A threshold of 0 is used for predictions
predictions = torch.where(outputs > 0, 1, 0)
# Compute binary F1 for each AU
final_f1 = [
    f1_score(labels[:, i], np.array(predictions[:, i]), average="binary")
    for i in range(outputs.shape[1])
]

print(final_f1)
print(np.mean(final_f1))
