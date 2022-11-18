import argparse

from meb import datasets


parser = argparse.ArgumentParser(description="Test dataset setup.")
parser.add_argument(
    "--dataset_name",
    type=str,
    help=(
        "Dataset name from Smic/Casme/Casme2/Samm/Mmew/"
        "Fourd/Casme3A/Casme3C/Megc/CrossDataset"
    ),
    required=True,
)
parser.add_argument(
    "--data_type",
    type=str,
    help="Data type from original/cropped/optical_flow",
    default="original",
    required=False,
)


def load_dataset(dataset_name, data_type):
    """Loads dataset and tests sample"""
    dataset = getattr(datasets, dataset_name)
    if data_type == "original":
        sample = dataset(cropped=False).data[0]
    elif data_type == "cropped":
        sample = dataset().data[0]
    elif data_type == "optical_flow":
        sample = dataset(optical_flow=True).data[0]
    else:
        raise NotImplementedError
    print(sample.shape)


args = parser.parse_args()
load_dataset(args.dataset_name, args.data_type)
