from dataclasses import dataclass
import pickle
from typing import Tuple

import torch
import torch.utils.data

__all__ = ["DatasetConfig", "get_dataset_config", "load_dataset"]


@dataclass
class DatasetConfig(object):
    """Dataset configuration.

    Attributes:
        dataset_name (str): Name of the dataset.
        dataset_path (str): Path to the dataset.
        n_tasks (int): Number of tasks.
        n_classes_per_task (int): Number of classes per task.
        input_shape (Tuple[int, int, int]): Input shape of each image.
    """
    dataset_name: str
    dataset_path: str
    n_tasks: int
    n_classes_per_task: int
    input_shape: Tuple[int, int, int]


DATASET_FACTORY = {
    "mnist": DatasetConfig("mnist", "data/multi_mnist.pickle", 2, 10, (1, 36, 36)),
    "fashion": DatasetConfig("fashion", "data/multi_fashion.pickle", 2, 10, (1, 36, 36)),
    "fashion_and_mnist": DatasetConfig("fashion_and_mnist", "data/multi_fashion_and_mnist.pickle", 2, 10, (1, 36, 36)),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    if dataset_name not in DATASET_FACTORY:
        raise ValueError(f"Dataset {config.dataset_name} is not supported!")
    return DATASET_FACTORY[dataset_name]


def load_dataset(
    config: DatasetConfig, batch_size: int = 256, num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return train/test data loaders.

    Args:
        config (DatasetConfig): Dataset configuration.
        batch_size (int): Dataloader batch size.
        num_workers (int): Dataloader workers.

    Returns:
        (torch.utils.data.DataLoader): Train dataloader.
        (torch.utils.data.DataLoader): Test dataloader.
    """
    if config.dataset_name in ["mnist", "fashion", "fashion_and_mnist"]:
        with open(config.dataset_path, "rb") as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

        trainX = torch.from_numpy(trainX.reshape(120000, *config.input_shape)).float()
        trainLabel = torch.from_numpy(trainLabel).long()
        testX = torch.from_numpy(testX.reshape(20000, *config.input_shape)).float()
        testLabel = torch.from_numpy(testLabel).long()

        train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
        test_set = torch.utils.data.TensorDataset(testX, testLabel)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
    else:
        raise ValueError(f"Dataset {config.dataset_name} is not supported!")

    print(f"Total training iterations: {len(train_loader)}")
    print(f"Total testing iterations: {len(test_loader)}")
    return train_loader, test_loader
