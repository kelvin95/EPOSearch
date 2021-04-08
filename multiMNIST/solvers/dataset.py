from dataclasses import dataclass
import pickle
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.utils.data
import torchvision

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
    "celeba": DatasetConfig("celeba", "data/celeba", 5, 1, (3, 64, 64)),  # TODO(amanjit): Change path or download from torchvision
    "cifar10": DatasetConfig("cifar10", "/scratch/ssd001/datasets/cifar10/", 10, 1, (3, 32, 32)),
    "cifar100": DatasetConfig("cifar100", "/scratch/ssd001/datasets/cifar100/", 5, 20, (3, 32, 32)),
}


class MTLCIFAR100(torch.utils.data.Dataset):
    """MTL CIFAR-100 dataset.

    We randomly assign each of the 100 fine labels into n_coarse_labels labels.
    Repeating this n_tasks times, we get n_tasks number of tasks. Then the goal
    of the MTL learner is to classify the images according to each of the coarse
    label groups.

    Args:
        dataset_path (str): Path to the dataset.
        train (bool): Whether to load the train vs test split.
        n_tasks (int): Number of tasks to construct.
        n_coarse_labels (int): Number of classes per task.
    """

    CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    def __init__(self, dataset_path: str, train: bool, tasks: List[List[int]]) -> None:
        super(MTLCIFAR100, self).__init__()
        transform = self._build_transform(train)
        self.dataset = torchvision.datasets.CIFAR100(dataset_path, train, transform=transform)
        self.tasks = tasks

    @classmethod
    def build_tasks(cls, n_tasks: int, n_coarse_labels: int) -> List[List[int]]:
        coarse_labels = []
        current_coarse_label = 0
        for _ in range(100):
            coarse_labels.append(current_coarse_label)
            current_coarse_label = (current_coarse_label + 1) % n_coarse_labels

        tasks = []
        for _ in range(n_tasks):
            np.random.shuffle(coarse_labels)
            tasks.append(np.copy(coarse_labels))
        return tasks

    def _build_transform(self, train: bool) -> Callable:
        transforms = []
        if train:
            transforms.append(torchvision.transforms.RandomCrop(32, padding=4))
            transforms.append(torchvision.transforms.RandomHorizontalFlip())
        transforms.append(torchvision.transforms.ToTensor())
        transforms.append(torchvision.transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD))
        return torchvision.transforms.Compose(transforms)

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Return one MTL example.

        Args:
            index (int): The desired index.

        Returns:
            (torch.FloatTensor): [3 x width x height] image tensor.
            (torch.LongTensor): [num_tasks] label tensor.
        """
        image, label = self.dataset[index]
        mtl_labels = []
        for task_id in range(len(self.tasks)):
            mtl_labels.append(self.tasks[task_id][label])
        mtl_labels = torch.tensor(mtl_labels, dtype=torch.long)
        return image, mtl_labels

    def __len__(self) -> int:
        return len(self.dataset)


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    # HACK: get celeba dataset with variable number of attributes
    if "celeba-" in dataset_name and dataset_name not in DATASET_FACTORY:
        celeba_config = DATASET_FACTORY["celeba"]
        num_attributes = int(dataset_name.split("-")[1])
        dataset_config = DatasetConfig(
            dataset_name, celeba_config.dataset_path, num_attributes, 1, celeba_config.input_shape
        )
        return dataset_config

    if dataset_name not in DATASET_FACTORY:
        raise ValueError(f"Dataset {config.dataset_name} is not supported!")
    return DATASET_FACTORY[dataset_name]


def load_dataset(
    config: DatasetConfig, batch_size: int = 256, num_workers: int = 4,
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

        train_dataset = torch.utils.data.TensorDataset(trainX, trainLabel)
        test_dataset = torch.utils.data.TensorDataset(testX, testLabel)
    elif "celeba" in config.dataset_name:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(config.input_shape[1:]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        if config.dataset_name == "celeba":
            # BlackHair, Moustache, Male, MouthSlightlyOpen, Young
            target_transform = lambda x: x[[8, 22, 20, 39, 21]]
        else:
            target_transform = lambda x: x[:config.n_tasks]

        train_dataset = torchvision.datasets.CelebA(
            config.dataset_path, target_type="attr", split="train", transform=transform, target_transform=target_transform
        )
        test_dataset = torchvision.datasets.CelebA(
            config.dataset_path, target_type="attr", split="valid", transform=transform, target_transform=target_transform
        )
    elif config.dataset_name == "cifar10":
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(MTLCIFAR100.CIFAR_MEAN, MTLCIFAR100.CIFAR_STD)
            ]
        )
        test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(MTLCIFAR100.CIFAR_MEAN, MTLCIFAR100.CIFAR_STD)
            ]
        )
        target_transform = lambda x: torch.nn.functional.one_hot(torch.tensor(x), 10)
        train_dataset = torchvision.datasets.CIFAR10(config.dataset_path, True, train_transform, target_transform)
        test_dataset = torchvision.datasets.CIFAR10(config.dataset_path, False, test_transform, target_transform)
    elif config.dataset_name == "cifar100":
        tasks = MTLCIFAR100.build_tasks(config.n_tasks, config.n_classes_per_task)
        train_dataset = MTLCIFAR100(config.dataset_path, True, tasks)
        test_dataset = MTLCIFAR100(config.dataset_path, False, tasks)
    else:
        raise ValueError(f"Dataset {config.dataset_name} is not supported!")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    print(f"Total training iterations: {len(train_loader)}")
    print(f"Total testing iterations: {len(test_loader)}")
    return train_loader, test_loader
