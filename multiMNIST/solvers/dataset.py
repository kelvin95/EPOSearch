import torch
import torch.utils.data
from torch.utils.data import DataLoader
import pickle
from typing import Tuple


def load_dataset(dataset: str) -> Tuple[DataLoader, DataLoader]:
    # LOAD DATASET
    # ------------
    # MultiMNIST: multi_mnist.pickle
    if dataset == "mnist":
        with open("data/multi_mnist.pickle", "rb") as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)
    # MultiFashionMNIST: multi_fashion.pickle
    elif dataset == "fashion":
        with open("data/multi_fashion.pickle", "rb") as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)
    # Multi-(Fashion+MNIST): multi_fashion_and_mnist.pickle
    elif dataset == "fashion_and_mnist":
        with open("data/multi_fashion_and_mnist.pickle", "rb") as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)
    else:
        raise ValueError(f"dataset: {dataset} is not valid!")

    trainX = torch.from_numpy(trainX.reshape(120000, 1, 36, 36)).float()
    trainLabel = torch.from_numpy(trainLabel).long()
    testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
    testLabel = torch.from_numpy(testLabel).long()

    train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
    test_set = torch.utils.data.TensorDataset(testX, testLabel)

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )

    print("Total training iterations (number of batches): {}".format(len(train_loader)))
    print("Total testing iterations: {}".format(len(test_loader)))
    # ---------***---------

    return train_loader, test_loader
