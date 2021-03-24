import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data
from time import time
import pickle
from typing import List, Dict, Tuple, Callable

from model_lenet import RegressionModel, RegressionTrain

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_float("lr", 1e-3, "learning rate", lower_bound=0.0)
flags.DEFINE_integer("epochs", 100, "number of training epochs", lower_bound=0)
flags.DEFINE_integer("n_tasks", 2, "number of tasks", lower_bound=2, upper_bound=2)
flags.DEFINE_multi_enum(
    "dataset",
    "all",
    ["mnist", "fashion", "fashion_and_mnist", "all"],
    "name of dataset to use",
)
# flags.DEFINE_enum(
#     "arch", "lenet", ["lenet", "resnet18"], "network architecture to use"
# )
flags.DEFINE_enum(
    "solver",
    "individual",
    [
        "graddrop",
        "gradnorm",
        "individual",
        "itmtl",
        "linscalar",
        "mgda",
        "pcgrad",
        "pmtl",
    ],
    "name of method/solver",
)

if FLAGS.dataset == "all" or (
    isinstance(FLAGS.dataset, list) and "all" in FLAGS.dataset
):
    FLAGS.dataset = ["mnist", "fashion", "fashion_and_mnist"]
else:
    # unique while preserving order passed in on cmdline
    FLAGS.dataset = list(dict.fromkeys(FLAGS.dataset))


def train(
    dataset: str, preference: np.ndarray, update_fn: Callable
) -> Tuple[Dict[str, List[float]], Dict[str, nn.Parameter]]:
    """Train a model on dataset and return a tuple (results, trained_weights)"""

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

    print("==>>> total trainning batch number: {}".format(len(train_loader)))
    print("==>>> total testing batch number: {}".format(len(test_loader)))
    # ---------***---------

    # DEFINE MODEL
    # ---------------------
    model = RegressionTrain(RegressionModel(FLAGS.n_tasks), preference)

    if torch.cuda.is_available():
        model.cuda()
    # ---------***---------

    # DEFINE OPTIMIZERS
    # -----------------
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.0)
    # ---------***---------

    # CONTAINERS FOR KEEPING TRACK OF PROGRESS
    # ----------------------------------------
    task_train_losses = []
    train_accs = []
    # ---------***---------

    # TRAIN
    # -----
    for t in range(FLAGS.epochs):

        model.train()
        for (it, batch) in enumerate(train_loader):

            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            update_fn(X, ts, model, optimizer, FLAGS)

        # Calculate and record performance
        if t == 0 or (t + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                total_train_loss = []
                train_acc = []

                correct1_train = 0
                correct2_train = 0

                for (it, batch) in enumerate(test_loader):

                    X = batch[0]
                    ts = batch[1]
                    if torch.cuda.is_available():
                        X = X.cuda()
                        ts = ts.cuda()

                    valid_train_loss = model(X, ts)
                    total_train_loss.append(valid_train_loss)
                    output1 = model.model(X).max(2, keepdim=True)[1][:, 0]
                    output2 = model.model(X).max(2, keepdim=True)[1][:, 1]
                    correct1_train += output1.eq(ts[:, 0].view_as(output1)).sum().item()
                    correct2_train += output2.eq(ts[:, 1].view_as(output2)).sum().item()

                train_acc = np.stack(
                    [
                        1.0 * correct1_train / len(test_loader.dataset),
                        1.0 * correct2_train / len(test_loader.dataset),
                    ]
                )

                total_train_loss = torch.stack(total_train_loss)
                average_train_loss = torch.mean(total_train_loss, dim=0)

            # record and print
            if torch.cuda.is_available():

                task_train_losses.append(average_train_loss.data.cpu().numpy())
                train_accs.append(train_acc)

                print(
                    "{}/{}: train_loss={}, train_acc={}".format(
                        t + 1, FLAGS.epochs, task_train_losses[-1], train_accs[-1]
                    )
                )

    result = {"training_losses": task_train_losses, "training_accuracies": train_accs}

    return result, model.model.state_dict()
