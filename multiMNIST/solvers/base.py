import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np
import copy
import pickle
from typing import List, Tuple, Dict, Any
from pathlib import Path

from .model_lenet import RegressionModel, RegressionTrain
from .dataset import load_dataset


class Solver(object):
    """Base class for solvers"""

    def __init__(self, dataset: str, flags: Any):
        """Initialize solver with parsed flags"""
        self._dataset = dataset
        # copy to avoid mutating unexpectedly
        self._flags = copy.deepcopy(flags)
        self.train_loader, self.test_loader = load_dataset(dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def flags(self):
        return self._flags

    @property
    def name(self):
        return "base"

    @property
    def prefix(self):
        return f"{self.name}_{self.dataset}_{self.flags.arch}_{self.flags.epochs}"

    def epoch_start(self) -> None:
        """Reset variables/attributes at start of epoch.
        For eg, epoch-wise accumulators etc.
        """
        pass

    def epoch_end(self) -> None:
        """Reset variables/attributes at end of epoch.
        For eg, print some information etc.
        """
        pass

    def pretrain(self, model: nn.Module, optimizer: Optimizer) -> None:
        """Pretrain model using training data (loader)"""
        pass

    def update_fn(
        self, X: torch.Tensor, Y: torch.Tensor, model: nn.Module, optimizer: Optimizer
    ) -> None:
        """Update model parameters for given batch (X, Y) of data.
        NOTE: Override this method
        """
        raise NotImplementedError

    def train(
        self, model: nn.Module, optimizer: Optimizer,
    ) -> Tuple[Dict[str, List[float]], Dict[str, torch.Tensor]]:
        """Train model"""
        # DO PRETRAINING
        # ----------------------------------------
        self.pretrain(model, optimizer)
        # CONTAINERS FOR KEEPING TRACK OF PROGRESS
        # ----------------------------------------
        task_train_losses = []
        train_accs = []
        # ---------***---------

        # TRAIN
        # -----
        for t in range(self.flags.epochs):
            self.epoch_start()
            model.train()
            for (it, batch) in enumerate(self.train_loader):

                X = batch[0]
                ts = batch[1]
                if torch.cuda.is_available():
                    X = X.cuda()
                    ts = ts.cuda()

                self.update_fn(X, ts, model, optimizer)

            self.epoch_end()

            # Calculate and record performance
            if t == 0 or (t + 1) % 2 == 0:
                model.eval()
                with torch.no_grad():
                    total_train_loss = []
                    train_acc = []

                    correct1_train = 0
                    correct2_train = 0

                    for (it, batch) in enumerate(self.test_loader):

                        X = batch[0]
                        ts = batch[1]
                        if torch.cuda.is_available():
                            X = X.cuda()
                            ts = ts.cuda()

                        valid_train_loss = model(X, ts)
                        total_train_loss.append(valid_train_loss)
                        output1 = model.model(X).max(2, keepdim=True)[1][:, 0]
                        output2 = model.model(X).max(2, keepdim=True)[1][:, 1]
                        correct1_train += (
                            output1.eq(ts[:, 0].view_as(output1)).sum().item()
                        )
                        correct2_train += (
                            output2.eq(ts[:, 1].view_as(output2)).sum().item()
                        )

                    train_acc = np.stack(
                        [
                            1.0 * correct1_train / len(self.test_loader.dataset),
                            1.0 * correct2_train / len(self.test_loader.dataset),
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
                            t + 1,
                            self.flags.epochs,
                            task_train_losses[-1],
                            train_accs[-1],
                        )
                    )

        result = {
            "training_losses": task_train_losses,
            "training_accuracies": train_accs,
        }

        return result, model.model.state_dict()

    def dump(self, contents, filename):
        """Dump contents to filename"""
        fpath = Path(self.flags.outdir, filename)
        with open(fpath, "wb") as f:
            pickle.dump(contents, f)

    def run(self):
        """Run a complete training phase.
        This can be overwritten to do a hyperparameter sweep etc."""

        # DEFINE MODEL
        # ---------------------
        model = RegressionTrain(
            RegressionModel(self.flags.n_tasks), np.array([0.5, 0.5])
        )

        if torch.cuda.is_available():
            model.cuda()
        # ---------***---------

        # DEFINE OPTIMIZERS
        # -----------------
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.0)
        # ---------***---------

        self.train(model, optimizer)
