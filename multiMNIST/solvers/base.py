import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np
import copy
import pickle
from typing import List, Tuple, Dict, Any, Union
from pathlib import Path

from .models import MTLModel, MTLModelWithCELoss, MTLLeNet, MTLResNet18
from .dataset import get_dataset_config, load_dataset


class Solver(object):
    """Base class for solvers"""

    def __init__(self, dataset_name: str, flags: Any):
        """Initialize solver with parsed flags"""
        # copy to avoid mutating unexpectedly
        self.flags = copy.deepcopy(flags)

        self.dataset = dataset_name
        self.dataset_config = get_dataset_config(dataset_name)
        self.train_loader, self.test_loader = load_dataset(
            self.dataset_config, self.flags.batch_size, self.flags.n_workers
        )

    @property
    def name(self):
        return "base"

    @property
    def prefix(self):
        return f"{self.name}_{self.dataset}_{self.flags.arch}_{self.flags.epochs}"

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available:
            return torch.device("cuda")
        return torch.device("cpu")

    def configure_model(self, with_ce_loss: bool = True) -> Union[MTLModel, MTLModelWithCELoss]:
        """Return a configured MTL model."""
        if self.flags.arch == "resnet18":
            model = MTLResNet18(
                self.dataset_config.n_tasks,
                self.dataset_config.n_classes_per_task,
                self.dataset_config.input_shape,
            )
        elif self.flags.arch == "lenet":
            model = MTLLeNet(
                self.dataset_config.n_tasks,
                self.dataset_config.n_classes_per_task,
                self.dataset_config.input_shape,
            )
        else:
            raise ValueError(f"Model {self.flags.arch} is not supported!")

        if with_ce_loss:
            model = MTLModelWithCELoss(model)
        model = model.to(self.device)
        return model

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
        self.pretrain(model, optimizer)

        train_losses = []
        train_accuracies = []

        for epoch in range(self.flags.epochs):
            self.epoch_start()

            model.train()
            for index, (images, labels) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                self.update_fn(images, labels, model, optimizer)

                if self.flags.log_frequency > 0 and index % self.flags.log_frequency == 0:
                    with torch.no_grad():
                        task_losses = model(images, labels)
                    print(
                        f"Train Epoch {epoch + 1}/{self.flags.epochs} "
                        f"[{index}/{len(self.train_loader)}]: "
                        f"Losses - {task_losses.cpu().numpy()}"
                    )

            self.epoch_end()

            # Calculate and record performance
            if epoch % self.flags.valid_frequency == 0:
                model.eval()

                valid_loss = []
                valid_accuracy = 0.
                for _, (images, labels) in enumerate(self.test_loader):
                    if torch.cuda.is_available():
                        images = images.cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)

                    with torch.no_grad():
                        loss, logits = model(images, labels, return_logits=True)

                    if logits.size(1) > 1:
                        predictions = torch.argmax(logits, dim=1)
                    else:
                        predictions = (logits >= 0.)[:, 0]

                    valid_loss.append(loss)
                    valid_accuracy += torch.sum(predictions.eq(labels), dim=0)

                valid_loss = torch.mean(torch.stack(valid_loss), dim=0)
                train_losses.append(valid_loss.cpu().numpy())

                valid_accuracy = valid_accuracy / len(self.test_loader.dataset)
                train_accuracies.append(valid_accuracy.cpu().numpy())

                print(
                    f"Validation Epoch {epoch + 1}/{self.flags.epochs}: "
                    f"train_loss = {train_losses[-1]} "
                    f"train_acc = {train_accuracies[-1]} "
                )

        result = {
            "training_losses": train_losses,
            "training_accuracies": train_accuracies,
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
        model = self.configure_model()
        if torch.cuda.is_available():
            model.cuda()
        # ---------***---------

        # DEFINE OPTIMIZERS
        # -----------------
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.0)
        # ---------***---------

        self.train(model, optimizer)
