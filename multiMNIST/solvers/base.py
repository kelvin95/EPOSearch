import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
import numpy as np
import os
import copy
import pickle
import time
from datetime import datetime
from socket import gethostname
from itertools import combinations
from typing import List, Tuple, Dict, Any, Union, Optional
from pathlib import Path

from .models import MTLModel, MTLModelWithCELoss, MTLLeNet, MTLResNet18
from .dataset import get_dataset_config, load_dataset
from .utils import cosine_angle, flatten_parameters, flatten_grad, gmsim, overload_print


class Solver(object):
    """Base class for solvers"""

    def __init__(self, dataset_name: str, flags: Any):
        """Initialize solver with parsed flags"""
        # copy to avoid mutating unexpectedly
        self.flags = copy.deepcopy(flags)
        self.set_random_seed()

        self.dataset = dataset_name
        self.dataset_config = get_dataset_config(dataset_name)
        self.train_loader, self.test_loader = load_dataset(
            self.dataset_config, self.flags.batch_size, self.flags.n_workers
        )

    def set_random_seed(self):
        torch.manual_seed(self.flags.seed)
        np.random.seed(self.flags.seed)
        # random.seed(self.flags.seed)

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

    def configure_writer(self) -> Tuple[Path, SummaryWriter]:
        """Configure tensorboard summary writer"""
        suffix = getattr(self, "suffix", "")
        basename = "-".join([self.prefix, suffix])
        exp_dir = Path(self.flags.outdir, basename)
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_runs = [f.name for f in exp_dir.iterdir() if f.name.isdigit()]
        if exp_runs:
            latest_run = int(sorted(exp_runs, key=lambda x: int(x))[-1])
            run = latest_run + 1
        else:
            run = 1
        log_dir = exp_dir.joinpath(str(run))
        log_dir.mkdir(parents=False, exist_ok=False)
        writer = SummaryWriter(log_dir=log_dir, flush_secs=120)
        return log_dir, writer

    def log_misc_info(self):
        """Log some info about the env"""
        print(f"Datetime: {datetime.now().strftime('%H:%M:%S on %b %d')}")
        print(f"Hostname: {gethostname()}")
        for key in [
            "SLURM_JOB_ID",
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_ARRAY_TASK_COUNT",
        ]:
            print(f"{key}: {os.environ.get(key, 'not set')}")
        print("Parsed flags: ")
        print(self.flags.flags_into_string())
        print(f"Dataset: {self.dataset}")
        print(f"CUDA: {torch.cuda.is_available()}")

    def configure_model(
        self, with_ce_loss: bool = True
    ) -> Union[MTLModel, MTLModelWithCELoss]:
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

    def epoch_end(self, epoch: int) -> None:
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

    def update_with_metrics(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        model: nn.Module,
        optimizer: Optimizer,
        global_step: int = None,
        metrics: bool = False,
    ) -> None:
        """Wrapped update to compute metrics like the multi-task curvature etc."""
        if not metrics:
            self.update_fn(X, Y, model, optimizer)
            return

        loss_before = model(X, Y).detach().clone()
        theta_before = flatten_parameters(model.parameters())
        self.update_fn(X, Y, model, optimizer)
        # after update, the grad attribute contains non-zero grad
        flat_grads = flatten_grad(model.parameters())["grad"]
        theta_after = flatten_parameters(model.parameters())
        loss_after = model(X, Y)
        mtc = 2 * (
            (loss_after.detach().clone() - loss_before).sum()
            - torch.dot(flat_grads, theta_after - theta_before).detach().clone()
        )
        self.writer.add_scalar("multi-task curvature", mtc, global_step)
        self.log_pairwise_metrics(loss_after, model, global_step)

    def log_pairwise_metrics(
        self,
        task_losses: torch.Tensor,
        model: nn.Module,
        global_step: int = None,
    ) -> None:
        flat_grads = []
        for i in range(self.dataset_config.n_tasks):
            model.zero_grad()
            task_losses[i].backward(retain_graph=True)
            flat_grads.append((i, flatten_grad(model.parameters())["grad"]))

        # clear graph
        model.zero_grad()
        del task_losses

        for t in combinations(flat_grads, 2):
            (i, grad_i), (j, grad_j) = t
            main_tag = f"tasks_{i}_{j}"
            tag_scalar_dict = {
                "cosine_angle": cosine_angle(grad_i, grad_j),
                "grad_magnitude_sim": gmsim(grad_i, grad_j),
            }
            if self.flags.debug:
                print(f"{main_tag} [{global_step}]: {tag_scalar_dict}")
            self.writer.add_scalars(
                main_tag,
                tag_scalar_dict,
                global_step,
            )

    def train(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        use_metrics: bool = True,
    ) -> Tuple[Dict[str, List[float]], Dict[str, torch.Tensor]]:
        """Train model"""
        self.logdir, self.writer = self.configure_writer()
        with overload_print(self.logdir.joinpath("train.log"), "w"):
            self.log_misc_info()

            self.pretrain(model, optimizer)

            train_losses = []
            train_accuracies = []

            for epoch in range(self.flags.epochs):
                self.epoch_start()

                model.train()
                start_time = time.time()
                for index, (images, labels) in enumerate(self.train_loader):
                    global_step = epoch * len(self.train_loader) + index
                    if torch.cuda.is_available():
                        images = images.cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)
                    # self.update_fn(images, labels, model, optimizer)
                    self.update_with_metrics(
                        images, labels, model, optimizer, global_step, use_metrics
                    )

                    if (
                        self.flags.log_frequency > 0
                        and index % self.flags.log_frequency == 0
                    ):
                        with torch.no_grad():
                            task_losses = model(images, labels)
                        print(
                            f"Train Epoch {epoch + 1}/{self.flags.epochs} "
                            f"[{index}/{len(self.train_loader)}]: "
                            f"Losses - {task_losses.cpu().numpy()}"
                        )

                end_time = time.time()
                self.epoch_end(epoch)

                # Calculate and record performance
                if epoch % self.flags.valid_frequency == 0:
                    model.eval()

                    valid_loss = []
                    valid_accuracy = 0.0
                    for _, (images, labels) in enumerate(self.test_loader):
                        if torch.cuda.is_available():
                            images = images.cuda(non_blocking=True)
                            labels = labels.cuda(non_blocking=True)

                        with torch.no_grad():
                            loss, logits = model(images, labels, return_logits=True)

                        if logits.size(1) > 1:
                            predictions = torch.argmax(logits, dim=1)
                        else:
                            predictions = (logits >= 0.0)[:, 0]

                        valid_loss.append(loss)
                        valid_accuracy += torch.sum(predictions.eq(labels), dim=0)

                    valid_loss = torch.mean(torch.stack(valid_loss), dim=0)
                    train_losses.append(valid_loss.cpu().numpy())
                    for i, tl in enumerate(train_losses[-1]):
                        self.writer.add_scalar(f"valid/loss_{i}", tl, epoch)

                    valid_accuracy = valid_accuracy / len(self.test_loader.dataset)
                    train_accuracies.append(valid_accuracy.cpu().numpy())
                    for i, ta in enumerate(train_accuracies[-1]):
                        self.writer.add_scalar(f"valid/acc_{i}", ta, epoch)

                    print(
                        f"Epoch {epoch + 1}/{self.flags.epochs}: "
                        f"valid loss = {train_losses[-1]} "
                        f"valid acc = {train_accuracies[-1]} "
                        f"time - {end_time - start_time:.2f} "
                    )

        result = {
            "training_losses": train_losses,
            "training_accuracies": train_accuracies,
        }
        return result, model.model.state_dict()

    def dump(self, contents, filename):
        """Dump contents to filename"""

        def convert_to_cpu(result):
            if isinstance(result, np.ndarray):
                return result
            if torch.is_tensor(result):
                return result.cpu()
            if isinstance(result, dict):
                return {k: convert_to_cpu(v) for k, v in result.items()}
            if isinstance(result, list):
                return [convert_to_cpu(x) for x in result]
            return result

        fpath = Path(self.logdir, filename)
        with open(fpath, "wb") as f:
            pickle.dump(convert_to_cpu(contents), f)

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
