# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import torch
import higher

from .base import Solver
from .utils import rand_unit_vectors, circle_points

import time
import datetime

from absl import flags

flags.DEFINE_float("meta_lr", 1e-1, "meta learning rate", lower_bound=0.0)


class MetaLearner(Solver):
    @property
    def name(self):
        return "meta"

    def update_fn(
        self,
        images: torch.FloatTensor,
        labels: torch.FloatTensor,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """One optimization step."""
        self.task_weights.requires_grad_()  # make task weights require grad

        with higher.innerloop_ctx(model, optimizer) as (fmodel, doptimizer):
            task_losses = fmodel(images, labels)
            weighted_loss = torch.sum(task_losses * self.task_weights)
            doptimizer.step(weighted_loss)

            new_task_losses = fmodel(images, labels) * self.preference_weights
            normalized_losses = new_task_losses / torch.sum(new_task_losses)
            kl_divergence = torch.sum(normalized_losses * torch.log(normalized_losses * len(self.task_weights)))
            task_weight_grads = torch.autograd.grad(kl_divergence, self.task_weights)[0]

            # gradient step on task weights
            with torch.no_grad():
                self.task_weights = torch.clamp(self.task_weights - self.flags.meta_lr * task_weight_grads, min=0)
                self.task_weights = self.task_weights / torch.sum(self.task_weights)

            # compute gradients using new task weights
            new_weighted_loss = torch.sum(task_losses * self.task_weights)
            param_grads = torch.autograd.grad(new_weighted_loss, fmodel.parameters(time=0))

        optimizer.zero_grad()
        for index, param in enumerate(model.parameters()):
            param.grad = param_grads[index]
        optimizer.step()

    def epoch_end(self, epoch: int) -> None:
        self.writer.add_scalars(
            f"task_weights",
            {f"task_weights_{i:02d}": self.task_weights[i] for i in range(len(self.task_weights))},
            epoch,
        )
        self.task_weights_history.append(self.task_weights.cpu().numpy())

    def run(self):
        """Run MetaLearner."""
        start_time = time.time()

        results = dict()
        if self.dataset_config.n_tasks == 2:
            preferences = circle_points(self.flags.n_preferences)
        else:
            preferences = rand_unit_vectors(self.dataset_config.n_tasks, self.flags.n_preferences, True)

        for i, preference in enumerate(preferences):
            print(f"[{i}/{len(preferences)}]: Running MetaLearner for preference = {preference}.")

            self.suffix = f"p{i}"
            self.preference_weights = torch.from_numpy(preference).to(self.device)
            self.task_weights = torch.full_like(self.preference_weights, 1. / self.dataset_config.n_tasks)
            self.task_weights_history = []  # store task weights

            model = self.configure_model()
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.flags.lr, momentum=self.flags.momentum
            )

            result, checkpoint = self.train(model, optimizer)
            results[i] = dict(
                r=preference, res=result, checkpoint=checkpoint, task_weights=np.stack(self.task_weights_history)
            )
            self.dump(results, self.prefix + f"_{self.flags.n_preferences}_from_0-{i}.pkl")

        total_time = datetime.timedelta(seconds=round(time.time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")

    def run_timing(self, num_timing_steps: int = 100) -> float:
        """Time the training phase."""
        preference = rand_unit_vectors(self.dataset_config.n_tasks, 1)[0]
        self.preference_weights = torch.from_numpy(preference).to(self.device)
        self.task_weights = torch.full_like(self.preference_weights, 1. / self.dataset_config.n_tasks)

        model = self.configure_model()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.flags.lr, momentum=self.flags.momentum
        )

        seconds_per_training_step = self.time_training_step(model, optimizer, num_timing_steps)
        return seconds_per_training_step