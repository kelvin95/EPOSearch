# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import torch

from .base import Solver

from time import time
from datetime import timedelta


class GradNorm(Solver):
    @property
    def name(self):
        return "gradnorm"

    def update_fn(self, X, ts, model, optimizer):
        weights = self.weights

        # compute loss
        optimizer.zero_grad()
        task_loss = model(X, ts)
        if self.initial_task_loss is None:
            self.initial_task_loss = task_loss.detach()
        initial_task_loss = self.initial_task_loss

        # compute parameter gradients
        weighted_loss = torch.sum(task_loss * weights)
        weighted_loss.backward(retain_graph=True)
        weights.grad.data = weights.grad.data * 0.0

        # compute gradient gradients
        grad_norms = []
        for i in range(len(task_loss)):
            grad = torch.autograd.grad(
                task_loss[i], model.model.parameters(), retain_graph=True
            )
            grad = torch.cat([torch.flatten(x) for x in grad])
            grad_norms.append(torch.linalg.norm(weights[i] * grad))
        grad_norms = torch.stack(grad_norms)

        mean_grad_norm = torch.mean(grad_norms.detach())
        loss_ratio = task_loss.detach() / initial_task_loss
        inverse_loss_ratio = loss_ratio / torch.mean(loss_ratio)
        weight_loss = torch.sum(
            torch.abs(grad_norms - mean_grad_norm * (inverse_loss_ratio ** self.alpha))
        )
        weights.grad.data = torch.autograd.grad(weight_loss, weights)[0]

        # SGD step
        optimizer.step()

        # normalize weights
        weights.data = weights.data / torch.norm(weights.data)

    def run(self):
        """Run gradnorm"""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        results = dict()
        alphas = np.arange(0, 1.1, 1.0 / max(1, self.flags.n_preferences - 1))
        for i, alpha in enumerate(alphas):
            self.alpha = alpha
            self.suffix = f"p{i}"
            self.initial_task_loss = None
            model = self.configure_model()

            # additional weights for gradnorm
            weights = torch.ones(self.dataset_config.n_tasks).to(self.device)
            weights.requires_grad_()
            self.weights = weights

            optimizer = torch.optim.SGD(
                list(model.parameters()) + [weights],
                lr=self.flags.lr,
                momentum=self.flags.momentum,
            )

            result, checkpoint = self.train(model, optimizer)
            results[i] = dict(r=None, res=result, checkpoint=checkpoint)
            self.dump(
                results, self.prefix + f"_{self.flags.n_preferences}_from_0-{i}.pkl"
            )

        total_time = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")
