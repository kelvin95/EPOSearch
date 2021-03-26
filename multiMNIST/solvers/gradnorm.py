# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import torch

from .base import Solver
from .model_lenet import RegressionModel, RegressionTrain

from time import time
from datetime import timedelta


# copied from
# https://discuss.pytorch.org/t/how-to-flatten-and-then-unflatten-all-model-parameters/34730
def flatten_grad(parameters):
    l = []
    indices = []
    shapes = []
    s = 0
    for p in parameters:
        if p.grad is None:
            shapes.append(None)
            continue
        shapes.append(p.grad.shape)
        p = torch.flatten(p.grad)
        size = p.shape[0]
        l.append(p)
        indices.append((s, s + size))
        s += size
    flat = torch.cat(l).view(-1)
    return {"grad": flat, "indices": indices, "shapes": shapes}


def recover_flattened(flat_grad, indices, shapes):
    l = [flat_grad[s:e] for (s, e) in indices]
    grads = []
    index = 0
    for i in range(len(shapes)):
        if shapes[i] is None:
            grads.append(None)
            continue
        grads.append(l[index].view(shapes[i]))
        index += 1
    return grads


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
        init_weight = np.array([0.5, 0.5])
        alphas = np.arange(0, 1.1, 0.25)
        npref = len(alphas)
        results = dict()
        for i, alpha in enumerate(alphas):
            s_t = time()
            self.alpha = alpha
            self.initial_task_loss = None
            model = RegressionTrain(RegressionModel(self.flags.n_tasks), init_weight)
            if torch.cuda.is_available():
                model.cuda()
            # additional weights for gradnorm
            weights = torch.ones(self.flags.n_tasks)
            if torch.cuda.is_available():
                weights = weights.cuda()
            weights.requires_grad_()
            self.weights = weights
            optimizer = torch.optim.SGD(
                list(model.parameters()) + [weights], lr=1e-3, momentum=0.0
            )
            res, checkpoint = self.train(model, optimizer)
            results[i] = {"r": None, "res": res, "checkpoint": checkpoint}
            t_t = timedelta(seconds=round(time() - s_t))
            print(f"**** Time taken for {self.dataset}_{i} = {t_t}")
            self.dump(results, self.prefix + f"_{npref}_from_0-{i}.pkl")
        total = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total}")