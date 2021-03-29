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


def get_d_graddrop(gradients, leak=0.0):
    purity = 0.5 * (1 + (torch.sum(gradients, dim=0) / torch.sum(torch.abs(gradients), dim=0)))
    mask = (purity > 0.5) * (gradients > 0) + (purity < 0.5) * (gradients < 0)
    gradients = (leak + (1 - leak) * mask) * gradients
    return torch.sum(gradients, dim=0)


class GradDropDeterministic(Solver):
    @property
    def name(self):
        return "graddrop_deterministic"

    def update_fn(self, X, ts, model, optimizer):
        # obtain and store the gradient
        flat_grads = {}
        task_losses = model(X, ts)
        for i in range(self.dataset_config.n_tasks):
            optimizer.zero_grad()
            task_losses[i].backward(retain_graph=True)
            flat_grads[i] = flatten_grad(model.parameters())

        # clear graph
        optimizer.zero_grad()
        del task_losses

        # calculate the gradient
        grads = torch.stack([flat_grads[i]["grad"] for i in range(len(flat_grads))])
        grads = get_d_graddrop(grads, self.leak)
        grads = recover_flattened(
            grads, flat_grads[0]["indices"], flat_grads[0]["shapes"]
        )

        # optimization step
        optimizer.zero_grad()
        for i, params in enumerate(model.parameters()):
            if grads[i] is not None:
                params.grad = grads[i]
        optimizer.step()

    def run(self):
        """Run deterministic graddrop"""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        results = dict()
        leaks = np.arange(0, 1.1, 1. / max(1, self.flags.n_preferences - 1))
        for i, leak in enumerate(leaks):
            self.leak = leak
            model = self.configure_model()
            optimizer = torch.optim.SGD(model.parameters(), lr=self.flags.lr, momentum=self.flags.momentum)

            result, checkpoint = self.train(model, optimizer)
            results[i] = dict(r=None, res=result, checkpoint=checkpoint)
            self.dump(results, self.prefix + f"_{self.flags.n_preferences}_from_0-{i}.pkl")

        total_time = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")
