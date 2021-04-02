# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import torch
from torch.autograd import Variable

from .base import Solver
from .min_norm_solvers import MinNormSolver

from time import time
from datetime import timedelta


def get_d_mgda(vec):
    r"""Calculate the gradient direction for MGDA."""
    sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])
    return torch.tensor(sol, device=vec.device, dtype=torch.float)


class MGDA(Solver):
    @property
    def name(self):
        return "mgda"

    def update_fn(self, images, labels, model, optimizer):
        optimizer.zero_grad()

        # obtain and store the gradient
        grads = {}
        task_losses = model(images, labels)
        for i in range(self.dataset_config.n_tasks):
            optimizer.zero_grad()
            task_losses[i].backward(retain_graph=True)

            # can use scalable method proposed in the MOO-MTL paper for large scale
            # problem but we keep use the gradient of all parameters in this experiment
            grads[i] = []
            for param in model.parameters():
                if param.grad is not None:
                    grads[i].append(param.grad.data.clone().flatten())

        # clear graph
        optimizer.zero_grad()
        del task_losses

        # calculate the weights
        grads = torch.stack([torch.cat(grads[i]) for i in range(len(grads))])
        weight_vec = get_d_mgda(grads)

        # optimization step
        optimizer.zero_grad()
        task_losses = model(images, labels)
        total_loss = torch.sum(weight_vec * task_losses)
        total_loss.backward()
        optimizer.step()

    def run(self):
        """Run mgda"""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        results = dict()
        for i in range(self.flags.n_preferences):
            self.suffix = f"p{i}"
            s_t = time()
            model = self.configure_model()
            optimizer = torch.optim.SGD(model.parameters(), lr=self.flags.lr, momentum=self.flags.momentum)

            result, checkpoint = self.train(model, optimizer)
            results[i] = dict(r=None, res=result, checkpoint=checkpoint)
            self.dump(results, self.prefix + f"_{self.flags.n_preferences}_from_0-{i}.pkl")

        total_time = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")
