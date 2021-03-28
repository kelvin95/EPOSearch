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
    return torch.tensor(sol).cuda().float()


class MGDA(Solver):
    @property
    def name(self):
        return "mgda"

    def update_fn(self, X, ts, model, optimizer):
        # obtain and store the gradient
        grads = {}
        for i in range(self.flags.n_tasks):
            optimizer.zero_grad()
            task_loss = model(X, ts)
            task_loss[i].backward()

            # can use scalable method proposed in the MOO-MTL paper for large scale
            # problem but we keep use the gradient of all parameters in this experiment
            grads[i] = []
            for param in model.parameters():
                if param.grad is not None:
                    grads[i].append(
                        Variable(param.grad.data.clone().flatten(), requires_grad=False)
                    )

        grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
        grads = torch.stack(grads_list)

        # calculate the weights
        weight_vec = get_d_mgda(grads)

        # optimization step
        optimizer.zero_grad()
        for i in range(len(task_loss)):
            task_loss = model(X, ts)
            if i == 0:
                loss_total = weight_vec[i] * task_loss[i]
            else:
                loss_total = loss_total + weight_vec[i] * task_loss[i]

        loss_total.backward()
        optimizer.step()

    def run(self):
        """Run mgda"""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        init_weight = np.array([0.5, 0.5])
        npref = 5
        results = dict()
        for i in range(npref):
            s_t = time()
            model = self.configure_model()
            if torch.cuda.is_available():
                model.cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.0)
            res, checkpoint = self.train(model, optimizer)
            results[i] = {"r": None, "res": res, "checkpoint": checkpoint}
            t_t = timedelta(seconds=round(time() - s_t))
            print(f"**** Time taken for {self.dataset}_{i} = {t_t}")
            self.dump(results, self.prefix + f"_{npref}_from_0-{i}.pkl")
        total = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total}")
