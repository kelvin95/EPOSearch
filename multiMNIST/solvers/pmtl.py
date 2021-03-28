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
from .utils import rand_unit_vectors

from time import time
from datetime import timedelta


def get_d_paretomtl_init(grads, value, weights, i):
    """
    calculate the gradient direction for ParetoMTL initialization
    """

    flag = False
    nobj = value.shape

    # check active constraints
    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight

    gx = torch.matmul(w, value / torch.norm(value))
    idx = gx > 0

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        flag = True
        return flag, torch.zeros(nobj)

    if torch.sum(idx) == 1:
        sol = torch.ones(1).cuda().float()
    else:
        vec = torch.matmul(w[idx], grads)
        sol, nd = MinNormSolver.find_min_norm_element(
            [[vec[t]] for t in range(len(vec))]
        )

    weight = torch.sum(torch.stack([sol[j] * w[idx][j] for j in torch.arange(0, torch.sum(idx))]), dim=0)
    return flag, weight


def get_d_paretomtl(grads, value, weights, i):
    """ calculate the gradient direction for ParetoMTL """

    # check active constraints
    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight

    gx = torch.matmul(w, value / torch.norm(value))
    idx = gx > 0

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        sol, nd = MinNormSolver.find_min_norm_element(
            [[grads[t]] for t in range(len(grads))]
        )
        return torch.tensor(sol).cuda().float()

    vec = torch.cat((grads, torch.matmul(w[idx], grads)))
    sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

    weight = torch.sum(torch.stack([sol[j] * w[idx][j - 2] for j in torch.arange(2, 2 + torch.sum(idx))]), dim=0)
    weight = weight + torch.from_numpy(sol[:len(grads)]).to(weight.device, torch.float)
    return weight


class PMTL(Solver):
    @property
    def name(self):
        return "pmtl"

    def pretrain(self, model, optimizer):
        """Pretraining """
        # run at most 2 epochs to find the initial solution
        # stop early once a feasible solution is found
        # usually can be found with a few steps
        for t in range(2):

            model.train()
            for _, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                grads = {}
                losses_vec = []

                # obtain and store the gradient value
                optimizer.zero_grad()
                task_losses = model(images, labels)
                for i in range(self.dataset_config.n_tasks):
                    optimizer.zero_grad()
                    losses_vec.append(task_losses[i].data)
                    task_losses[i].backward(retain_graph=True)

                    # can use scalable method proposed in the MOO-MTL paper for
                    # large scale problem but we keep use the gradient of all
                    # parameters in this experiment
                    grads[i] = []
                    for param in model.parameters():
                        if param.grad is not None:
                            grads[i].append(param.grad.data.clone().flatten())

                # calculate the weights
                losses_vec = torch.stack(losses_vec)
                grads = torch.stack([torch.cat(grads[i]) for i in range(len(grads))])
                flag, weight_vec = get_d_paretomtl_init(
                    grads, losses_vec, self.ref_vec, self.pref_idx
                )

                # early stop once a feasible solution is obtained
                if flag:
                    print("feasible solution is obtained.")
                    break

                # optimization step
                optimizer.zero_grad()
                task_losses = model(images, labels)
                total_loss = torch.sum(task_losses * weight_vec)
                total_loss.backward()
                optimizer.step()

            else:
                # continue if no feasible solution is found
                continue
            # break the loop once a feasible solutions is found
            break

    def update_fn(self, images, labels, model, optimizer):
        optimizer.zero_grad()

        # obtain and store the gradient
        grads = {}
        losses_vec = []

        task_losses = model(images, labels)
        for i in range(self.dataset_config.n_tasks):
            optimizer.zero_grad()
            losses_vec.append(task_losses[i].data)
            task_losses[i].backward(retain_graph=True)

            # can use scalable method proposed in the MOO-MTL paper for large scale
            # problem but we keep use the gradient of all parameters in this experiment
            grads[i] = []
            for param in model.parameters():
                if param.grad is not None:
                    grads[i].append(param.grad.data.clone().flatten())


        # calculate the weights
        losses_vec = torch.stack(losses_vec)
        grads = torch.stack([torch.cat(grads[i]) for i in range(len(grads))])
        weight_vec = get_d_paretomtl(grads, losses_vec, self.ref_vec, self.pref_idx)

        # normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
        normalize_coeff = 1.0 / torch.sum(torch.abs(weight_vec))
        weight_vec = weight_vec * normalize_coeff

        # optimization step
        optimizer.zero_grad()
        task_losses = model(images, labels)
        total_loss = torch.sum(task_losses * weight_vec)
        total_loss.backward()
        optimizer.step()

    def run(self):
        """Run Pareto MTL"""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        preferences = rand_unit_vectors(self.dataset_config.n_tasks, self.flags.n_preferences, True)
        preferences = torch.tensor(preferences, device=self.device, dtype=torch.float)

        results = dict()
        for i, preference in enumerate(preferences):
            self.ref_vec = preferences
            self.pref_idx = i
            model = self.configure_model()
            optimizer = torch.optim.SGD(model.parameters(), lr=self.flags.lr, momentum=self.flags.momentum)

            result, checkpoint = self.train(model, optimizer)
            results[i] = dict(r=preference, res=result, checkpoint=checkpoint)
            self.dump(results, self.prefix + f"_{self.flags.n_preferences}_from_0-{i}.pkl")

        total_time = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")
