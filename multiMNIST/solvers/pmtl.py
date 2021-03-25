# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import torch
from torch.autograd import Variable

from .base import Solver
from .model_lenet import RegressionModel, RegressionTrain
from .min_norm_solvers import MinNormSolver
from .utils import circle_points

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

    weight0 = torch.sum(
        torch.stack([sol[j] * w[idx][j, 0] for j in torch.arange(0, torch.sum(idx))])
    )
    weight1 = torch.sum(
        torch.stack([sol[j] * w[idx][j, 1] for j in torch.arange(0, torch.sum(idx))])
    )
    weight = torch.stack([weight0, weight1])

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

    weight0 = sol[0] + torch.sum(
        torch.stack(
            [sol[j] * w[idx][j - 2, 0] for j in torch.arange(2, 2 + torch.sum(idx))]
        )
    )
    weight1 = sol[1] + torch.sum(
        torch.stack(
            [sol[j] * w[idx][j - 2, 1] for j in torch.arange(2, 2 + torch.sum(idx))]
        )
    )
    weight = torch.stack([weight0, weight1])

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
            for (it, batch) in enumerate(self.train_loader):
                X = batch[0]
                ts = batch[1]
                if torch.cuda.is_available():
                    X = X.cuda()
                    ts = ts.cuda()

                grads = {}
                losses_vec = []

                # obtain and store the gradient value
                for i in range(self.flags.n_tasks):
                    optimizer.zero_grad()
                    task_loss = model(X, ts)
                    losses_vec.append(task_loss[i].data)

                    task_loss[i].backward()

                    grads[i] = []

                    # can use scalable method proposed in the MOO-MTL paper for
                    # large scale problem but we keep use the gradient of all
                    # parameters in this experiment
                    for param in model.parameters():
                        if param.grad is not None:
                            grads[i].append(
                                Variable(
                                    param.grad.data.clone().flatten(),
                                    requires_grad=False,
                                )
                            )

                grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
                grads = torch.stack(grads_list)

                # calculate the weights
                losses_vec = torch.stack(losses_vec)
                flag, weight_vec = get_d_paretomtl_init(
                    grads, losses_vec, self.ref_vec, self.pref_idx
                )

                # early stop once a feasible solution is obtained
                if flag:
                    print("fealsible solution is obtained.")
                    break

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

            else:
                # continue if no feasible solution is found
                continue
            # break the loop once a feasible solutions is found
            break

    def update_fn(self, X, ts, model, optimizer):
        # obtain and store the gradient
        grads = {}
        losses_vec = []

        for i in range(self.flags.n_tasks):
            optimizer.zero_grad()
            task_loss = model(X, ts)
            losses_vec.append(task_loss[i].data)

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
        losses_vec = torch.stack(losses_vec)
        weight_vec = get_d_paretomtl(grads, losses_vec, self.ref_vec, self.pref_idx)

        # normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
        normalize_coeff = 1.0 / torch.sum(torch.abs(weight_vec))
        weight_vec = weight_vec * normalize_coeff

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
        """Run Pareto MTL"""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        init_weight = np.array([0.5, 0.5])
        npref = 5
        rvecs = circle_points(
            npref, min_angle=0.0001 * np.pi / 2, max_angle=0.9999 * np.pi / 2
        )
        preferences = torch.tensor(rvecs).cuda().float()
        results = dict()
        for i, preference in enumerate(preferences):
            s_t = time()
            self.ref_vec = preferences
            self.pref_idx = i
            model = RegressionTrain(RegressionModel(self.flags.n_tasks), init_weight)
            if torch.cuda.is_available():
                model.cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.0)
            res, checkpoint = self.train(model, optimizer)
            results[i] = {"r": preference, "res": res, "checkpoint": checkpoint}
            t_t = timedelta(seconds=round(time() - s_t))
            print(f"**** Time taken for {self.dataset}_{i} = {t_t}")
            self.dump(results, self.prefix + f"_{npref}_from_0-{i}.pkl")
        total = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total}")
