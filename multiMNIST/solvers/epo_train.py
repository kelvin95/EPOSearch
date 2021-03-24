import numpy as np
import os

import torch
import torch.utils.data
from torch.autograd import Variable

from model_lenet import RegressionModel, RegressionTrain

from .epo_lp import EPO_LP
from time import time
import pickle


def getNumParams(params):
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable


# Instantia EPO Linear Program Solver
_, n_params = getNumParams(model.parameters())
print(f"# params={n_params}")
epo_lp = EPO_LP(m=n_tasks, n=n_params, r=preference)


def update_fn(X, ts, model, optimizer, flags):
    # Obtain losses and gradients
    grads = {}
    losses = []
    for i in range(flags.n_tasks):
        optimizer.zero_grad()
        task_loss = model(X, ts)
        losses.append(task_loss[i].data.cpu().numpy())
        task_loss[i].backward()

        # One can use scalable method proposed in the MOO-MTL paper
        # for large scale problem; but we use the gradient
        # of all parameters in this experiment.
        grads[i] = []
        for param in model.parameters():
            if param.grad is not None:
                grads[i].append(
                    Variable(param.grad.data.clone().flatten(), requires_grad=False)
                )

    grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
    G = torch.stack(grads_list)
    GG = G @ G.T
    losses = np.stack(losses)

    try:
        # Calculate the alphas from the LP solver
        alpha = flags.epo.epo_lp.get_alpha(losses, G=GG.cpu().numpy(), C=True)
        if epo_lp.last_move == "dom":
            flags.epo.descent += 1
    except Exception as e:
        print(e)
        alpha = None
    if alpha is None:  # A patch for the issue in cvxpy
        alpha = flags.preference / flags.preference.sum()
        flags.epo.n_manual_adjusts += 1

    if torch.cuda.is_available:
        alpha = flags.n_tasks * torch.from_numpy(alpha).cuda()
    else:
        alpha = flags.n_tasks * torch.from_numpy(alpha)
    # Optimization step
    optimizer.zero_grad()
    task_losses = model(X, ts)
    weighted_loss = torch.sum(task_losses * alpha)  # * 5. * max(epo_lp.mu_rl, 0.2)
    weighted_loss.backward()
    optimizer.step()

    print(f"\tdescent={flags.epo.descent/len(flags.train_loader)}")
    if flags.epo.n_manual_adjusts > 0:
        print(f"\t # manual tweek={flags.epo.n_manual_adjusts}")


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20.0 if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20.0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


def run(dataset="mnist", base_model="lenet", niter=100, npref=5):
    """
    run Pareto MTL
    """
    start_time = time()
    init_weight = np.array([0.5, 0.5])
    preferences = circle_points(
        npref, min_angle=0.0001 * np.pi / 2, max_angle=0.9999 * np.pi / 2
    )  # preference
    results = dict()
    out_file_prefix = f"epo_{dataset}_{base_model}_{niter}_{npref}_from_0-"
    for i, pref in enumerate(preferences[::-1]):
        s_t = time()
        res, model = train(dataset, base_model, niter, pref)
        results[i] = {"r": pref, "res": res, "checkpoint": model.model.state_dict()}
        print(f"**** Time taken for {dataset}_{i} = {time() - s_t}")
        results_file = os.path.join("results", out_file_prefix + f"{i}.pkl")
        pickle.dump(results, open(results_file, "wb"))

    print(f"**** Time taken for {dataset} = {time() - start_time}")


if __name__ == "__main__":
    run(dataset="mnist", base_model="lenet", niter=100, npref=5)
    run(dataset="fashion", base_model="lenet", niter=100, npref=5)
    run(dataset="fashion_and_mnist", base_model="lenet", niter=100, npref=5)
