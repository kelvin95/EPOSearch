# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import torch
from torch.autograd import Variable
from .min_norm_solvers import MinNormSolver


def get_d_mgda(vec):
    r"""Calculate the gradient direction for MGDA."""
    sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])
    return torch.tensor(sol).cuda().float()


def update_fn(X, ts, model, optimizer, flags):
    # obtain and store the gradient
    grads = {}
    for i in range(flags.n_tasks):
        optimizer.zero_grad()
        task_loss = model(X, ts)
        task_loss[i].backward()

        # can use scalable method proposed in the MOO-MTL paper for large scale problem
        # but we keep use the gradient of all parameters in this experiment
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


def run(dataset="mnist", base_model="lenet", niter=100, npref=5):
    """
    run MGDA
    """
    start_time = time()
    init_weight = np.array([0.5, 0.5])
    out_file_prefix = f"mgda_{dataset}_{base_model}_{niter}_{npref}_from_0-"
    results = dict()
    for i in range(npref):
        s_t = time()
        res, model = train(dataset, base_model, niter, npref, init_weight, i)
        results[i] = {"r": None, "res": res, "checkpoint": model.model.state_dict()}
        print(f"**** Time taken for {dataset}_{i} = {time() - s_t}")
        results_file = os.path.join("results", out_file_prefix + f"{i}.pkl")
        pickle.dump(results, open(results_file, "wb"))
    print(f"**** Time taken for {dataset} = {time() - start_time}")


if __name__ == "__main__":
    run(dataset="mnist", base_model="lenet", niter=100, npref=5)
    run(dataset="fashion", base_model="lenet", niter=100, npref=5)
    run(dataset="fashion_and_mnist", base_model="lenet", niter=100, npref=5)
