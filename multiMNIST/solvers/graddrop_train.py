# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import os
import torch
from torch.autograd import Variable
from min_norm_solvers import MinNormSolver


# copied from https://discuss.pytorch.org/t/how-to-flatten-and-then-unflatten-all-model-parameters/34730
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
    purity = 0.5 * (1 + torch.sum(gradients)) / torch.sum(torch.abs(gradients), dim=0)
    uniform = torch.rand_like(purity)
    mask = (purity > uniform) * (gradients > 0) + (purity < uniform) * (gradients < 0)
    gradients = (leak + (1 - leak) * mask) * gradients
    return torch.sum(gradients, dim=0)


def update_fn(X, ts, model, optimizer, flags):
    "Graddrop update"
    # obtain and store the gradient
    flat_grads = {}
    for i in range(flags.n_tasks):
        optimizer.zero_grad()
        task_loss = model(X, ts)
        task_loss[i].backward()
        flat_grads[i] = flatten_grad(model.parameters())

    grads = [flat_grads[i]["grad"] for i in range(len(flat_grads))]
    grads = torch.stack(grads)

    # calculate the gradient
    grads = get_d_graddrop(grads, flags.graddrop.leak)
    grads = recover_flattened(grads, flat_grads[0]["indices"], flat_grads[0]["shapes"])

    # optimization step
    optimizer.zero_grad()
    for i, params in enumerate(model.parameters()):
        if grads[i] is not None:
            params.grad = grads[i]
    optimizer.step()


def run(dataset="mnist", base_model="lenet", niter=100, npref=5):
    """
    run PCGrad
    """
    start_time = time()
    init_weight = np.array([0.5, 0.5])
    leak = np.arange(0, 1.1, 0.25)
    out_file_prefix = f"graddrop_{dataset}_{base_model}_{niter}_{npref}_from_0-"
    results = dict()
    for i in range(npref):
        s_t = time()
        res, model = train(dataset, base_model, niter, npref, init_weight, i, leak[i])
        results[i] = {"r": None, "res": res, "checkpoint": model.model.state_dict()}
        print(f"**** Time taken for {dataset}_{i} = {time() - s_t}")
        results_file = os.path.join("results", out_file_prefix + f"{i}.pkl")
        pickle.dump(results, open(results_file, "wb"))
    print(f"**** Time taken for {dataset} = {time() - start_time}")


if __name__ == "__main__":
    run(dataset="mnist", base_model="lenet", niter=100, npref=5)
    run(dataset="fashion", base_model="lenet", niter=100, npref=5)
    run(dataset="fashion_and_mnist", base_model="lenet", niter=100, npref=5)
