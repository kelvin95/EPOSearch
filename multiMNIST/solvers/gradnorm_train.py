# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import os
import torch


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


def update_fn(X, ts, model, optimizer, flags):
    if len(optimizer.param_groups) == 1:
        # initialize weights
        weights = torch.ones(flags.n_tasks)
        if torch.cuda.is_available():
            weights = weights.cuda()
        weights.requires_grad_()
        optimizer.add_param_group({"params": weights})

    weights = optimizer.param_groups[-1]["params"][0]

    # compute loss
    optimizer.zero_grad()
    task_loss = model(X, ts)
    if not hasattr(flags.gradnorm, "initial_task_loss"):
        flags.gradnorm.initial_task_loss = task_loss.detach()
    initial_task_loss = flags.gradnorm.initial_task_loss

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
        torch.abs(
            grad_norms - mean_grad_norm * (inverse_loss_ratio ** flags.gradnorm.alpha)
        )
    )
    weights.grad.data = torch.autograd.grad(weight_loss, weights)[0]

    # SGD step
    optimizer.step()

    # normalize weights
    weights.data = weights.data / torch.norm(weights.data)


def run(dataset="mnist", base_model="lenet", niter=100, npref=5):
    """
    run PCGrad
    """
    start_time = time()
    init_weight = np.array([0.5, 0.5])
    alpha = np.arange(0.0, 1.1, 0.25)
    out_file_prefix = f"gradnorm_{dataset}_{base_model}_{niter}_{npref}_from_0-"
    results = dict()
    for i in range(npref):
        s_t = time()
        res, model = train(dataset, base_model, niter, npref, init_weight, i, alpha[i])
        results[i] = {"r": None, "res": res, "checkpoint": model.model.state_dict()}
        print(f"**** Time taken for {dataset}_{i} = {time() - s_t}")
        results_file = os.path.join("results", out_file_prefix + f"{i}.pkl")
        pickle.dump(results, open(results_file, "wb"))
    print(f"**** Time taken for {dataset} = {time() - start_time}")


if __name__ == "__main__":
    run(dataset="mnist", base_model="lenet", niter=100, npref=5)
    run(dataset="fashion", base_model="lenet", niter=100, npref=5)
    run(dataset="fashion_and_mnist", base_model="lenet", niter=100, npref=5)
