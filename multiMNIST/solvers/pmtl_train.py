# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
from .min_norm_solvers import MinNormSolver


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


def circle_points_(r, n):
    """
    generate evenly distributed unit preference vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20.0 if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20.0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


# ref_vec = torch.tensor(circle_points([1], [npref])[0]).cuda().float()
rvecs = circle_points(npref, min_angle=0.0001 * np.pi / 2, max_angle=0.9999 * np.pi / 2)
ref_vec = torch.tensor(rvecs).cuda().float()

# print the current preference vector
print("Preference Vector ({}/{}):".format(pref_idx + 1, npref))
print(ref_vec[pref_idx].cpu().numpy())


def pretrain(train_loader, model, optimizer, flags):

    # run at most 2 epochs to find the initial solution
    # stop early once a feasible solution is found
    # usually can be found with a few steps
    for t in range(2):

        model.train()
        for (it, batch) in enumerate(train_loader):
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            grads = {}
            losses_vec = []

            # obtain and store the gradient value
            for i in range(flags.n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts)
                losses_vec.append(task_loss[i].data)

                task_loss[i].backward()

                grads[i] = []

                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(
                            Variable(
                                param.grad.data.clone().flatten(), requires_grad=False
                            )
                        )

            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)

            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            flag, weight_vec = get_d_paretomtl_init(
                grads, losses_vec, ref_vec, pref_idx
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


def update_fn(X, ts, model, optimizer, flags):
    # obtain and store the gradient
    grads = {}
    losses_vec = []

    for i in range(flags.n_tasks):
        optimizer.zero_grad()
        task_loss = model(X, ts)
        losses_vec.append(task_loss[i].data)

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
    losses_vec = torch.stack(losses_vec)
    weight_vec = get_d_paretomtl(grads, losses_vec, ref_vec, pref_idx)

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


def run(dataset="mnist", base_model="lenet", niter=100, npref=5):
    """
    run Pareto MTL
    """

    init_weight = np.array([0.5, 0.5])
    start_time = time()
    preferences = circle_points(npref)  # preference
    results = dict()
    out_file_prefix = f"pmtl_{dataset}_{base_model}_{niter}_{npref}_from_0-"
    for i, pref in enumerate(preferences):
        s_t = time()
        pref_idx = i
        res, model = train(dataset, base_model, niter, npref, init_weight, pref_idx)
        results[i] = {"r": pref, "res": res, "checkpoint": model.model.state_dict()}
        print(f"**** Time taken for {dataset}_{i} = {time() - s_t}")
        results_file = os.path.join("results", out_file_prefix + f"{i}.pkl")
        pickle.dump(results, open(results_file, "wb"))
    print(f"**** Time taken for {dataset} = {time() - start_time}")


run(dataset="mnist", base_model="lenet", niter=100, npref=5)
run(dataset="fashion", base_model="lenet", niter=100, npref=5)
run(dataset="fashion_and_mnist", base_model="lenet", niter=100, npref=5)
