import numpy as np
import torch


def update_fn(X, ts, model, optimizer, flags):
    if torch.cuda.is_available:
        alpha = torch.from_numpy(flags.preference).cuda()
    else:
        alpha = torch.from_numpy(flags.preference)
    # Optimization step
    optimizer.zero_grad()
    task_losses = model(X, ts)
    weighted_loss = torch.sum(task_losses * alpha)  # * 5. * max(epo_lp.mu_rl, 0.2)
    weighted_loss.backward()
    optimizer.step()


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
    out_file_prefix = f"linscalar_{dataset}_{base_model}_{niter}_{npref}_from_0-"
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
