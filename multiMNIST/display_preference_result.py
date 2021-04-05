import glob
import numpy as np
import pathlib
import pickle as pkl
import os

import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from latex_utils import latexify

baseline = "individual"

methods = [
    "epo", 
    "pmtl", 
    "linscalar",
    "meta",
]

markers = {
    "epo": "*",
    "pmtl": "^",
    "linscalar": "s",
    "meta": "+",
    "individual": "o"
}

datasets = [
    "mnist",
    "fashion",
    "fashion_and_mnist"
]

method_names = {
    "epo": "EPO",
    "pmtl": "PMTL",
    "linscalar": "LinScalar",
    "meta": "MetaWeights",
}

model = "lenet"
niters, nprefs = 100, 10
folder = pathlib.Path("/scratch/ssd002/home/kelvin/projects/EPOSearch2/multiMNIST/runs/meta_mnist_v3")

data = dict()
for dataset in datasets:
    print(dataset)
    data[dataset] = dict()
    for method in [baseline] + methods:
        print(f"\t{method}")
        method_folders_glob = folder / f"{method}_{dataset}_{model}_{niters}-*"
        method_folders = sorted(glob.glob(str(method_folders_glob)))

        rs = []
        last_ls = []
        last_acs = []
        for preference_id, method_folder in enumerate(method_folders):
            if preference_id % 2 == 1 and method != baseline:
                continue

            run_id = max(map(int, os.listdir(method_folder)))
            fn_glob = pathlib.Path(method_folder) / f"{run_id}" / "*.pkl"
            fn = glob.glob(str(fn_glob))[0]  # NOTE: we assume there is only one pickle file

            with open(fn, "rb") as f:
                results = pkl.load(f)

            rs.append(results[preference_id]["r"])
            last_ls.append(results[preference_id]["res"]["training_losses"][-1])
            last_acs.append(results[preference_id]["res"]["training_accuracies"][-1])

        last_ls = np.asarray(last_ls)
        last_acs = np.asarray(last_acs)
        data[dataset][method] = dict(last_ls=last_ls, last_acs=last_acs, rs=rs)

    # setup baseline loss/accuracies
    base_task1_acc = np.max(data[dataset][baseline]["last_acs"][:, 0])
    base_task2_acc = np.max(data[dataset][baseline]["last_acs"][:, 1])
    base_task1_loss = np.min(data[dataset][baseline]["last_ls"][:, 0])
    base_task2_loss = np.min(data[dataset][baseline]["last_ls"][:, 1])
    max_task1_loss = max(np.max(v["last_ls"][:, 0]) for k, v in data[dataset].items() if k != baseline)
    max_task2_loss = max(np.max(v["last_ls"][:, 1]) for k, v in data[dataset].items() if k != baseline)

    data[dataset]["baseline_loss"] = [
        ([0, max_task1_loss + 0.05], [base_task2_loss, base_task2_loss]),
        ([base_task1_loss, base_task1_loss], [0, max_task2_loss + 0.05]),
    ]

    data[dataset]["baseline_acc"] = [
        ([0, 1], [base_task2_acc, base_task2_acc]),
        ([base_task1_acc, base_task1_acc], [0, 1]),
    ]

    # setup preferences
    data[dataset]["rs"] = data[dataset][methods[0]]["rs"]

latexify(fig_width=3, fig_height=2)
for dataset in datasets:
    fig, ax = plt.subplots(1, 1)
    x_min, x_max = data[dataset]["baseline_loss"][0][0]
    y_min, y_max = data[dataset]["baseline_loss"][1][1]
    ax.set_xlim(left=x_min, right=x_max + 0.05)
    ax.set_ylim(bottom=y_min, top=y_max + 0.05)

    ax.set_xlabel(r"Task 1 Loss")
    ax.set_ylabel(r"Task 2 Loss")
    for i, (xs, ys) in enumerate(data[dataset]["baseline_loss"]):
        ax.plot(xs, ys, lw=1, alpha=0.3, c="k")

    colors = []
    for i, r in enumerate(data[dataset]["rs"]):
        label = r"Preference" if i == 0 else ""
        r_inv = np.sqrt(1 - r ** 2)
        r_inv = 1000 * r_inv / np.linalg.norm(r_inv)
        lines = ax.plot([0, r_inv[0]], [0, r_inv[1]], lw=1, alpha=0.5, ls="--", dashes=(10, 2), label=label)
        colors.append(lines[0].get_color())

    for method in methods:
        last_ls = data[dataset][method]["last_ls"]
        if method == "pmtl":
            ax.scatter(last_ls[:, 0], last_ls[:, 1], c=colors[::-1], s=20, marker=markers[method])#, label=method)
        else:
            ax.scatter(last_ls[:, 0], last_ls[:, 1], c=colors, s=20, marker=markers[method])#, label=method)

    ax.legend(loc="upper right")
    leg = ax.get_legend()
    for handle in ax.get_legend().legendHandles:
        handle.set_color("black")

    plt.tight_layout()
    fig.savefig(f"figures/{dataset}_loss.pdf")

latexify(fig_width=3, fig_height=2)
for dataset in datasets:
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_label_position("left")
    ax.xaxis.set_label_position("bottom")

    ax.set_xlabel(r"Task 1 Accuracy")
    ax.set_ylabel(r"Task 2 Accuracy")
    for i, (xs, ys) in enumerate(data[dataset]["baseline_acc"]):
        if i == 0:
            ax.plot(xs, ys, lw=1, alpha=0.3, c="k", label="Baseline")
        else:
            ax.plot(xs, ys, lw=1, alpha=0.3, c="k")

    for method in methods:
        last_acs = data[dataset][method]["last_acs"]
        if method == "pmtl":
            ax.scatter(
                last_acs[:, 0], last_acs[:, 1], c=colors[::-1], s=20, marker=markers[method], label=method_names[method]
            )
        else:
            ax.scatter(
                last_acs[:, 0], last_acs[:, 1], c=colors, s=20, marker=markers[method], label=method_names[method]
            )

    ax.legend(loc="lower left")
    leg = ax.get_legend()
    for handle in ax.get_legend().legendHandles:
        handle.set_color("black")

    plt.tight_layout()
    fig.savefig(f"figures/{dataset}_acc.pdf")