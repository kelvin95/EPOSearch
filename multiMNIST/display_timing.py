import glob
import numpy as np
import pathlib
import pickle

import matplotlib.pyplot as plt
from latex_utils import latexify

methods = [
    "epo",
    "pmtl",
    "linscalar",
    "meta"
]

method_names = {
    "epo": "EPO",
    "pmtl": "PMTL",
    "linscalar": "LinScalar",
    "meta": "MetaWeights",
}

formats = {
    "epo": "-*",
    "pmtl": "-^",
    "linscalar": "-s",
    "meta": "-+",
    "individual": "-o"
}

models = [
    "lenet",
    "resnet18",
    # "resnet34"
]

dataset = "celeba"
niters = 100
folder = pathlib.Path("/scratch/ssd002/home/kelvin/projects/EPOSearch2/multiMNIST/runs/meta_celeba_timings")


data = dict()
for model_name in models:
    print(model_name)
    data[model_name] = dict()
    for method in methods:
        method_fn = folder / f"{method}_{dataset}_{model_name}_{niters}-timing.pkl"
        with open(method_fn, "rb") as f:
            timings = pickle.load(f)
        data[model_name][method] = timings


latexify(fig_width=3, fig_height=2)
for model_name in models:
    y_min = min([t for ts in data[model_name].values() for t in ts.values()])
    y_max = max([t for ts in data[model_name].values() for t in ts.values()])

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(left=0, right=41)
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_yscale("log")

    ax.set_xlabel(r"\# Tasks")
    ax.set_ylabel(r"Timing (log sec.)")

    for method in methods:
        num_attributes = [n for n, t in data[model_name][method].items()]
        seconds_per_training_step = [t for n, t in data[model_name][method].items()]
        ax.plot(
            num_attributes,
            seconds_per_training_step,
            "-o",
            markersize=3,
            lw=1,
            label=method_names[method]
        )

    ax.legend()
    plt.tight_layout()
    fig.savefig(f"figures/{model_name}_timings.pdf")