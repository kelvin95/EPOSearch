from pathlib import Path
from time import time
from datetime import timedelta

from solvers import SOLVER_FACTORY

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 100, "number of training epochs", lower_bound=0)
flags.DEFINE_integer("log_frequency", 0, "validation frequency", lower_bound=0)
flags.DEFINE_integer("valid_frequency", 1, "validation frequency", lower_bound=1)

flags.DEFINE_float("lr", 1e-3, "learning rate", lower_bound=0.0)
flags.DEFINE_float("momentum", 0.0, "momentum", lower_bound=0.0)
flags.DEFINE_integer("batch_size", 256, "batch size ", lower_bound=1)
flags.DEFINE_integer("n_workers", 4, "number of data workers", lower_bound=0)
flags.DEFINE_integer("n_preferences", 5, "number of preference vectors", lower_bound=1)

flags.DEFINE_multi_enum(
    "dset",
    "all",
    ["mnist", "fashion", "fashion_and_mnist", "all", "celeba", "cifar10", "cifar100"],
    "name of dataset to use",
)
flags.DEFINE_string("outdir", "runs", "Output dir to save results")
flags.DEFINE_enum("arch", "lenet", ["lenet", "resnet18", "resnet34"], "network architecture to use")
flags.DEFINE_enum(
    "init",
    "xavier",
    ["xavier", "uniform", "normal", "kaiming"],
    "weight initialization method",
)
flags.DEFINE_enum(
    "solver",
    "epo",
    list(SOLVER_FACTORY.keys()),
    "name of method/solver",
)
flags.DEFINE_integer("seed", 46848, "Seed for reproducibility")
flags.DEFINE_boolean("debug", False, "Produces debugging output.")


def main(argv):
    if FLAGS.dset == "all" or (isinstance(FLAGS.dset, list) and "all" in FLAGS.dset):
        FLAGS.dset = ["mnist", "fashion", "fashion_and_mnist"]
    else:
        # unique while preserving order passed in on cmdline
        FLAGS.dset = list(dict.fromkeys(FLAGS.dset))

    Path(FLAGS.outdir).mkdir(parents=True, exist_ok=True)

    if FLAGS.debug:
        print("non-flag arguments: ", argv)

    start = time()
    for dataset in FLAGS.dset[:]:
        solver = SOLVER_FACTORY[FLAGS.solver](dataset, FLAGS)
        solver.run()
    total = round(time() - start)
    print(f"**** Datasets: {FLAGS.dset}, Solver: {FLAGS.solver}")
    print(f"**** Total time: {timedelta(seconds=total)}")


if __name__ == "__main__":
    app.run(main)
