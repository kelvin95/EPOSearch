from pathlib import Path
from time import time
from datetime import timedelta

from solvers import SOLVER_FACTORY

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_float("lr", 1e-3, "learning rate", lower_bound=0.0)
flags.DEFINE_integer("epochs", 1, "number of training epochs", lower_bound=0)
flags.DEFINE_integer("n_tasks", 2, "number of tasks", lower_bound=2, upper_bound=2)
flags.DEFINE_multi_enum(
    "dset",
    "all",
    ["mnist", "fashion", "fashion_and_mnist", "all"],
    "name of dataset to use",
)
flags.DEFINE_string("outdir", "out", "Output dir to save results")
flags.DEFINE_enum("arch", "lenet", ["lenet"], "network architecture to use")
flags.DEFINE_enum(
    "solver", "epo", list(SOLVER_FACTORY.keys()), "name of method/solver",
)
flags.DEFINE_boolean("debug", False, "Produces debugging output.")


def main(argv):
    if FLAGS.dset == "all" or (isinstance(FLAGS.dset, list) and "all" in FLAGS.dset):
        FLAGS.dset = ["mnist", "fashion", "fashion_and_mnist"]
    else:
        # unique while preserving order passed in on cmdline
        FLAGS.dset = list(dict.fromkeys(FLAGS.dset))

    if not Path(FLAGS.outdir).exists():
        Path(FLAGS.outdir).mkdir(parents=True)

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
