from pathlib import Path
from time import time
from datetime import timedelta
import tempfile
import pickle
import os

from solvers import SOLVER_FACTORY
import train  # import training flags

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_timing_steps", 100, "number of timing steps", lower_bound=1)


CELEBA = "celeba-{:d}"


def main(argv):
    """Time each method in `SOLVERS`."""
    outdir = FLAGS.outdir  # save original outdir
    with tempfile.TemporaryDirectory() as tempdir:
        FLAGS.outdir = tempdir # don't write anything
        print(f"Writing to {tempdir}")

        timing_results = dict()
        num_attributes_list = [1] + list(range(5, 41, 5))
        if FLAGS.solver == "pmtl":
            num_attributes_list = [2] + list(range(5, 41, 5))

        for num_attributes in num_attributes_list:
            dataset_name = CELEBA.format(num_attributes)
            solver = SOLVER_FACTORY[FLAGS.solver](dataset_name, FLAGS)
            seconds_per_training_step = solver.run_timing(FLAGS.num_timing_steps)
            timing_results[num_attributes] = seconds_per_training_step

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    output_fn = os.path.join(outdir, f"{FLAGS.solver}_celeba_{FLAGS.arch}_{FLAGS.num_timing_steps}-timing.pkl")
    with open(output_fn, "wb") as f:
        pickle.dump(timing_results, f)

    print(f"Saved timing results to {output_fn}.")
    print(timing_results)


if __name__ == "__main__":
    app.run(main)
