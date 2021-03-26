# Solvers #

The base solver class `Solver` is defined in `base.py`, and should be used to
define your own solvers. The main method of the `Solver` class is `update_fn`.
This method performs the gradient update for a given batch, and is used
internally in `Solver.train`. Besides this, the `Solver` class can also
utilize the `epoch_start, epoch_end` and `pretrain` methods to perform some
miscellaneous operations at the start and end of each epoch, and do
pre-training if necessary. The `Solver.dump` method can be used to dump
results to a `pickle` file. The `Solver.run` method defines the training phase
of the solver, and can be customized to do a hyperparameter sweep etc.

