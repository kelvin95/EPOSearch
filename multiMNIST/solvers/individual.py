import numpy as np
import torch

from .base import Solver
from .model_lenet import RegressionModel, RegressionTrain

from time import time
from datetime import timedelta


class Individual(Solver):
    @property
    def name(self):
        return "individual"

    def update_fn(self, X, ts, model, optimizer):
        # Update using only j th task
        optimizer.zero_grad()
        task_j_loss = model(X, ts, self.j)
        task_j_loss.backward()
        optimizer.step()

    def run(self):
        """Run individual"""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        results = dict()
        for j in range(2):
            s_t = time()
            init_weight = np.array([1 - j, j])
            model = RegressionTrain(RegressionModel(self.flags.n_tasks), init_weight)
            if torch.cuda.is_available():
                model.cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.0)
            self.j = j
            res, checkpoint = self.train(model, optimizer)
            results[j] = {
                "r": init_weight,
                "res": res,
                "checkpoint": checkpoint,
            }
            t_t = timedelta(seconds=round(time() - s_t))
            print(f"**** Time taken for {self.dataset}_{j} = {t_t}")
        self.dump(results, self.prefix + ".pkl")
        total = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total}")
