import numpy as np
import torch

from .base import Solver
from .utils import circle_points

from time import time
from datetime import timedelta


class LinScalar(Solver):
    @property
    def name(self):
        return "linscalar"

    def update_fn(self, X, ts, model, optimizer):
        if torch.cuda.is_available:
            alpha = torch.from_numpy(self.preference).cuda()
        else:
            alpha = torch.from_numpy(self.preference)
        # Optimization step
        optimizer.zero_grad()
        task_losses = model(X, ts)
        weighted_loss = torch.sum(task_losses * alpha)  # * 5. * max(epo_lp.mu_rl, 0.2)
        weighted_loss.backward()
        optimizer.step()

    def run(self):
        """Run linscalar"""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        npref = 5
        preferences = circle_points(
            5, min_angle=0.0001 * np.pi / 2, max_angle=0.9999 * np.pi / 2
        )  # preference
        results = dict()
        for i, preference in enumerate(preferences[::-1]):
            s_t = time()
            model = self.configure_model()
            if torch.cuda.is_available():
                model.cuda()
            self.preference = preference
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.0)
            res, checkpoint = self.train(model, optimizer)
            results[i] = {"r": preference, "res": res, "checkpoint": checkpoint}
            t_t = timedelta(seconds=round(time() - s_t))
            print(f"**** Time taken for {self.dataset}_{i} = {t_t}")
            self.dump(results, self.prefix + f"_{npref}_from_0-{i}.pkl")
        total = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total}")
