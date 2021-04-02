import numpy as np
import torch

from .base import Solver
from .utils import rand_unit_vectors

from time import time
from datetime import timedelta


class LinScalar(Solver):
    @property
    def name(self):
        return "linscalar"

    def update_fn(self, images, labels, model, optimizer):
        optimizer.zero_grad()
        task_losses = model(images, labels)
        alpha = torch.from_numpy(self.preference).to(self.device)
        weighted_loss = torch.sum(task_losses * alpha)  # * 5. * max(epo_lp.mu_rl, 0.2)
        weighted_loss.backward()
        optimizer.step()

    def run(self):
        """Run linscalar"""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        results = dict()
        preferences = rand_unit_vectors(self.dataset_config.n_tasks, self.flags.n_preferences, True)
        for i, preference in enumerate(preferences):
            self.preference = preference
            self.suffix = f"p{i}"
            model = self.configure_model()
            optimizer = torch.optim.SGD(model.parameters(), lr=self.flags.lr, momentum=self.flags.momentum)
            result, checkpoint = self.train(model, optimizer)

            results[i] = dict(r=preference, res=result, checkpoint=checkpoint)
            self.dump(results, self.prefix + f"_{self.flags.n_preferences}_from_0-{i}.pkl")

        total_time = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")
