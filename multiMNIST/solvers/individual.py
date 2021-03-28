import numpy as np
import torch

from .base import Solver

from time import time
from datetime import timedelta


class Individual(Solver):
    @property
    def name(self):
        return "individual"

    def update_fn(self, images, labels, model, optimizer):
        optimizer.zero_grad()
        task_loss = model(images, labels, self.task_id)
        task_loss.backward()
        optimizer.step()

    def run(self):
        """Run individual"""
        print(f"**** Now running {self.name} on {self.dataset}...")
        start_time = time()
        results = dict()
        for task_id in range(self.dataset_config.n_tasks):
            init_weight = np.zeros(self.dataset_config.n_tasks)
            init_weight[task_id] = 1

            self.task_id = task_id
            model = self.configure_model()
            optimizer = torch.optim.SGD(model.parameters(), lr=self.flags.lr, momentum=self.flags.momentum)

            result, checkpoint = self.train(model, optimizer)
            results[task_id] = dict(r=init_weight, res=result, checkpoint=checkpoint)

        self.dump(results, self.prefix + ".pkl")
        total_time = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")
