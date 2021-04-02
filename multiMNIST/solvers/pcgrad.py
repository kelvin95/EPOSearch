# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import torch

from .base import Solver
from .utils import flatten_grad, recover_flattened

from time import time
from datetime import timedelta


def get_d_pcgrad(gradients):
    final_grad = 0.0
    for grad_index, grad in enumerate(gradients):
        indices = np.arange(len(gradients))
        indices = np.concatenate([indices[:grad_index], indices[grad_index + 1:]])
        np.random.shuffle(indices)
        for index in indices:
            other_grad = gradients[index]
            cos_sim = torch.clamp(torch.dot(grad, other_grad), max=0)
            grad = grad - ((cos_sim / torch.linalg.norm(other_grad)) * other_grad)
        final_grad = final_grad + grad
    return final_grad


class PCGrad(Solver):
    @property
    def name(self):
        return "pcgrad"

    def update_fn(self, images, labels, model, optimizer):
        optimizer.zero_grad()

        # obtain and store the gradient
        flat_grads = {}
        task_losses = model(images, labels)
        for i in range(self.dataset_config.n_tasks):
            optimizer.zero_grad()
            task_losses[i].backward(retain_graph=True)
            flat_grads[i] = flatten_grad(model.parameters())

        # clear graph
        optimizer.zero_grad()
        del task_losses

        # calculate the gradient
        grads = torch.stack([flat_grads[i]["grad"] for i in range(len(flat_grads))])
        grads = get_d_pcgrad(grads)
        grads = recover_flattened(
            grads, flat_grads[0]["indices"], flat_grads[0]["shapes"]
        )

        # optimization step
        optimizer.zero_grad()
        for i, params in enumerate(model.parameters()):
            if grads[i] is not None:
                params.grad = grads[i]
        optimizer.step()

    def run(self):
        """Run PCGrad."""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        results = dict()
        for i in range(self.flags.n_preferences):
            self.suffix = f"p{i}"
            s_t = time()
            model = self.configure_model()
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.flags.lr, momentum=self.flags.momentum
            )

            result, checkpoint = self.train(model, optimizer)
            results[i] = dict(r=None, res=result, checkpoint=checkpoint)
            self.dump(
                results, self.prefix + f"_{self.flags.n_preferences}_from_0-{i}.pkl"
            )

        total_time = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")
