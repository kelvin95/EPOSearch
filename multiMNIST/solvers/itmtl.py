# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import torch

from .base import Solver
from .utils import flatten_grad, recover_flattened, flatten_param_grad

from time import time
from datetime import timedelta


@torch.no_grad()
def get_transferences(model, batch, list_of_gradients, initial_task_loss, lr):
    # NOTE: we assume task-specific gradients have been applied.
    initial_params = [v.clone() for k, v in model.model.get_shared_parameters().items()]

    transferences = []
    for gradients in list_of_gradients:
        # take one SGD step
        shared_params = [v for k, v in model.model.get_shared_parameters().items()]
        for index, param in enumerate(shared_params):
            param.data = initial_params[index] - lr * gradients[index]

        # compute transference
        task_loss = model(batch[0], batch[1])
        transference = torch.sum(1 - task_loss / initial_task_loss)
        transferences.append(transference)

    # reset original parameters
    shared_params = [v for k, v in model.model.get_shared_parameters().items()]
    for index, param in enumerate(shared_params):
        param.data = initial_params[index].data

    return torch.stack(transferences)


def get_d_pcgrad(gradients):
    final_grad = 0.0
    for grad_index, grad in enumerate(gradients):
        indices = np.arange(len(gradients))
        indices = np.concatenate([indices[:grad_index], indices[grad_index + 1 :]])
        np.random.shuffle(indices)
        for index in indices:
            other_grad = gradients[index]
            cos_sim = torch.clamp(torch.dot(grad, other_grad), max=0)
            grad = grad - ((cos_sim / torch.linalg.norm(other_grad)) * other_grad)
        final_grad = final_grad + grad
    return final_grad


class ITMTL(Solver):
    @property
    def name(self):
        return "itmtl"

    def update_fn(self, X, ts, model, optimizer):
        # clear the graph
        model.zero_grad()
        # compute shared gradients
        flat_grads = {}
        shared_params = [v for k, v in model.model.get_shared_parameters().items()]

        task_loss = model(X, ts)
        for i in range(self.dataset_config.n_tasks):
            shared_grads = torch.autograd.grad(
                task_loss[i], shared_params, retain_graph=True
            )
            flat_grads[i] = flatten_param_grad(shared_grads)

        # update task parameters
        for i in range(self.dataset_config.n_tasks):
            task_params = [v for k, v in model.model.get_task_parameters(i).items()]
            task_grads = torch.autograd.grad(
                task_loss[i],
                task_params,
                retain_graph=True,
            )
            for index, params in enumerate(task_params):
                params.data = params.data - self.flags.lr * task_grads[index]
                # add a gradient for metrics computation
                params.grad = task_grads[index]

        # compute PCGrad
        pcgrads = [flat_grads[i]["grad"] for i in range(len(flat_grads))]
        pcgrads = get_d_pcgrad(torch.stack(pcgrads))
        pcgrads = recover_flattened(
            pcgrads, flat_grads[0]["indices"], flat_grads[0]["shapes"]
        )

        # compute original gradients
        oggrads = [flat_grads[i]["grad"] for i in range(len(flat_grads))]
        oggrads = torch.mean(torch.stack(oggrads), dim=0)
        oggrads = recover_flattened(
            oggrads, flat_grads[0]["indices"], flat_grads[0]["shapes"]
        )

        # compute transference
        gradient_candidates = [pcgrads, oggrads]
        transferences = get_transferences(
            model, (X, ts), gradient_candidates, task_loss, self.flags.lr
        )
        gradients = gradient_candidates[torch.argmax(transferences).item()]

        # update shared parameters
        shared_params = [v for k, v in model.model.get_shared_parameters().items()]
        for index, params in enumerate(shared_params):
            params.data = params.data - self.flags.lr * gradients[index]
            # add a gradient for metrics computation
            params.grad = gradients[index]

    def run(self):
        """Run itmtl-pcgrad"""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        results = dict()
        for i in range(self.flags.n_preferences):
            self.suffix = f"p{i}"
            model = self.configure_model()
            result, checkpoint = self.train(model, None)
            results[i] = dict(r=None, res=result, checkpoint=checkpoint)
            self.dump(
                results, self.prefix + f"_{self.flags.n_preferences}_from_0-{i}.pkl"
            )

        total_time = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")
