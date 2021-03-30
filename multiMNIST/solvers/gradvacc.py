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


def compute_grad_coeff(cos_sim, ema_cos_sim, grad_k_norm, other_grad_k_norm):
    numerator = grad_k_norm * (
        ema_cos_sim * torch.sqrt(1 - (cos_sim ** 2))
        - cos_sim * torch.sqrt(1 - (ema_cos_sim ** 2))
    )
    denominator = other_grad_k_norm * torch.sqrt(1 - ema_cos_sim ** 2)
    return numerator / denominator


class GradVacc(Solver):
    @property
    def name(self):
        return "gradvacc"

    def get_d_gradvacc(self, gradients, flat_grads):
        final_grad = torch.zeros(flat_grads[0]["grad"].shape).cuda()
        grad_indices = flat_grads[0]["indices"]
        # For each set of parameters, update EMA and check sim
        for k, (s, e) in enumerate(grad_indices):
            for grad_index, grad in enumerate(gradients):
                grad_k = grad[s:e]
                indices = np.arange(len(gradients))
                indices = np.concatenate(
                    [indices[:grad_index], indices[grad_index + 1 :]]
                )
                np.random.shuffle(indices)
                grad_k_norm = torch.linalg.norm(grad_k)
                for index in indices:
                    other_grad = gradients[index][s:e]
                    other_grad_k_norm = torch.linalg.norm(other_grad)
                    cos_sim = torch.dot(grad_k, other_grad) / (grad_k_norm * other_grad_k_norm)
                    ema_cos_sim = self.grad_sim_dict[grad_index][index][k]
                    if cos_sim < ema_cos_sim:
                        coeff = compute_grad_coeff(
                            cos_sim, ema_cos_sim, grad_k_norm, other_grad_k_norm
                        )
                        grad_k = grad_k + (coeff * other_grad)
                    # Update EMA
                    self.grad_sim_dict[grad_index][index][k] = (
                        1 - self.beta
                    ) * self.grad_sim_dict[grad_index][index][k] + (self.beta * cos_sim)
                final_grad[s:e] = final_grad[s:e] + grad_k
        return final_grad

    def update_fn(self, images, labels, model, optimizer):
        optimizer.zero_grad()

        # obtain and store the gradient
        flat_grads = {}
        task_losses = model(images, labels)
        for i in range(self.dataset_config.n_tasks):
            optimizer.zero_grad()
            task_losses[i].backward(retain_graph=True)
            flat_grads[i] = flatten_grad(model.parameters())

        # calculate the gradient
        grads = torch.stack([flat_grads[i]["grad"] for i in range(len(flat_grads))])
        grads = self.get_d_gradvacc(grads, flat_grads)
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
        """Run GradVacc."""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        results = dict()
        beta = np.array([1e-2, 1e-3, 1e-1, 1e-4, 5e-2])

        for i in range(self.flags.n_preferences):
            s_t = time()
            self.beta = beta[i]

            model = self.configure_model()
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.flags.lr, momentum=self.flags.momentum
            )

            #  initial exponential moving average grad sim pairs
            optimizer.zero_grad()
            grad_sim_dict = {}
            indices = list()
            s = 0
            for k, param in enumerate(model.parameters()):
                size = torch.numel(param)
                indices.append((s, s + size))
                for m in range(self.dataset_config.n_tasks):
                    if m not in grad_sim_dict.keys():
                        grad_sim_dict[m] = {}
                    for j in range(self.dataset_config.n_tasks):
<<<<<<< HEAD
                        if j not in grad_sim_dict[m].keys():
                            grad_sim_dict[m][j] = dict()
                        if (m != j):
                            grad_sim_dict[m][j][k] = torch.tensor(0.0).cuda()
            
=======
                        if j not in grad_sim_dict[i].keys():
                            grad_sim_dict[i][j] = dict()
                        if i != j:
                            grad_sim_dict[i][j][k] = torch.tensor(0.0).cuda()

>>>>>>> 5bdd9ba595aadfcb219aa86296ff65d79ca56399
            self.grad_sim_dict = grad_sim_dict
            print(self.grad_sim_dict)
            result, checkpoint = self.train(model, optimizer)
            results[i] = dict(r=None, res=result, checkpoint=checkpoint)
<<<<<<< HEAD
            self.dump(results, self.prefix + f"_{self.flags.n_preferences}_from_0-{i}.pkl")
            for m in self.grad_sim_dict.keys():
                for j in self.grad_sim_dict[m].keys():
                    for k in self.grad_sim_dict[m][j].keys():
                        self.grad_sim_dict[m][j][k] = np.asarray(self.grad_sim_dict[m][j][k].detach().cpu())
            self.dump(self.grad_sim_dict, self.prefix + f"_{self.flags.n_preferences}_gradsim_dict_from_0-{i}.pkl")
=======
            self.dump(
                results, self.prefix + f"_{self.flags.n_preferences}_from_0-{i}.pkl"
            )
>>>>>>> 5bdd9ba595aadfcb219aa86296ff65d79ca56399

        total_time = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")
