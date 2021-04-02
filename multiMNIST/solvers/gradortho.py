# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import torch

from .base import Solver
from .utils import rand_unit_vectors

from time import time
from datetime import timedelta

# copied from
# https://discuss.pytorch.org/t/how-to-flatten-and-then-unflatten-all-model-parameters/34730
def flatten_grad(parameters):
    l = []
    indices = []
    shapes = []
    s = 0
    for p in parameters:
        if p.grad is None:
            shapes.append(None)
            continue
        shapes.append(p.grad.shape)
        p = torch.flatten(p.grad)
        size = p.shape[0]
        l.append(p)
        indices.append((s, s + size))
        s += size
    flat = torch.cat(l).view(-1)
    return {"grad": flat, "indices": indices, "shapes": shapes}


def recover_flattened(flat_grad, indices, shapes):
    l = [flat_grad[s:e] for (s, e) in indices]
    grads = []
    index = 0
    for i in range(len(shapes)):
        if shapes[i] is None:
            grads.append(None)
            continue
        grads.append(l[index].view(shapes[i]))
        index += 1
    return grads


class GradOrtho(Solver):
    @property
    def name(self):
        return "gradortho"

    def update_fn(self, images, labels, model, optimizer):
        optimizer.zero_grad()
        task_losses = model(images, labels)
        alpha = torch.from_numpy(self.preference).to(self.device)
        weighted_loss = torch.sum(task_losses * alpha)  # * 5. * max(epo_lp.mu_rl, 0.2)
        weighted_loss.backward(retain_graph=True)

        # obtain and store the gradient for each task
        flat_grads = list()
        for i in range(self.dataset_config.n_tasks):
            # optimizer.zero_grad()
            last_layer = model.model.get_last_shared_layer()
            for name, param in last_layer:
                if name in ["weight"]:
                    gygw = torch.autograd.grad(task_losses[i], param, create_graph=True)

            # normalize
            flat_grads.append(gygw[0].flatten() / torch.linalg.norm(gygw[0].flatten()))

        # optimizer.zero_grad()
        # compute the cos similarity between tasks for the gradients on the shared parameters
        running_sum = torch.tensor(0.0, requires_grad=True).cuda()
        for i, i_grad in enumerate(flat_grads):
            for j, j_grad in enumerate(flat_grads):
                if i != j:
                    cos_sim_2 = torch.dot(i_grad, j_grad) ** 2
                    running_sum += cos_sim_2

                    self.running_grad_sim_dict[i][j]["running_sum"].append(
                        np.asarray(cos_sim_2.detach().cpu())
                    )

        # minimize the cos similarity
        cos_align_loss = (
            self.cos_penalty
            / (self.dataset_config.n_tasks * (self.dataset_config.n_tasks - 1))
            * running_sum
        )
        cos_align_loss.backward()
        optimizer.step()

    def run(self):
        """Run GradOrtho with Linear Scalarization."""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        results = dict()
        preferences = rand_unit_vectors(
            self.dataset_config.n_tasks, self.flags.n_preferences, True
        )

        for i, preference in enumerate(preferences):
            self.preference = preference
            self.suffix = f"p{i}"
            s_t = time()
            self.cos_penalty = 1
            model = self.configure_model()
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.flags.lr, momentum=self.flags.momentum
            )

            #  initial exponential moving average grad sim pairs
            optimizer.zero_grad()
            grad_sim_dict = {}
            for m in range(self.dataset_config.n_tasks):
                if m not in grad_sim_dict.keys():
                    grad_sim_dict[m] = {}
                for j in range(self.dataset_config.n_tasks):
                    if j not in grad_sim_dict[m].keys():
                        grad_sim_dict[m][j] = dict()
                        grad_sim_dict[m][j]["running_sum"] = list()

            self.running_grad_sim_dict = grad_sim_dict
            print(self.running_grad_sim_dict)
            result, checkpoint = self.train(model, optimizer)
            results[i] = dict(r=preference, res=result, checkpoint=checkpoint)
            self.dump(
                results,
                self.prefix
                + f"_{self.flags.n_preferences}_{self.cos_penalty}_from_0-{i}.pkl",
            )
            self.dump(
                self.running_grad_sim_dict,
                self.prefix
                + f"_{self.flags.n_preferences}_gradsim_{self.cos_penalty}_dict_from_0-{i}.pkl",
            )

        total_time = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")
