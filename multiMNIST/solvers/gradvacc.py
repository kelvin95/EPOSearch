# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import torch

from .base import Solver

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


def compute_grad_coeff(cos_sim, ema_cos_sim, grad_k_norm, other_grad_k_norm):
    numerator = grad_k_norm * (ema_cos_sim * torch.sqrt(1 - (cos_sim**2)) - cos_sim * torch.sqrt(1-(ema_cos_sim**2)))
    denominator = other_grad_k_norm * torch.sqrt(1 - ema_cos_sim**2)
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
                indices = np.concatenate([indices[:grad_index], indices[grad_index + 1 :]])
                np.random.shuffle(indices)
                grad_k_norm = torch.linalg.norm(grad_k)
                for index in indices:
                    other_grad = gradients[index][s:e]
                    other_grad_k_norm = torch.linalg.norm(other_grad)
                    cos_sim = torch.clamp(torch.dot(grad_k, other_grad), max=0) / (grad_k_norm * other_grad_k_norm)
                    ema_cos_sim = self.grad_sim_dict[grad_index][index][k]
                    if cos_sim < ema_cos_sim:
                        coeff = compute_grad_coeff(cos_sim, ema_cos_sim, grad_k_norm, other_grad_k_norm)
                        grad_k = grad_k + (coeff * other_grad)
                    # Update EMA
                    self.grad_sim_dict[grad_index][index][k] = (1 - self.beta) * self.grad_sim_dict[grad_index][index][k] + (self.beta * cos_sim)
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
        """Run PCGrad."""
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        results = dict()
        beta = np.array([1e-2, 1e-3, 1e-1])


        for i in range(self.flags.n_preferences):
            s_t = time()
            self.beta = beta[i]

            model = self.configure_model()
            optimizer = torch.optim.SGD(model.parameters(), lr=self.flags.lr, momentum=self.flags.momentum)

            #  initial exponential moving average grad sim pairs
            optimizer.zero_grad()
            grad_sim_dict = {}
            indices = list()
            s = 0
            for k, param in enumerate(model.parameters()):
                size = torch.numel(param)
                indices.append((s, s + size))
                for i in range(self.dataset_config.n_tasks):
                    if i not in grad_sim_dict.keys():
                        grad_sim_dict[i] = {}
                    for j in range(self.dataset_config.n_tasks):
                        if j not in grad_sim_dict[i].keys():
                            grad_sim_dict[i][j] = dict()
                        if (i != j):
                            grad_sim_dict[i][j][k] = torch.tensor(0.0).cuda()
            
            self.grad_sim_dict = grad_sim_dict
            print(self.grad_sim_dict)
            result, checkpoint = self.train(model, optimizer)
            results[i] = dict(r=None, res=result, checkpoint=checkpoint)
            self.dump(results, self.prefix + f"_{self.flags.n_preferences}_from_0-{i}.pkl")

        total_time = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")
