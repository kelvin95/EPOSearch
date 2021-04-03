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
        
        # obtain and store the gradient for each task
        flat_grads = list()
        for i in range(self.dataset_config.n_tasks):
            optimizer.zero_grad()
            last_layer = model.model.get_last_shared_layer()
            for name, param in last_layer:
                if name in ["weight"]:
                    gygw = torch.autograd.grad(task_losses[i], param, retain_graph=True)   
            # normalize
            flat_grads.append( gygw[0].flatten() / torch.linalg.norm(gygw[0].flatten()))
        # compute the cos similarity between tasks for the gradients on the shared parameters
        running_sum = torch.tensor(0.0).cuda()
        for i, i_grad in enumerate(flat_grads):
            for j, j_grad in enumerate(flat_grads): 
                if i != j:
                    cos_sim_2 = torch.dot(i_grad, j_grad) ** 2
                    running_sum += cos_sim_2
                    
                    self.running_grad_sim_dict[i][j]["running_sum"].append(np.asarray(cos_sim_2.detach().cpu()))
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        # optimizer.zero_grad()

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
            grad_sim_dict = {}
            for m in range(self.dataset_config.n_tasks):
                if m not in grad_sim_dict.keys():
                    grad_sim_dict[m] = {}
                for j in range(self.dataset_config.n_tasks):
                    if j not in grad_sim_dict[m].keys():
                        grad_sim_dict[m][j] = dict()
                        grad_sim_dict[m][j]["running_sum"] = list()
            
            self.running_grad_sim_dict = grad_sim_dict
            result, checkpoint = self.train(model, optimizer)

            results[i] = dict(r=preference, res=result, checkpoint=checkpoint)
            self.dump(results, self.prefix + f"_{self.flags.n_preferences}_from_0-{i}.pkl")
            self.dump(self.running_grad_sim_dict, self.prefix + f"_{self.flags.n_preferences}_gradsim_dict_from_0-{i}.pkl")

        total_time = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")
