import numpy as np
import torch
from torch.autograd import Variable

from .epo_lp import EPO_LP
from .base import Solver
from .utils import rand_unit_vectors, getNumParams, circle_points

from time import time
from datetime import timedelta


class EPO(Solver):
    @property
    def name(self):
        return "epo"

    def epoch_start(self) -> None:
        self.descent = 0
        self.n_manual_adjusts = 0

    def epoch_end(self, epoch: int) -> None:
        print(f"\tdescent={self.descent/len(self.train_loader)}")
        if self.n_manual_adjusts > 0:
            print(f"\t # manual tweek={self.n_manual_adjusts}")

    def update_fn(self, images, labels, model, optimizer):
        optimizer.zero_grad()

        # Obtain losses and gradients
        grads = {}
        losses = []
        task_losses = model(images, labels)
        for i in range(self.dataset_config.n_tasks):
            optimizer.zero_grad()
            losses.append(task_losses[i].data.cpu().numpy())
            task_losses[i].backward(retain_graph=True)

            # One can use scalable method proposed in the MOO-MTL paper
            # for large scale problem; but we use the gradient
            # of all parameters in this experiment.
            grads[i] = []
            for param in model.parameters():
                if param.grad is not None:
                    grads[i].append(param.grad.data.clone().flatten())

        grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
        G = torch.stack(grads_list)
        GG = G @ G.T
        losses = np.stack(losses)

        try:
            # Calculate the alphas from the LP solver
            alpha = self.epo_lp.get_alpha(losses, G=GG.cpu().numpy(), C=True)
            if self.epo_lp.last_move == "dom":
                self.descent += 1
        except Exception as e:
            print(e)
            alpha = None

        if alpha is None:  # A patch for the issue in cvxpy
            alpha = self.preference / self.preference.sum()
            self.n_manual_adjusts += 1

        alpha = self.dataset_config.n_tasks * torch.from_numpy(alpha).to(self.device)

        # Optimization step
        optimizer.zero_grad()
        task_losses = model(images, labels)
        weighted_loss = torch.sum(task_losses * alpha)  # * 5. * max(epo_lp.mu_rl, 0.2)
        weighted_loss.backward()
        optimizer.step()

    def run(self):
        """ Run Pareto MTL """
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()

        model = self.configure_model()
        _, n_params = getNumParams(model.parameters())
        print(f"# params={n_params}")

        results = dict()
        if self.dataset_config.n_tasks == 2:
            preferences = circle_points(self.flags.n_preferences)
        else:
            preferences = rand_unit_vectors(self.dataset_config.n_tasks, self.flags.n_preferences, True)

        for i, preference in enumerate(preferences):
            self.suffix = f"p{i}"
            model = self.configure_model()
            optimizer = torch.optim.SGD(model.parameters(), lr=self.flags.lr, momentum=self.flags.momentum)

            # Instantia EPO Linear Program Solver
            self.epo_lp = EPO_LP(m=self.dataset_config.n_tasks, n=n_params, r=preference)
            self.preference = preference

            # uses self.epo_lp and self.update_fn
            result, checkpoint = self.train(model, optimizer)
            results[i] = dict(r=preference, res=result, checkpoint=checkpoint)
            self.dump(results, self.prefix + f"_{self.flags.n_preferences}_from_0-{i}.pkl")

        total_time = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total_time}s.")
