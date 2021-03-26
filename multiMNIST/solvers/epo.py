import numpy as np
import torch
from torch.autograd import Variable

from .epo_lp import EPO_LP
from .base import Solver
from .model_lenet import RegressionModel, RegressionTrain
from .utils import circle_points, getNumParams

from time import time
from datetime import timedelta


class EPO(Solver):
    @property
    def name(self):
        return "epo"

    def epoch_start(self) -> None:
        self.descent = 0
        self.n_manual_adjusts = 0

    def epoch_end(self) -> None:
        print(f"\tdescent={self.descent/len(self.train_loader)}")
        if self.n_manual_adjusts > 0:
            print(f"\t # manual tweek={self.n_manual_adjusts}")

    def update_fn(self, X, ts, model, optimizer):
        # Obtain losses and gradients
        grads = {}
        losses = []
        for i in range(self.flags.n_tasks):
            optimizer.zero_grad()
            task_loss = model(X, ts)
            losses.append(task_loss[i].data.cpu().numpy())
            task_loss[i].backward()

            # One can use scalable method proposed in the MOO-MTL paper
            # for large scale problem; but we use the gradient
            # of all parameters in this experiment.
            grads[i] = []
            for param in model.parameters():
                if param.grad is not None:
                    grads[i].append(
                        Variable(param.grad.data.clone().flatten(), requires_grad=False)
                    )

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

        if torch.cuda.is_available():
            alpha = self.flags.n_tasks * torch.from_numpy(alpha).cuda()
        else:
            alpha = self.flags.n_tasks * torch.from_numpy(alpha)
        # Optimization step
        optimizer.zero_grad()
        task_losses = model(X, ts)
        weighted_loss = torch.sum(task_losses * alpha)  # * 5. * max(epo_lp.mu_rl, 0.2)
        weighted_loss.backward()
        optimizer.step()

    def run(self):
        """ Run Pareto MTL """
        print(f"**** Now running {self.name} on {self.dataset} ... ")
        start_time = time()
        model = RegressionTrain(
            RegressionModel(self.flags.n_tasks), np.array([0.5, 0.5])
        )
        _, n_params = getNumParams(model.parameters())
        print(f"# params={n_params}")
        npref = 5
        preferences = circle_points(
            npref, min_angle=0.0001 * np.pi / 2, max_angle=0.9999 * np.pi / 2
        )  # preference
        results = dict()
        for i, preference in enumerate(preferences[::-1]):
            s_t = time()
            model = RegressionTrain(RegressionModel(self.flags.n_tasks), preference)
            # Instantia EPO Linear Program Solver
            self.epo_lp = EPO_LP(m=self.flags.n_tasks, n=n_params, r=preference)
            self.preference = preference
            if torch.cuda.is_available():
                model.cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.0)
            # uses self.epo_lp and self.update_fn
            res, checkpoint = self.train(model, optimizer)
            results[i] = {"r": preference, "res": res, "checkpoint": checkpoint}
            t_t = timedelta(seconds=round(time() - s_t))
            print(f"**** Time taken for {self.dataset}_{i} = {t_t}")
            self.dump(results, self.prefix + f"_{npref}_{i}_from_0-.pkl")
        total = timedelta(seconds=round(time() - start_time))
        print(f"**** Time taken for {self.name} on {self.dataset} = {total}")
