# This code is adapted from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import CrossEntropyLoss


class RegressionTrain(torch.nn.Module):

    def __init__(self, model, init_weight):
        super(RegressionTrain, self).__init__()

        self.model = model
        self.weights = torch.nn.Parameter(
            torch.from_numpy(init_weight).float())
        self.ce_loss = CrossEntropyLoss()

    def forward(self, x, ts, i=None):
        if i is not None:
            y = self.model(x, i)
            return self.ce_loss(y, ts[:, i])

        n_tasks = self.model.n_tasks
        ys = self.model(x)
        task_loss = []
        for i in range(n_tasks):
            task_loss.append(self.ce_loss(ys[:, i], ts[:, i]))
        task_loss = torch.stack(task_loss)

        return task_loss

    def randomize(self):
        self.model.apply(weights_init)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)


class RegressionModel(torch.nn.Module):
    def __init__(self, n_tasks):
        super(RegressionModel, self).__init__()
        self.n_tasks = n_tasks
        self.conv1 = nn.Conv2d(1, 10, 9, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 20, 50)

        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

    def forward(self, x, i=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 20)
        x = F.relu(self.fc1(x))

        if i is not None:
            layer_i = getattr(self, 'task_{}'.format(i))
            return layer_i(x)

        outs = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outs.append(layer(x))

        return torch.stack(outs, dim=1)

    def get_shared_parameters(self):
        return {k: v for k, v in self.named_parameters() if "task" not in k}

    def get_task_parameters(self, task):
        return {k: v for k, v in self.named_parameters() if f"task_{task}" in k}
