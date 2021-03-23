# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL


print("Script started.")

import numpy as np
import os

import torch
import torch.utils.data

from model_lenet import RegressionModel, RegressionTrain

from min_norm_solvers import MinNormSolver
from time import time
import pickle


# copied from https://discuss.pytorch.org/t/how-to-flatten-and-then-unflatten-all-model-parameters/34730
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
        indices.append((s, s+size))
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


# def get_d_pcgrad(gradients):
#     final_grad = 0.
#     for grad_index, grad in enumerate(gradients):
#         indices = np.arange(len(gradients))
#         indices = np.concatenate([indices[:grad_index], indices[grad_index + 1:]])
#         np.random.shuffle(indices)
#         for index in indices:
#             other_grad = gradients[index]
#             cos_sim = torch.clamp(torch.dot(grad, other_grad), max=0)
#             grad = grad - ((cos_sim / torch.linalg.norm(other_grad)) * other_grad)
#         final_grad = final_grad + grad
#     return final_grad


def train(dataset, base_model, niter, npref, init_weight, pref_idx, alpha=0.0):

    # generate #npref preference vectors
    n_tasks = 2

    # load dataset
    print(f"loading dataset {dataset}")

    # MultiMNIST: multi_mnist.pickle
    if dataset == 'mnist':
        with open('data/multi_mnist.pickle','rb') as f:
            trainX, trainLabel,testX, testLabel = pickle.load(f)

    # MultiFashionMNIST: multi_fashion.pickle
    if dataset == 'fashion':
        with open('data/multi_fashion.pickle','rb') as f:
            trainX, trainLabel,testX, testLabel = pickle.load(f)


    # Multi-(Fashion+MNIST): multi_fashion_and_mnist.pickle
    if dataset == 'fashion_and_mnist':
        with open('data/multi_fashion_and_mnist.pickle','rb') as f:
            trainX, trainLabel,testX, testLabel = pickle.load(f)

    trainX = torch.from_numpy(trainX.reshape(120000,1,36,36)).float()
    trainLabel = torch.from_numpy(trainLabel).long()
    testX = torch.from_numpy(testX.reshape(20000,1,36,36)).float()
    testLabel = torch.from_numpy(testLabel).long()


    train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
    test_set  = torch.utils.data.TensorDataset(testX, testLabel)


    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=batch_size,
                     shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))


    # define the base model for ParetoMTL
    if base_model == "lenet":
        model = RegressionTrain(RegressionModel(n_tasks), init_weight)
    if base_model == "resnet18":
        model = RegressionTrainResNet(MnistResnNet(n_tasks), init_weight)

    if torch.cuda.is_available():
        model.cuda()

    # initialize weights
    weights = torch.ones(n_tasks)
    if torch.cuda.is_available():
        weights = weights.cuda()
    weights.requires_grad_()

    # choose different optimizer for different base model
    if base_model == 'lenet':
        optimizer = torch.optim.SGD(list(model.parameters()) + [weights], lr=1e-3, momentum=0.)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45,60,75,90], gamma=0.5)

    if base_model == 'resnet18':
        optimizer = torch.optim.Adam(list(model.parameters()) + [weights], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)


    # store infomation during optimization
    task_train_losses = []
    train_accs = []

    # initialize loss
    initial_task_loss = None

    # run niter epochs of MGDA
    for t in range(niter):

        # scheduler.step()

        model.train()
        for (it, batch) in enumerate(train_loader):

            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            # compute loss
            optimizer.zero_grad()
            task_loss = model(X, ts)
            if initial_task_loss is None:
                initial_task_loss = task_loss.detach()

            # compute parameter gradients
            weighted_loss = torch.sum(task_loss * weights)
            weighted_loss.backward(retain_graph=True)
            weights.grad.data = weights.grad.data * 0.

            # compute gradient gradients
            grad_norms = []
            for i in range(len(task_loss)):
                grad = torch.autograd.grad(task_loss[i], model.model.parameters(), retain_graph=True)
                grad = torch.cat([torch.flatten(x) for x in grad])
                grad_norms.append(torch.linalg.norm(weights[i] * grad))
            grad_norms = torch.stack(grad_norms)

            mean_grad_norm = torch.mean(grad_norms.detach())
            loss_ratio = task_loss.detach() / initial_task_loss
            inverse_loss_ratio = loss_ratio / torch.mean(loss_ratio)
            weight_loss = torch.sum(torch.abs(grad_norms - mean_grad_norm * (inverse_loss_ratio ** alpha)))
            weights.grad.data = torch.autograd.grad(weight_loss, weights)[0]

            # SGD step
            optimizer.step()

            # normalize weights
            weights.data = weights.data / torch.norm(weights.data)

        # calculate and record performance
        if t == 0 or (t + 1) % 2 == 0:

            model.eval()
            with torch.no_grad():

                total_train_loss = []
                train_acc = []

                correct1_train = 0
                correct2_train = 0

                for (it, batch) in enumerate(test_loader):

                    X = batch[0]
                    ts = batch[1]
                    if torch.cuda.is_available():
                        X = X.cuda()
                        ts = ts.cuda()

                    valid_train_loss = model(X, ts)
                    total_train_loss.append(valid_train_loss)
                    output1 = model.model(X).max(2, keepdim=True)[1][:,0]
                    output2 = model.model(X).max(2, keepdim=True)[1][:,1]
                    correct1_train += output1.eq(ts[:,0].view_as(output1)).sum().item()
                    correct2_train += output2.eq(ts[:,1].view_as(output2)).sum().item()


                train_acc = np.stack([1.0 * correct1_train / len(test_loader.dataset),1.0 * correct2_train / len(test_loader.dataset)])

                total_train_loss = torch.stack(total_train_loss)
                average_train_loss = torch.mean(total_train_loss, dim = 0)

            # record and print
            if torch.cuda.is_available():
                task_train_losses.append(average_train_loss.data.cpu().numpy())
                train_accs.append(train_acc)
                print('{}/{}: weight={} train_loss={}, train_acc={}'.format(
                        t + 1, niter, weights, task_train_losses[-1],train_accs[-1]))

    result = {"training_losses": task_train_losses,
              "training_accuracies": train_accs}
    return result, model


def run(dataset = 'mnist',base_model = 'lenet', niter = 100, npref = 5):
    """
    run PCGrad
    """
    start_time = time()
    init_weight = np.array([0.5 , 0.5 ])
    alpha = np.arange(0., 1.1, 0.25)
    out_file_prefix = f"gradnorm_{dataset}_{base_model}_{niter}_{npref}_from_0-"
    results = dict()
    for i in range(npref):
        s_t = time()
        res, model = train(dataset, base_model, niter, npref, init_weight, i, alpha[i])
        results[i] = {"r": None, "res": res, "checkpoint": model.model.state_dict()}
        print(f"**** Time taken for {dataset}_{i} = {time() - s_t}")
        results_file = os.path.join("results", out_file_prefix + f"{i}.pkl")
        pickle.dump(results, open(results_file, "wb"))
    print(f"**** Time taken for {dataset} = {time() - start_time}")


if __name__ == "__main__":
    run(dataset = 'mnist', base_model = 'lenet', niter = 100, npref = 5)
    run(dataset = 'fashion', base_model = 'lenet', niter = 100, npref = 5)
    run(dataset = 'fashion_and_mnist', base_model = 'lenet', niter = 100, npref = 5)
    #run(dataset = 'mnist', base_model = 'resnet18', niter = 20, npref = 5)
    #run(dataset = 'fashion', base_model = 'resnet18', niter = 20, npref = 5)
    #run(dataset = 'fashion_and_mnist', base_model = 'resnet18', niter = 20, npref = 5)
