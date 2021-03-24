# This code is from
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
import os

import torch
import torch.utils.data
from torch.autograd import Variable

from model_lenet import RegressionModel, RegressionTrain

from min_norm_solvers import MinNormSolver
from time import time
import pickle


def get_d_mgda(vec):
    r"""Calculate the gradient direction for MGDA."""
    sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])
    return torch.tensor(sol).cuda().float()


def train(dataset, base_model, niter, npref, init_weight, pref_idx):

    # generate #npref preference vectors
    n_tasks = 2

    # load dataset

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


    # choose different optimizer for different base model
    if base_model == 'lenet':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45,60,75,90], gamma=0.5)

    if base_model == 'resnet18':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)


    # store infomation during optimization
    weights = []
    task_train_losses = []
    train_accs = []

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

            # obtain and store the gradient
            grads = {}
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts)
                task_loss[i].backward()

                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment
                grads[i] = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))



            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)

            # calculate the weights
            weight_vec = get_d_mgda(grads)

            # optimization step
            optimizer.zero_grad()
            for i in range(len(task_loss)):
                task_loss = model(X, ts)
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]

            loss_total.backward()
            optimizer.step()


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

                weights.append(weight_vec.cpu().numpy())

                print('{}/{}: weights={}, train_loss={}, train_acc={}'.format(
                        t + 1, niter,  weights[-1], task_train_losses[-1],train_accs[-1]))

    result = {"training_losses": task_train_losses,
              "training_accuracies": train_accs}
    return result, model


def run(dataset = 'mnist',base_model = 'lenet', niter = 100, npref = 5):
    """
    run MGDA
    """
    start_time = time()
    init_weight = np.array([0.5 , 0.5 ])
    out_file_prefix = f"mgda_{dataset}_{base_model}_{niter}_{npref}_from_0-"
    results = dict()
    for i in range(npref):
        s_t = time()
        res, model = train(dataset, base_model, niter, npref, init_weight, i)
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

