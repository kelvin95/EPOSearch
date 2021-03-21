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
def flatten_grad(grads):
    l = []
    indices = []
    shapes = []
    s = 0
    for grad in grads:
        if grad is None:
            shapes.append(None)
            continue
        shapes.append(grad.shape)
        grad = torch.flatten(grad)
        size = grad.shape[0]
        l.append(grad)
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
    final_grad = 0.
    for grad_index, grad in enumerate(gradients):
        indices = np.arange(len(gradients))
        indices = np.concatenate([indices[:grad_index], indices[grad_index + 1:]])
        np.random.shuffle(indices)
        for index in indices:
            other_grad = gradients[index]
            cos_sim = torch.clamp(torch.dot(grad, other_grad), max=0)
            grad = grad - ((cos_sim / torch.linalg.norm(other_grad)) * other_grad)
        final_grad = final_grad + grad
    return final_grad


def train(dataset, base_model, niter, npref, init_weight, pref_idx):

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


    # choose different optimizer for different base model
    lr = 1e-3

    # store infomation during optimization
    task_train_losses = []
    train_accs = []

    # run niter epochs of MGDA
    for t in range(niter):
        model.train()
        for (it, batch) in enumerate(train_loader):

            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            # compute shared gradients
            flat_grads = {}
            shared_params = [v for k, v in model.model.get_shared_parameters().items()]

            task_loss = model(X, ts)
            for i in range(n_tasks):
                shared_grads = torch.autograd.grad(task_loss[i], shared_params, retain_graph=True)
                flat_grads[i] = flatten_grad(shared_grads)

            # update task parameters
            for i in range(n_tasks):
                task_params = [v for k, v in model.model.get_task_parameters(i).items()]
                task_grads = torch.autograd.grad(task_loss[i], task_params, retain_graph=True)
                for index, params in enumerate(task_params):
                    params.data = params.data - lr * task_grads[index]

            # compute PCGrad
            pcgrads = [flat_grads[i]["grad"] for i in range(len(flat_grads))]
            pcgrads = get_d_pcgrad(torch.stack(pcgrads))
            pcgrads = recover_flattened(pcgrads, flat_grads[0]["indices"], flat_grads[0]["shapes"])

            # compute original gradients
            oggrads = [flat_grads[i]["grad"] for i in range(len(flat_grads))]
            oggrads = torch.mean(torch.stack(oggrads), dim=0)
            oggrads = recover_flattened(oggrads, flat_grads[0]["indices"], flat_grads[0]["shapes"])

            # compute transference
            gradient_candidates = [pcgrads, oggrads]
            transferences = get_transferences(model, (X, ts), gradient_candidates, task_loss, lr)
            gradients = gradient_candidates[torch.argmax(transferences).item()]

            # update shared parameters
            shared_params = [v for k, v in model.model.get_shared_parameters().items()]
            for index, params in enumerate(shared_params):
                params.data = params.data - lr * gradients[index]

            # clear the graph
            model.zero_grad()


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


                train_acc = np.stack([
                    1.0 * correct1_train / len(test_loader.dataset),
                    1.0 * correct2_train / len(test_loader.dataset)
                ])

                total_train_loss = torch.stack(total_train_loss)
                average_train_loss = torch.mean(total_train_loss, dim = 0)

            # record and print
            if torch.cuda.is_available():
                task_train_losses.append(average_train_loss.data.cpu().numpy())
                train_accs.append(train_acc)
                print('{}/{}: train_loss={}, train_acc={}'.format(
                        t + 1, niter,  task_train_losses[-1],train_accs[-1]))

    result = {"training_losses": task_train_losses,
              "training_accuracies": train_accs}
    return result, model


def run(dataset = 'mnist',base_model = 'lenet', niter = 100, npref = 5):
    """
    run IT-MTL with PCGrad
    """
    start_time = time()
    init_weight = np.array([0.5, 0.5])
    out_file_prefix = f"itmtl_{dataset}_{base_model}_{niter}_{npref}_from_0-"
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

