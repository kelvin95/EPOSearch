def update_fn(X, ts, model, optimizer, flags):
    j = flags.individual.j
    # Update using only j th task
    optimizer.zero_grad()
    task_j_loss = model(X, ts, j)
    task_j_loss.backward()
    optimizer.step()


def run(dataset="mnist", base_model="lenet", niter=100):
    """
    run Pareto MTL
    """
    start_time = time()
    results = dict()
    out_file_prefix = f"indiv_{dataset}_{base_model}_{niter}"
    for j in range(2):
        s_t = time()
        res, model = train(dataset, base_model, niter, j)
        results[j] = {
            "r": np.array([1 - j, j]),
            "res": res,
            "checkpoint": model.model.state_dict(),
        }
        print(f"**** Time taken for {dataset}_{j} = {time() - s_t}")

    results_file = os.path.join("results", out_file_prefix + ".pkl")
    pickle.dump(results, open(results_file, "wb"))
    print(f"**** Time taken for {dataset} = {time() - start_time}")


if __name__ == "__main__":
    run(dataset="mnist", base_model="lenet", niter=100)
    run(dataset="fashion", base_model="lenet", niter=100)
    run(dataset="fashion_and_mnist", base_model="lenet", niter=100)
