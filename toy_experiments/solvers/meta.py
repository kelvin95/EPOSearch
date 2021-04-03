from typing import Callable, Dict, Tuple

import jax
import jax.numpy as np
import numpy as onp


def loss1_fn(x: np.ndarray) -> np.ndarray:
    """Compute first biobjective loss function."""
    return 1 - np.exp(-np.linalg.norm(x - 1 / np.sqrt(x.shape[-1]), axis=-1) ** 2)


def loss2_fn(x: np.ndarray) -> np.ndarray:
    """Compute second biobjective loss function."""
    return 1 - np.exp(-np.linalg.norm(x + 1 / np.sqrt(x.shape[-1]), axis=-1) ** 2)


def concave_fun_eval(params: np.ndarray) -> np.ndarray:
    return np.stack([loss1_fn(params), loss2_fn(params)], axis=-1)


def meta_search(
    autograd_loss_fn: Callable,
    r: onp.ndarray,
    x: onp.ndarray,
    step_size: float = 1e-1,
    max_iters: int = 100,
    meta_step_size: float = 60.0,
    num_meta_updates: int = 1,
) -> Tuple[onp.ndarray, Dict]:
    # convert inputs to jax arrays
    params = np.asarray(x)
    preferences = np.asarray(r)
    task_weights = np.asarray(r)

    # use jax version of objective
    loss_fn = concave_fun_eval

    @jax.jit
    def task_uniformity_kl(params: np.ndarray, task_weights: np.ndarray) -> np.ndarray:
        # one-step task weight updates
        loss_with_task_weights = lambda x: np.sum(loss_fn(x) * task_weights)
        params_grad = jax.grad(loss_with_task_weights)(params)
        new_params = params - step_size * params_grad

        # compute normalized task losses
        task_losses = loss_fn(new_params) * preferences
        task_losses = task_losses / np.sum(task_losses)
        
        # compute kl-divergence
        return np.sum(task_losses * np.log(task_losses * len(task_weights)))

    @jax.jit
    def update_params(params: np.ndarray, task_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # adjust task weights to maximize uniformity
        grad_fn = jax.grad(task_uniformity_kl, argnums=1)
        for _ in range(num_meta_updates):
            task_weights_grad = grad_fn(params, task_weights)
            task_weights = jax.nn.relu(task_weights - meta_step_size * task_weights_grad)
            task_weights = task_weights / np.sum(task_weights)

        # compute new params based on updated task_weights
        loss_with_task_weights = lambda x: np.sum(loss_fn(x) * task_weights)
        params_grad = jax.grad(loss_with_task_weights, argnums=0)(params)
        new_params = params - step_size * params_grad
        return new_params, task_weights

    losses = []
    for _ in range(max_iters):
        params, task_weights = update_params(params, task_weights)
        losses.append(loss_fn(params))

    results = dict(ls=onp.array(np.stack(losses)))
    return params, results
