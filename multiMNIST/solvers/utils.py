import numpy as np
import torch
import sys
from contextlib import contextmanager


@contextmanager
def overload_print(name, mode):
    """Overload print function in this context with logging to file + stdout"""
    tee_file = TeeNoFile(name, mode)
    try:
        yield tee_file
    finally:
        tee_file.close()


# NOTE: do NOT inherit from file
class TeeNoFile(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def close(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        self.close()


def flatten_parameters(parameters):
    """flatten copy of parameters"""
    return torch.cat([torch.flatten(p.data.detach().clone()) for p in parameters])


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


def getNumParams(params):
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable


def rand_unit_vectors(
    ndims: int, num_vectors: int = 1, absolute: bool = True
) -> np.ndarray:
    """Return a uniformly random unit vector.

    Args:
        ndims (int): Number of dimensions.
        num_vectors (int): Number of vectors to return.
        absolute (bool): Whether to return absolute coordinates.

    Returns:
        (np.ndarray): [ndims] unit vector.
    """
    vector = np.random.randn(num_vectors, ndims)
    vector = vector / np.linalg.norm(vector, axis=-1)[:, None]
    return np.abs(vector) if absolute else vector


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20.0 if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20.0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


def circle_points_(r, n):
    """
    generate evenly distributed unit preference vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles


def cosine_angle(v: torch.Tensor, w: torch.Tensor) -> float:
    """Return the cosine angle between vectors v and w
    Args:
        v, w (torch.Tensor): Vectors between which angle is to be caclulated
    Returns:
        (float): cos(Θ)
    """
    return torch.dot(v, w) / (torch.norm(v) * torch.norm(w))


def gmsim(v: torch.Tensor, w: torch.Tensor) -> float:
    """Return the gradient magnitude similarity between gradients v and w
    Args:
        v, w (torch.Tensor)
    Returns:
        (float)
    """
    v_norm = torch.norm(v)
    w_norm = torch.norm(w)
    return 2 * v_norm * w_norm / (v_norm ** 2 + w_norm ** 2)


def mtc_bound(v: torch.Tensor, w: torch.Tensor) -> float:
    """Return the multi-task curvature bounding measure
    Args:
        v, w (torch.Tensor)
    Returns:
        (float)
    """
    angle = cosine_angle(v, w)
    return (1 - angle ** 2) * (torch.norm(v - w) ** 2 / torch.norm(v + w) ** 2)
