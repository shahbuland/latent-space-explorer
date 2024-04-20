import random
import math
import torch

def random_circle_init(min_r : float = 0.5, on_edge : bool = False):
    theta = random.uniform(0, 2 * math.pi)
    if on_edge:
        r = 1.0
    else:
        r = random.uniform(min_r, 1.0)
    x = r * math.cos(theta)
    y = r * math.sin(theta)

    return x, y

def recursive_find_dtype(x):
    """
    Assuming x is some list/tuple of things that could be tensors, searches for any tensors and returns dtype
    """
    for i in x:
        if isinstance(i, list):
            res = recursive_find_dtype(i)
            if res is None:
                continue
            else:
                return res
        elif isinstance(i, torch.Tensor):
            return i.dtype

def recursive_find_device(x):
    """
    Assuming x is some list/tuple of things that could be tensors, searches for any tensors and returns device
    """
    for i in x:
        if isinstance(i, list):
            res = recursive_find_device(i)
            if res is None:
                continue
            return res
        elif isinstance(i, torch.Tensor):
            return i.device
