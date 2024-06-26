from math import floor, ceil, comb # last one is choose
from copy import deepcopy

def _lerp(a, b, t):
    return a + t * (b - a)

def _multi_lerp(L, t):
    """
    Core of multi-lerp, assumes L is a simple list of tensors with no batch axis
    """
    if not isinstance(L, list):
        L = list(L) # Batch -> List

    t *= len(L) - 1
    
    min_ind = floor(t) # a = L[min_ind]
    max_ind  = ceil(t) # b = L[m]

    if t < 0:
        return _lerp(L[0], L[1], t)
    elif t > len(L) - 1:
        return _lerp(L[-2], L[-1], t) 
    elif floor(t) == min_ind:
        return L[min_ind]

    t = t % 1
    return _lerp(L[min_ind], L[max_ind], t)

def multi_lerp(L, t):
    """
    L can be a list of tensors, or a tensor
    """
    if isinstance(L, tuple) or isinstance(L, list):
        return (
            [_multi_lerp(L_i, t) if L_i is not None else None for L_i in L]
        )
    else:
        return _multi_lerp(L, t)

def _bezier(L, t, weights = None):
    """
    Core of bezier, assumes L is list of tensors with no batch axis

    :param weights: Weights for points between first and final in L
    """
    if not isinstance(L, list):
        L = list(L) # batched -> list

    if weights is None:
        weights = [1]*(len(L)-2)

    terms = L.copy()

    n = len(terms) - 1
    
    for i in range(1, len(terms) - 1):
        terms[i] = terms[i] * weights[i-1]
    
    for i in range(len(terms)):
        terms[i] = terms[i] * comb(n, i) * (1 - t) ** (n - i) * t ** i

    return sum(terms)
    
def bezier(L, t, weights = None):
    if isinstance(L, tuple) or isinstance(L, list):
        return (
            [_bezier(L_i, t, weights) if L_i is not None else None for L_i in L]
        )
    else:
        return _bezier(L, t, weights)

def partial_bezier(weights):
    """
    Return callable bezier with fixed weights
    """

    return lambda L, t: bezier(L, t, weights)
