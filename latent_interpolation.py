from math import floor, ceil, comb # last one is choose

def _lerp(a, b, t):
    return a + t * (b - a)

def _multi_lerp(L, t):
    """
    Core of multi-lerp, assumes L is a simple list of tensors (batched is ok)
    """
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
            _multi_lerp(L_i, t) for L_i in L
        )
    else:
        return _multi_lerp(L, t)

def _bezier(L, t, weights = None):
    """
    Core of bezier, assumes L is (batched?) list of tensors

    :param weights: Weights for points between first and final in L
    """
    if weights is None:
        weights = [1]*len(L)-2

    terms = L
    n = len(terms)
    for i in range(1, len(terms) - 1):
        terms[i] *= weights[i]
    
    for i in range(len(terms)):
        terms[i] = terms[i] * choose(n, i) * (1 - t) ** (n - i) * t ** i

    return sum(terms)
    
def bezier(L, t, weights = None):
    if isinstance(L, tuple) or isinstance(L, list):
        return (
            _bezier(L_i, t) for L_i in L
        )
    else:
        return _bezier(L, t)

    