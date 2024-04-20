from abc import abstractmethod

import torch
import numpy as np

from .utils import recursive_find_device, recursive_find_dtype

class EncodingSampler:
    """
    Class to sample encodings given low dimensional spatial relationships.
    """
    def __init__(self, encodes):
        self.encodes = encodes

    def apply_coefs(self, coefs):
        """
        Linear combination of encodings given coefs
        """
        device = recursive_find_device(self.encodes)
        dtype = recursive_find_dtype(self.encodes)
        coefs = torch.from_numpy(coefs).to(device).to(dtype)

        def single_apply(encodes):
            if encodes is None:
                return None
            elif len(encodes.shape) == 3:
                return (coefs[:,None,None] * encodes).sum(0)
            elif len(encodes.shape) == 2:
                return (coefs[:,None] * encodes).sum(0)
            else:
                raise ValueError("Encoding Sampler couldn't figure out shape of encodings")
            
        if isinstance(self.encodes, list) or isinstance(self.encodes, tuple):
            return list(map(single_apply, self.encodes))
        else:
            return single_apply(self.encodes)

    @abstractmethod
    def __call__(self, point, other_points):
        """
        :param point: Point in low space representing user input ([2,] array)
        :param other_points: Points in low space representing existing prompts ([N,2] array)
        """
        pass

class DistanceSampling(EncodingSampler):
    """
    Sample based on distances between points in low dim space
    """
    def __call__(self, point, other_points):
        coefs = 1. / ((1. + np.linalg.norm(point[None,:] - other_points, axis = 1) ** 2))
        return self.apply_coefs(coefs)
    
class CircleSampling(EncodingSampler):
    """
    Sampler that views all encodings as points on a unit circle
    """
    def __call__(self, point, other_points):
        # Idea: weight of points in same direction should be 1
        # weight of points in opposite should be 0
        cos_sims = point @ other_points.transpose() # [2] x [2, N] -> N

        # Negative values don't work, but we want something analagous for "negative signals"
        # tanh is like -x for low values, but then caps out at 1
        #cos_sims = np.where(cos_sims<0, np.tanh(cos_sims), cos_sims)
        return self.apply_coefs(cos_sims)
