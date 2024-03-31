from typing import List, Callable
from PIL import Image

from fast_sd import fast_diffusion_pipeline
from latent_interpolation import bezier, multi_lerp, partial_bezier

import torch
import numpy as np
from tqdm import tqdm

class LatentSpaceExplorer:
    def __init__(self, compile = False):
        self.pipe = fast_diffusion_pipeline(compile = compile)
        self.encodes = None

        self.curve = multi_lerp
    
    def fixed_seed(self, seed : int = 0):
        return torch.Generator('cuda').manual_seed(0)
    
    def set_curve(self, fn : Callable):
        """
        Set the type of curve we want to use for exploration
        """
        self.curve = fn

    def set_prompts(self, prompts : List[str]):
        """
        Set the internal cache of prompts
        """
        assert len(prompts) > 1, "Need more than one prompt to explore"

        self.encodes = self.pipe.get_encodes(prompts, generator = self.fixed_seed())

    def draw_sample(self, t):
        """
        Draw a sample at a specific t value
        """
        encodes = self.curve(self.encodes, t)
        res = self.pipe.generate_from_encodes(encodes, generator = self.fixed_seed()).images[0]
        return res

    def draw_path(self, min_t, max_t, n_steps = 50, progress_bar : bool = True) -> List[Image.Image]:
        t_values = np.linspace(min_t, max_t, n_steps)
        samples = [self.draw_sample(t) for t in tqdm(t_values)]
        return samples

if __name__ == "__main__":
    # Example showing how to use the model to explore the curve between "photograph of a dog" and "photograph of a cat", with the latent point "photograph of a car"

    prompts = [f"photograph of a {thing}" for thing in ["dog", "car", "cat"]]

    explorer = LatentSpaceExplorer()
    explorer.set_prompts(prompts)
    explorer.set_curve(partial_bezier([0.00001])) # Very low weight to car
    
    print("cp1")
    frames = explorer.draw_path(0, 1, n_steps = 10)
    