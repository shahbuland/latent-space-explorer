from diffusers import AutoencoderTiny, StableDiffusionXLPipeline
from .hacked_sdxl_pipeline import HackedSDXLPipeline
import torch

def fast_diffusion_pipeline(model_id = "stabilityai/sdxl-turbo", vae_id = "madebyollin/taesdxl", compile = False):
    """
    :param compile: If true, does a bunch of stuff to make calls fast, but the first call will be very slow as a consequence
        - If you use this, don't vary the batch size (probably)
    """

    pipe = HackedSDXLPipeline.from_pretrained(model_id, torch_dtype = torch.float16)
    pipe.set_progress_bar_config(disable=True)
    pipe.cached_encode = None
    pipe.vae = AutoencoderTiny.from_pretrained(vae_id, torch_dtype=torch.float16)

    pipe.to('cuda')

    if compile:
        pipe.unet = torch.compile(pipe.unet)
        pipe.vae.decode = torch.compile(pipe.vae.decode)
        """
        from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)

        config = CompilationConfig()
        config.enable_jit = True
        config.enable_jit_freeze = True
        config.trace_scheduler = True
        config.enable_cnn_optimization = True
        config.preserve_parameters = False
        config.prefer_lowp_gemm = True

        pipe = compile(pipe, config)
        """
    return pipe
