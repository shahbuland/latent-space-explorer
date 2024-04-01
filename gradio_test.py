import gradio as gr
import numpy as np
from PIL import Image

from main import LatentSpaceExplorer
from latent_interpolation import bezier, multi_lerp, partial_bezier

explorer = LatentSpaceExplorer(compile = False)
explorer.set_curve(bezier)

def process_data(strings, slider_value):
    try:
        prompts = strings.split(",")
        prompts = [prompt.strip() for prompt in prompts]
        assert len(prompts) >= 2
    except:
        return None

    explorer.set_prompts(prompts)
    return explorer.draw_sample(slider_value), explorer.visualize(slider_value)

def modify_weights(strings, t_slider_value, *slider_values):
    weights = slider_values
    try:
        prompts = strings.split(",")
        prompts = [prompt.strip() for prompt in prompts]
        assert len(prompts) >= 2
    except:
        return None
    weights = weights[:len(prompts)-2]

    explorer.set_curve(partial_bezier(weights))
    explorer.set_prompts(prompts)
    return explorer.draw_sample(t_slider_value), explorer.visualize(t_slider_value)

N_SLIDERS = 10

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(lines = 2, value = "a photo of a dog, a photo of a cat")
            with gr.Row():
                with gr.Column():
                    slider = gr.Slider(0,1)
                    vis = gr.Image()
                with gr.Column():
                    weight_sliders = [gr.Slider(0, 1, value = 0) for _ in range(N_SLIDERS)]
        with gr.Column():
            output_img = gr.Image()

    slider.release(process_data, inputs = [input_box, slider], outputs = [output_img, vis])
    for i in range(N_SLIDERS):
        weight_sliders[i].release(modify_weights, inputs = [input_box, slider] + weight_sliders, outputs = [output_img, vis])

demo.launch()#share = True)

