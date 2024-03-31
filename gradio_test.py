import gradio as gr
import numpy as np
from PIL import Image

from main import LatentSpaceExplorer
from latent_interpolation import bezier, multi_lerp, partial_bezier

explorer = LatentSpaceExplorer(compile = False)
explorer.set_curve(bezier)
def process_data(strings, slider_value):
    prompts = strings.split(",")
    prompts = [prompt.strip() for prompt in prompts]

    explorer.set_prompts(prompts)
    return explorer.draw_sample(slider_value)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(lines = 2)
            slider = gr.Slider(0,1)
        with gr.Column():
            output_img = gr.Image()

    slider.release(process_data, inputs = [input_box, slider], outputs = [output_img])

demo.launch(share = True)

