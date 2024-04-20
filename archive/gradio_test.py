import gradio as gr
import numpy as np
from PIL import Image

from main import LatentSpaceExplorer
from latent_interpolation import bezier, multi_lerp, partial_bezier

help = "Usage: Enter prompts separated by commas in input box. \
    \nAt least 2 prompts must be given. \
    \nA bezier curve will be generated between the first and last prompt using intermediate prompts as control points. \
    \nYou can vary where you are along this curve with the left slider. \
    \nYou can vary the influence of each control point with a corresponding slider. \
    \nYou can see an interactive visualization of the curve and where you are below the slider. \
    \nNote this just illustrative and in no way represents the space/shape of the true curve. \
    \nThe image will update whenever you move the T slider or when you move a weight slider"

# Global constants
N_SLIDERS = 10

T_MIN = 0
T_MAX = 1
SLIDER_STEP = 0.001

WEIGHT_MIN = 0
WEIGHT_MAX = 5

# Default with explorer on a standard bezier curve
explorer = LatentSpaceExplorer(compile = False)
explorer.set_curve(bezier)

# After main t slider is moved, update generated image
def process_data(strings, slider_value):
    try:
        prompts = strings.split(",")
        prompts = [prompt.strip() for prompt in prompts]
        assert len(prompts) >= 2
    except:
        return None

    if prompts != explorer.prompts:
        explorer.set_prompts(prompts)
    return explorer.draw_sample(slider_value), explorer.visualize(slider_value)

# After weight sliders are moved, update generated image
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
    if prompts != explorer.prompts:
        explorer.set_prompts(prompts)

    return explorer.draw_sample(t_slider_value), explorer.visualize(t_slider_value)

# Update which weight sliders are visible after changing prompts
def update_weight_visibility(strings, *slider_values):
    visibilities = [False] * len(slider_values)
    labels = [None] * len(slider_values)

    try:
        prompts = strings.split(",")
        prompts = [prompt.strip() for prompt in prompts]
        assert len(prompts) >= 2
    except:
        return [gr.Slider(WEIGHT_MIN,WEIGHT_MAX, val, visible = False, label = "T") for val in slider_values]
    
    for i in range(1, len(prompts) - 1):
        visibilities[i-1] = True
        labels[i-1] = prompts[i]

    if prompts != explorer.prompts:
        explorer.set_prompts(prompts)

    
    return [gr.Slider(
        WEIGHT_MIN, WEIGHT_MAX,
        slider_values[i], visible = visibilities[i], label = labels[i]
        ) for i in range(len(slider_values))]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(lines = 2, value = "a photo of a dog, an illustration of a lovecraftian horror floating in the skys above mars, a badass sports car in front of a fancy mansion in hollywood, a photo of a cat")
            update_weights_btn = gr.Button(value = 'Update Weights')
            with gr.Row():
                with gr.Column():
                    slider = gr.Slider(T_MIN,T_MAX, step = SLIDER_STEP)
                    vis = gr.Image()
                with gr.Column():
                    weight_sliders = [gr.Slider(WEIGHT_MIN, WEIGHT_MAX, value = 1, visible = False) for _ in range(N_SLIDERS)]
            gr.Textbox(help, interactive = False)
        with gr.Column():
            output_img = gr.Image()

    slider.release(process_data, inputs = [input_box, slider], outputs = [output_img, vis])
    for i in range(N_SLIDERS):
        weight_sliders[i].release(modify_weights, inputs = [input_box, slider] + weight_sliders, outputs = [output_img, vis])
    update_weights_btn.click(update_weight_visibility, inputs = [input_box] + weight_sliders, outputs = weight_sliders)

demo.launch(share = True)

