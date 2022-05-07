import cv2
import torch
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

def inference(img):
    img = cv2.imread(img.name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    # img = Image.fromarray(formatted)
    depth_map = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    depth_map = (depth_map * 255 / np.max(depth_map)).astype('uint8')
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

    return depth_map

inputs =  gr.inputs.Image(type='file', label="Original Image")
outputs = gr.outputs.Image(type="pil",label="Output Image")

title = "DPT-Large"
description = "Gradio demo for DPT-Large:Vision Transformers for Dense Prediction.To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2103.13413' target='_blank'>Vision Transformers for Dense Prediction</a> | <a href='https://github.com/intel-isl/MiDaS' target='_blank'>Github Repo</a></p>"

examples=[['dog.jpg']]
gr.Interface(inference, inputs, outputs, title=title, description=description, article=article, analytics_enabled=False,examples=examples,    enable_queue=True).launch(debug=True)
