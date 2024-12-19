#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:57:30 2024

@author: sushu
"""
from PIL import Image

from ultralytics import YOLO


# Load a pretrained YOLO11n model
model = YOLO("yolo11n.onnx")

# Define path to the image file
source = "bus.jpg"


# Run inference on the source
results = model(source)  # list of Results objects

# # View results
# for r in results:
#     print(r.boxes)  # print the Boxes object containing the detection bounding boxes

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")
    
# # Export the model
# model.export(format="onnx")