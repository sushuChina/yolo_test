#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:24:14 2024

@author: sushu
"""

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/sushu/code/python/yolo-test/datasets/tr_cigarette.yaml",
                      epochs=200,
                      imgsz=768)