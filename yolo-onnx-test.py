#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:57:30 2024

@author: sushu
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import os
import time 
import onnx
import onnxruntime
import random

# 存储class_id和随机颜色的映射
random.seed(1024)
class_id_colors = {}

def get_random_color(class_id):
    # 如果class_id已经有颜色，则返回该颜色
    if class_id in class_id_colors:
        return class_id_colors[class_id]
    # 否则，生成一个新的随机颜色并存储
    else:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        class_id_colors[class_id] = color
        return color

def area_box(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def iou(box1, box2):
    left   = max(box1[0], box2[0])
    top    = max(box1[1], box2[1])
    right  = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])
    cross  = max((right-left), 0) * max((bottom-top), 0)
    union  = area_box(box1) + area_box(box2) - cross
    if cross == 0 or union == 0:
        return 0
    return cross / union

def NMS(boxes, iou_thres, box_area_thres):
    remove_flags = [False] * len(boxes)
    keep_boxes = []
    for i, ibox in enumerate(boxes):
        if remove_flags[i]:
            continue
        if area_box(ibox) < box_area_thres:
            continue
        keep_boxes.append(ibox)
        for j in range(i + 1, len(boxes)):
            if remove_flags[j]:
                continue
            jbox = boxes[j]
            if iou(ibox, jbox) > iou_thres:
                remove_flags[j] = True
    return keep_boxes

def draw_box(boxes, img_origin_resize,class_map_dict):
    for box in boxes:
        left, top, right, bottom, confidence, class_id = box
    
        # 绘制边界框
        start_point = (int(left), int(top))
        end_point = (int(right), int(bottom))
        color = get_random_color(class_id)
        thickness = 2
        cv2.rectangle(img_origin_resize, start_point, end_point, color, thickness)
    
        # 绘制类别和置信度
        class_name = class_map_dict[class_id] # 假设您有一个类别ID到名称的映射
        confidence = "{:.2f}".format(confidence)
        text = f"{class_name}: {confidence}"
        cv2.rectangle(img_origin_resize, (start_point[0]-2, start_point[1]-22), (start_point[0]+len(class_name)*8+55, start_point[1]-2), color, -1)
        cv2.putText(img_origin_resize, text, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255), 2)
    return img_origin_resize

def pad_to_square(img):
    image_shape = np.shape(img)
    if image_shape[0] == image_shape[1]:
        return img
    shape_max = np.max(image_shape)
    pad_col_left = int((shape_max-image_shape[0])/2)
    pad_col_right = shape_max-pad_col_left-image_shape[0]
    pad_row_left = int((shape_max-image_shape[1])/2)
    pad_row_right = shape_max-pad_row_left-image_shape[1]
    img_square = np.pad(img,((pad_col_left,pad_col_right),(pad_row_left,pad_row_right),(0,0)))
    return img_square

if __name__ == '__main__':
    
    import ast
    #%% 读取图像及前处理
    image_name = '000000000036'
    img_origin = cv2.imread('./pic/'+image_name+'.jpg')
    image_square = pad_to_square(img_origin)
    img_origin_resize = cv2.resize(image_square, (640,640))
    img_CHW = np.transpose(img_origin_resize,(2,0,1))
    img_NCHW = np.expand_dims(img_CHW, axis=0).astype(np.float32)
    # 前处理--归一化　此处应与训练中的前处理相同
    img_NCHW_nor = img_NCHW/255
    
    #%% 模型推理
    # 读取onnx模型转换为session
    model_name = 'yolo11n'  
    model = onnx.load('./model/'+model_name+'.onnx')
    model_Session = onnxruntime.InferenceSession(model.SerializeToString())
    # 获取input_name，input_shape,out_put_name
    model_input_name = model_Session.get_inputs()[0].name
    model_output_name = model_Session.get_outputs()[0].name
    model_input_shape = model_Session.get_inputs()[0].shape
    # 从model的Metadata之中获取classification类别和类别名
    model_meta = model_Session.get_modelmeta()
    classification = model_meta.custom_metadata_map['names']
    class_map_dict = ast.literal_eval(classification)
    # 模型推理
    result = model_Session.run(None,{'images':img_NCHW_nor})
    #%% 框筛选
    # 筛选阈值，conf_thres：置信度，iou_thres：交并比阈值（非极大值抑制），box_area_thres：最小面积阈值
    conf_thres = 0.35
    iou_thres = 0.4
    box_area_thres = 20
    # result = np.load('yolo_out_put.npy')
    result_t = np.transpose(result,(0,1,3,2))
    
    boxes = []
    IM= np.array([[1,0],[1,0]])
    # x,y,w,h转换为left,top,right,bottom
    for item in result_t[0,0]:
        cx, cy, w, h = item[:4]
        label = item[4:].argmax()
        confidence = item[4 + label]
        if confidence < conf_thres:
            continue
        left    = cx - w * 0.5
        top     = cy - h * 0.5
        right   = cx + w * 0.5
        bottom  = cy + h * 0.5
        boxes.append([left, top, right, bottom, confidence, label])
        
    boxes = np.array(boxes)
    # lr = boxes[:,[0, 2]]
    # tb = boxes[:,[1, 3]]
    # boxes[:,[0,2]] = IM[0][0] * lr + IM[0][1]
    # boxes[:,[1,3]] = IM[1][0] * tb + IM[1][1]
    boxes = sorted(boxes.tolist(), key=lambda x:x[4], reverse=True)
    A_post_result = NMS(boxes, iou_thres, box_area_thres)
    
    #%% 图像后处理及存图
    draw_box(A_post_result,img_origin_resize,class_map_dict)
    cv2.imwrite('./result/result_'+image_name+'.jpg', img_origin_resize)
