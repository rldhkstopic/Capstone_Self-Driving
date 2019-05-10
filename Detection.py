from __future__ import division

import time
import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
import os
import os.path as osp
import pickle as pkl
import pandas as pd
import random

from util import *
from Lane import *
from darknet import Darknet
from torch.autograd import Variable

start = 0
batch_size = 1
confidence = 0.5
nms_thesh = 0.4
resol = 416 # 해상도
color = [10, 10, 0]
font = cv2.FONT_HERSHEY_PLAIN

cfg = "cfg/yolov3.cfg"
weights = "weights/yolov3.weights"
image_name = "Drive.mp4"
image_directory = "test_videos/"
video = image_directory + image_name

num_classes = 80
classes = load_classes("data/capstone.names")
CUDA = torch.cuda.is_available()

print("Reading cfg files..")
model = Darknet(cfg)
print("Reading weights files..")
model.load_weights(weights)
print("Network successfully loaded!\n")

model.net_info["height"] = resol
input_dim = int(model.net_info["height"])
assert input_dim % 32 == 0, "Please, Set the Input resolution which can divide into 32 parts"
assert input_dim > 32, "Please, Set the Input resolution to above 32"

if CUDA:
    model. cuda()

model.eval()

def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])

    image = results
    label = "{0}".format(classes[cls])
    cv2.rectangle(image, c1, c2,color, 1)

    t_size = cv2.getTextSize(label, font, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4

    cv2.rectangle(image, c1, c2, color, -1)
    cv2.putText(image, label, (c1[0], c1[1] + t_size[1] + 4), font, 1, [225,255,255], 1);
    return image

print("Now Loading the video..")
frames = 0
start = time.time()
cap = cv2.VideoCapture(video)
# clip1 = save_video('out_videos/detected_' + image_name)
print("Success!")
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cpframe = frame.copy()
        region = Lane_Detection(cpframe)

        image = prep_image(frame, input_dim)
        image_dim = frame.shape[1], frame.shape[0]
        image_dim = torch.FloatTensor(image_dim).repeat(1,2)

        if CUDA:
            image_dim = image_dim.cuda()
            image= image.cuda()

        with torch.no_grad():
            output = model(Variable(image, True), CUDA)
        output = write_results(output, confidence, num_classes, nms_thesh)

        if type(output) == int:
            frames += 1
            frame = cv2.addWeighted(frame, 1, region, 0.7, 0)
            cv2.imshow("frame", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        image_dim = image_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/image_dim,1)[0].view(-1,1)

        output[:, [1,3]] -= (input_dim - scaling_factor*image_dim[:,0].view(-1,1))/2
        output[:, [2,4]] -= (input_dim - scaling_factor*image_dim[:,1].view(-1,1))/2
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, image_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, image_dim[i,1])

        list(map(lambda x: write(x, frame), output))

        frame = cv2.addWeighted(frame, 1, region, 0.7, 0)
        cv2.imshow("frame", frame)
        # clip1.write(frame)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
        frames += 1
        now_fps = round(frames / (time.time() - start),2)
        print("Now FPS is {0}".format(now_fps))
    else:
        break
