# pose generated from the results of detector
# example usage through yolo
from PIL import Image
from draw_pose import draw
from ultralytics import YOLO
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import yolopose_preprocessing

image = Image.open("../data/sample/mt_2.png")
yolo = YOLO("../model/pose_coco/yolo11x-pose.pt")
save_path = "../data/pose/yolo_pose_1.png"

result = yolo(image,verbose=False)
xyn = result[0].keypoints.xyn[0]
conf = result[0].keypoints.conf[0]
width, height = result[0].orig_shape[1], result[0].orig_shape[0]

data = yolopose_preprocessing(xyn, conf, width, height)
draw([data], 512, 512, 0.7).save(save_path)



