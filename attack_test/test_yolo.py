import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decoder import yolo_decode
from ultralytics import YOLO
from attack import *
from tqdm import tqdm

def test(label,binary_code):
    correct = sum(p == g for p, g in zip(binary_code, label))
    accuracy = correct / len(binary_code)
    return accuracy

yolo = YOLO("../model/pose_coco/yolo11x-pose.pt")
image_folder = "/home/haoyi/vitton-hd"

# 所有攻击函数的字典，方便统一调用
attack_methods = {
    "gaussian_noise": lambda img: add_gaussian_noise(img),
    # "salt_pepper_noise": lambda img: add_salt_pepper_noise(img),
    # "speckle_noise": lambda img: add_speckle_noise(img),
    # "median_filter": median_filter,
    # "mean_filter": mean_filter,
    # "gaussian_filter": gaussian_filter,
    # "jpeg_compression_50": lambda img: jpeg_compression(img, 50),
    # "jpeg_compression_90": lambda img: jpeg_compression(img, 90),
    # "gamma_correction": lambda img: gamma_correction(img, 0.8),
    # "scaling": lambda img: scaling(img, 1.5),
    # "center_crop_0.1": lambda img: center_crop(img, 0.1),
    # "center_crop_0.2": lambda img: center_crop(img, 0.2),
    # "edge_crop_0.1": lambda img: edge_crop(img, 0.1),
    # "edge_crop_0.2": lambda img: edge_crop(img, 0.2),
    # "rotate_10": lambda img: rotate(img, 10),
    # "rotate_20": lambda img: rotate(img, 20),
    # "rotate_30": lambda img: rotate(img, 30),
    # "translate_(16,10)": lambda img: translate(img, 16, 10),
    # "translate_(36,20)": lambda img: translate(img, 36, 20),
}

# 存储每种攻击的准确率
attack_accuracies = {}

for attack_name, attack_fn in attack_methods.items():
    accuracy_list = []
    print(f"Running attack: {attack_name}")
    for filename in tqdm(os.listdir(image_folder)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            continue
        attacked_img = attack_fn(img)
        predicted_code = yolo_decode(attacked_img,yolo)
        ground_truth_code = yolo_decode(img,yolo)
        if(ground_truth_code is None):
            continue
        if(predicted_code is None):
            # accuracy_list.append(0)
            continue
        if len(predicted_code) != len(ground_truth_code):
            predicted_code = yolo_decode(attacked_img,yolo,True)
            ground_truth_code = yolo_decode(img,yolo,True)
            # continue
        acc = test(ground_truth_code, predicted_code)
        accuracy_list.append(acc)

    if accuracy_list:
        print(len(accuracy_list))
        acc = sum(accuracy_list) / len(accuracy_list)
    attack_accuracies[attack_name] = acc
    print(f"Accuracy for {attack_name}: {acc:.3f}")




#gaussian_noise:        0.877 #0.983 0.990
#salt_pepper_noise:     0.928 #0.982
#speckle_noise:         0.936 #0.988
#median_filter:         0.882 #0.990
#mean_filter:           0.880 #0.982
#gaussian_filter:       0.897 #0.989
#jpeg_compression_50:   0.860 #0.984
#jpeg_compression_90:   0.924 #0.994
#gamma_correction:      0.950 #0.997
#scaling:               0.931 #0.994
#center_crop_0.1:       0.767 #0.884
#center_crop_0.2:       0.682 #0.365
#edge_crop_0.1:         0.787 #0.901
#edge_crop_0.2:         0.722 #0.456
#rotate_10:             0.786 #0.941
#rotate_20:             0.739 #0.870
#rotate_30:             0.707 #0.748
#translate_(16,10):     0.815 #0.979
#translate_(36,20):     0.815 #0.972