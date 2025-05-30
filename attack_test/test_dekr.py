import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decoder import dekr_decode
from attack import *
from tqdm import tqdm
import types
from lib.config import cfg, update_config
import torch
import torch.backends.cudnn as cudnn
import lib.models as models

def test(label,binary_code):
    correct = sum(p == g for p, g in zip(binary_code, label))
    accuracy = correct / len(binary_code)
    return accuracy

image_folder = "/home/haoyi/vitton-hd"
cfg_path = "../dekr.yaml"

args = types.SimpleNamespace(cfg=cfg_path,opts = [])
update_config(cfg, args)

cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    cfg, is_train=False
)
model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
model = model.to(device)
model.eval()

attack_methods = {
    "gaussian_noise": lambda img: add_gaussian_noise(img),
    "salt_pepper_noise": lambda img: add_salt_pepper_noise(img),
    "speckle_noise": lambda img: add_speckle_noise(img),
    "median_filter": median_filter,
    "mean_filter": mean_filter,
    "gaussian_filter": gaussian_filter,
    "jpeg_compression_50": lambda img: jpeg_compression(img, 50),
    "jpeg_compression_90": lambda img: jpeg_compression(img, 90),
    "gamma_correction": lambda img: gamma_correction(img, 0.8),
    "scaling": lambda img: scaling(img, 1.5),
    "center_crop_0.1": lambda img: center_crop(img, 0.1),
    "center_crop_0.2": lambda img: center_crop(img, 0.2),
    "edge_crop_0.1": lambda img: edge_crop(img, 0.1),
    "edge_crop_0.2": lambda img: edge_crop(img, 0.2),
    "rotate_10": lambda img: rotate(img, 10),
    "rotate_20": lambda img: rotate(img, 20),
    "rotate_30": lambda img: rotate(img, 30),
    "translate_(16,10)": lambda img: translate(img, 16, 10),
    "translate_(36,20)": lambda img: translate(img, 36, 20),
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
        # print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            continue
        attacked_img = attack_fn(img)
        predicted_code = dekr_decode(attacked_img,cfg,model)
        ground_truth_code = dekr_decode(img,cfg,model)
        if(ground_truth_code is None):
            continue
        if(predicted_code is None):
            accuracy_list.append(0)
            continue
        if len(predicted_code) != len(ground_truth_code):
            predicted_code = dekr_decode(attacked_img,cfg,model,True)
            ground_truth_code = dekr_decode(img,cfg,model,True)
        if(ground_truth_code is None):
            continue
        if(predicted_code is None):
            accuracy_list.append(0)
            continue
        acc = test(ground_truth_code, predicted_code)
        accuracy_list.append(acc)

    if accuracy_list:
        print(len(accuracy_list))
        acc = sum(accuracy_list) / len(accuracy_list)
    attack_accuracies[attack_name] = acc
    print(f"Accuracy for {attack_name}: {acc:.3f}")


# gaussian_noise:      0.824 0.849 0.912
# salt_pepper_noise:   0.915 0.921 0.911
# speckle_noise:       0.935 0.939 0.929
# median_filter:       0.809 0.844 0.942
# mean_filter:         0.777 0.793 0.930
# gaussian_filter:     0.821 0.834 0.935
# jpeg_compression_50: 0.721 0.735 0.925
# jpeg_compression_90: 0.883 0.887 0.952
# gamma_correction:    0.925 0.937 0.948
# scaling:             0.911 0.914 0.953
# center_crop_0.1      0.530 0.579 0.838
# center_crop_0.2      0.249 0.292 0.749
# edge_crop_0.1:       0.655 0.707 0.853
# edge_crop_0.2:       0.461 0.535 0.787
# rotate_10:           0.786 0.809 0.901
# rotate_20: 0.892     0.728 0.774 0.892
# rotate_30:           0.656 0.707 0.876
# translate_(16,10):   0.823 0.824 0.914
# translate_(36,20):   0.799 0.796 0.907