# import sys
# import os
import numpy as np
# import models

import torch
import torchvision.transforms
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from lib.config import cfg, update_config  # 你的配置模块
from lib.core.inference import get_multi_stage_outputs
from lib.core.inference import aggregate_results
from lib.core.nms import pose_nms
from lib.core.match import match_pose_to_heatmap

from lib.utils.transforms import resize_align_multi_scale
from lib.utils.transforms import get_final_preds
from lib.utils.transforms import get_multi_scale_size
# from lib.utils.rescore import rescore_valid
# from lib.dataset import make_test_dataloader

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def dekr_detect(image, cfg, model):
    # cudnn related setting
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    base_size, center, scale = get_multi_scale_size(image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0)
    heatmap_sum = 0
    poses = []

    # 多尺度推理
    for scale_factor in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
        image_resized, center, scale_resized = resize_align_multi_scale(
            image, cfg.DATASET.INPUT_SIZE, scale_factor, 1.0
        )
        input_tensor = transforms(image_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            heatmap, posemap = get_multi_stage_outputs(cfg, model, input_tensor, cfg.TEST.FLIP_TEST)
            heatmap_sum, poses = aggregate_results(cfg, heatmap_sum, poses, heatmap, posemap, scale_factor)

    heatmap_avg = heatmap_sum / len(cfg.TEST.SCALE_FACTOR)
    poses, scores = pose_nms(cfg, heatmap_avg, poses)

    if len(poses) == 0:
        return None, None, None
    else:
        poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)
        final_poses = get_final_preds(poses, center, scale_resized, base_size)
        # for pose in final_poses:
        #     print(np.mean(pose[:, 2]))
        #     print("-"*80)
        # 获取图像尺寸ß
        height, width, _ = image.shape
        # filtered_poses = [
        #     pose for pose in final_poses 
        #     if np.mean(pose[:, 2]) > 0.1
        # ]
        # if len(filtered_poses) == 0:
        #     return None, None, None
        # 2. 排序：按关键点的平均 x 坐标进行排序（从左到右）
        sorted_poses = sorted(final_poses, key=lambda pose: np.mean(pose[:, 2]), reverse=True)
        # 3. 归一化坐标（x / width, y / height），置信度保持不变
        normalized_poses = []
        for pose in sorted_poses:
            normalized_pose = pose.copy()
            normalized_pose[:, 0] /= width   # x坐标归一化
            normalized_pose[:, 1] /= height  # y坐标归一化
            normalized_poses.append(normalized_pose)

        # 返回标准化后的人体关键点，以及图像尺寸
        return normalized_poses, width, height
