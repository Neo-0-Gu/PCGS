import math
import numpy as np
from matplotlib import pyplot as plt
import json

def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两向量间夹角(0-180度)"""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def compute_distance(array, i, j):
    dx = array[i][0] - array[j][0]
    dy = array[i][1] - array[j][1]
    return np.sqrt(dx**2 + dy**2)

def find_samples(json_file_path = "pose_code_sequences.json", target_pose_code = None):
    # 打开并加载 JSON 文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 遍历每个元素，查找目标 pose_code
    for entry in data:
        if entry['pose_code'] == target_pose_code:
            return entry['samples']
    
    # 如果找不到，返回 None
    return None