import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils import vector_angle, compute_distance
import math

def extract_features(keypoints: list):
    """提取角度和比例特征 (输入为list, 元素为xy坐标或None)"""
    features = {'angles': [], 'ratios': []}

    def calc_angle(a_idx, b_idx, c_idx):
        if all(keypoints[i] is not None for i in [a_idx, b_idx, c_idx]):
            v1 = np.array(keypoints[b_idx]) - np.array(keypoints[a_idx])
            v2 = np.array(keypoints[c_idx]) - np.array(keypoints[b_idx])
            return vector_angle(v1, v2)
        return None

    def calc_ratio(hip_idx, knee_idx, ankle_idx):
        if all(keypoints[i] is not None for i in [hip_idx, knee_idx, ankle_idx]):
            upper = np.linalg.norm(np.array(keypoints[knee_idx]) - np.array(keypoints[hip_idx]))
            lower = np.linalg.norm(np.array(keypoints[ankle_idx]) - np.array(keypoints[knee_idx]))
            return upper / (lower + 1e-6)
        return None

    # 左右肘角度
    for indices in [(5, 6, 7), (8, 9, 10)]:
        angle = calc_angle(*indices)
        if angle is not None:
            features['angles'].append(angle)

    # 右腿比例，若右腿缺失则尝试左腿
    ratio = calc_ratio(12, 14, 16) or calc_ratio(11, 13, 15)
    features['ratios'].append(ratio)

    return features

def quantize_features(features: list) -> str:
    """特征分箱量化（与原逻辑相同）"""
    binary_str = []
    
    # 角度分箱（网页7改进建议）
    for angle in features['angles']:
        if angle is None:
            continue
        bin_idx = int(angle // 12)
        binary_str.append(f"{bin_idx:04b}")
    
    # 比例分箱
    for ratio in features['ratios']:
        if ratio is None:
            continue
        # 截断比例到0.5到2之间
        ratio = max(0.5, min(ratio, 2))
        # 减去0.5再除以0.1
        bin_idx = int((ratio - 0.5) // 0.1)
        binary_str.append(f"{bin_idx:04b}")
    
    return ''.join(binary_str)

def get_feature(keypoints):
    
    kp = keypoints[4:17]
    dist = np.zeros(12)

    for i in range(12):
        if kp[i] is None or kp[i+1] is None:
            dist[i] = np.nan
        else:
            dist[i] = compute_distance(kp, i, i+1)

    feature = np.zeros(11)
    for i in range(0,11):
        if kp[i] is None or kp[i+1] is None or kp[i+2] is None:
            feature[i] = np.nan
            continue
        a = dist[i]
        b = dist[i+1]
        c = compute_distance(kp, i, i+2)
        # print(c)
        # print('-'*80)
        C = (a + b + c) / 2
        if C <= 0:
            return None
        im = C*(C-a)*(C-b)*(C-c)
        if np.any(im < 0):
            # raise ValueError("输入负数！")
            return None
        # print(im)
        T = np.sqrt(im) 

        feature[i] = math.pi*((T/C)**2)

    return feature

def generate_hash(binary_str: str, LSH_length: int = 25) -> str:
    """LSH哈希生成(与原逻辑相同)"""
    np.random.seed(7)
    # pdb.set_trace()
    proj_matrix = np.random.randn(LSH_length, len(binary_str))
    hash_bits = np.dot(proj_matrix, [int(b) for b in binary_str])
    return ''.join(['1' if bit > 0 else '0' for bit in hash_bits])

def decode(keypoints, LSH: bool = False, LSH_length: int = 25) -> str:

    """解码函数"""
    features_1 = extract_features(keypoints)
    # print(features) 
    code_str_1 = quantize_features(features_1)

    feature_2 = get_feature(keypoints)

    if feature_2 is None:
        return None
    
    length = len(feature_2)
    code_sequence = np.zeros(length-1, dtype=int)
    for i in range(length-1):
        if feature_2[i]>feature_2[i+1]:
            code_sequence[i]=0
        else:
            code_sequence[i]=1
    
    code_str_2 = ''.join(map(str, code_sequence))

    binary_str = code_str_1 + code_str_2
    
    if(len(binary_str) < 15):
        return None
    
    if LSH:
        # Perform LSH encoding
        final_str = generate_hash(binary_str,LSH_length)  # Truncate to LSH length
        # Add LSH encoding logic here if needed
    else:
        # Perform normal encoding
        final_str = binary_str
         
    return final_str