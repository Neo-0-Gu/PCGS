from .decode import decode
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import dekr_preprocessing, adjust_kp, dekr_detect

def dekr_decode(image, cfg, model, LSH: bool = False, LSH_length = 25) -> str:
    """HRNet解码函数"""
    final_poses, width, height = dekr_detect(image, cfg, model)
    if(final_poses is None):
        return None
    # print(len(final_poses))
    # print("-"*80)
    data = dekr_preprocessing(final_poses[0], width, height)
    kp = adjust_kp(data)
    binary_code = decode(kp, LSH, LSH_length)
    return binary_code