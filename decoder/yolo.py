import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import yolopose_preprocessing, adjust_kp
from .decode import decode


def yolo_decode(image, model,LSH: bool = False,LSH_length = 25) -> str:
    """YOLO解码函数"""
    result = model(image,verbose=False)
    if len(result) == 0:
        return None
    if result[0].keypoints is None:
        return None
    if result[0].keypoints.xyn is None:
        return None
    if result[0].keypoints.conf is None:
        return None
    if result[0].keypoints.xyn[0] is None:
        return None
    if result[0].keypoints.conf[0] is None:
        return None
    xyn = result[0].keypoints.xyn[0]
    conf = result[0].keypoints.conf[0]
    width, height = result[0].orig_shape[1], result[0].orig_shape[0]
    data = yolopose_preprocessing(xyn, conf, width, height) 
    kp = adjust_kp(data)
    binary_code = decode(kp, LSH, LSH_length)
    return binary_code