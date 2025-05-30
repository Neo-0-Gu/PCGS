from utils import openpose_preprocessing, adjust_kp
from .decode import decode

def openpose_decode(image, model, LSH: bool = False, LSH_length = 25) -> str:
    """OpenPose解码函数"""
    pose = model.detect_poses(image,0,0)
    if pose is None or len(pose) == 0:
        return None
    if pose[0].body is None:
        return None
    if pose[0].body.keypoints is None:
        return None
    data = openpose_preprocessing(pose[0].body.keypoints,image.shape[1], image.shape[0])
    kp = adjust_kp(data)
    binary_code = decode(kp, LSH, LSH_length)
    return binary_code