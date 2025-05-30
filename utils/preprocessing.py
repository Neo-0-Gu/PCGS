import torch

def yolopose_preprocessing(keypoints:torch.tensor, conf:torch.tensor, img_w:int, img_h:int) -> list[int, int, list[list[float]| None]]:
    """
    Preprocess the keypoints from YOLO model output.
    Args:
        keypoints (torch.tensor): Keypoints tensor from YOLO model.
        img_w (int): Width of the image.
        img_h (int): Height of the image.
    Returns:
        List: A list containing width, height, and preprocessed keypoints as a list of lists.
    """
    pose = []
    # keypoints = [[int(kp[0].item()), int(kp[1].item())] if kp is not None else None for kp in keypoints]
    for i, keypoint in enumerate(keypoints):
        if conf[i] <= 0.5:  # Assuming the confidence score is the last element in the keypoint
            pose.append(None)
        else:
            pose.append([float(keypoint[0]), float(keypoint[1])])
    return [img_w, img_h, pose]

def dekr_preprocessing(keypoints:list, img_w:int, img_h:int) -> list[int, int, list[list[float]| None]]:
    pose = []
    for kp in keypoints:
        if kp is not None and kp[2] > 0:
            pose.append([kp[0], kp[1]])
        else:
            pose.append(None)
    return [img_w, img_h, pose]

def openpose_preprocessing(keypoints:list, img_w:int, img_h:int) -> list[int, int, list[list[float]| None]]:
    """
    Preprocess the keypoints from OpenPose model output.
    Args:
        keypoints (list): Keypoints list from OpenPose model.
        img_w (int): Width of the image.
        img_h (int): Height of the image.
    Returns:
        List: A list containing width, height, and preprocessed keypoints as a list of lists.
    """
    pose = []
    for kp in keypoints:
        if kp is not None and kp.score > 0:
            pose.append([kp.x, kp.y])
        else:
            pose.append(None)
    
    adjust_order = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    ad_pose = [pose[i] for i in adjust_order]
    return [img_w, img_h, ad_pose]

def adjust_kp(data: list[int, int, list[list[float]| None]]) -> list[list[float]| None]:
    """
    Adjust the keypoints to the specific order and return to the original size.
    Args:
        data (list): A list containing width, height, and keypoints.
    Returns:
        list: A list of adjusted keypoints, including None elements.
    """
    img_w, img_h, pose = data
    adjusted_order = [0, 1, 2, 3, 4, 5, 7, 9, 6, 8, 10, 11, 12, 13, 14, 15, 16]
    
    # Adjust keypoints while keeping None elements
    return [[kp[0] * img_w, kp[1] * img_h] if (kp := pose[i]) is not None else None for i in adjusted_order]



