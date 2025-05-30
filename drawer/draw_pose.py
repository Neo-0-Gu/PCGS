import numpy as np
import cv2
import math
from PIL import Image

def denormalize_and_scale(keypoints, orig_width, orig_height, target_box, scale_factor=0.7):
    """
    将归一化坐标的 keypoints 放大到原图尺寸，然后将包含 pose 的 bounding box 缩放并移动到指定区域，
    使 pose 只占该区域的一部分（由 scale_factor 控制）

    :param keypoints: List of (x, y) or None,归一化坐标
    :param orig_width: 原图宽度
    :param orig_height: 原图高度
    :param target_box: (x_start, y_start, region_width, region_height)
    :param scale_factor: 占区域大小的比例(0~1 之间)，默认 0.7
    :return: List of (x, y) or None, 映射后的坐标
    """
    x_start, y_start, region_w, region_h = target_box

    # 第一步：放大为原图坐标
    keypoints_abs = []
    for kp in keypoints:
        if kp is None:
            keypoints_abs.append(None)
        else:
            x, y = kp[0] * orig_width, kp[1] * orig_height
            keypoints_abs.append((x, y))

    # 第二步：计算包含有效关键点的 bounding box
    valid_points = [kp for kp in keypoints_abs if kp is not None]
    if not valid_points:
        return [None for _ in keypoints]

    min_x = min(p[0] for p in valid_points)
    max_x = max(p[0] for p in valid_points)
    min_y = min(p[1] for p in valid_points)
    max_y = max(p[1] for p in valid_points)

    box_w = max_x - min_x
    box_h = max_y - min_y

    if box_w == 0 or box_h == 0:
        return [None for _ in keypoints]

    # 第三步：计算缩放系数，加入 scale_factor
    effective_region_w = region_w * scale_factor
    effective_region_h = region_h * scale_factor
    scale = min(effective_region_w / box_w, effective_region_h / box_h)

    # 第四步：计算偏移量，使缩放后的 bbox 居中对齐到 target_box 中心
    new_box_w = box_w * scale
    new_box_h = box_h * scale
    offset_x = x_start + (region_w - new_box_w) / 2 - min_x * scale
    offset_y = y_start + (region_h - new_box_h) / 2 - min_y * scale

    # 第五步：变换 keypoints
    keypoints_transformed = []
    for kp in keypoints_abs:
        if kp is None:
            keypoints_transformed.append(None)
        else:
            x_new = kp[0] * scale + offset_x
            y_new = kp[1] * scale + offset_y
            keypoints_transformed.append((x_new, y_new))

    return keypoints_transformed

def draw_canvas(canvas: np.ndarray, keypoints) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    if keypoints[5] is not None and keypoints[6] is not None:
        im_x = (keypoints[5][0]+keypoints[6][0]) / 2
        im_y = (keypoints[5][1]+keypoints[6][1]) / 2
        keypoints.append([im_x, im_y])
    else:
        keypoints.append(None)

    stickwidth = 5

    limbSeq = [
        [17, 6], [17, 5], [6, 8], [8, 10], 
        [5, 7], [7, 9], [17, 12], [12, 14], 
        [14, 16], [17, 11], [11, 13], [13, 15], 
        [17, 0], [0, 2], [2, 4], [0, 1], 
        [1, 3],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index]
        keypoint2 = keypoints[k2_index]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1[0], keypoint2[0]]) 
        X = np.array([keypoint1[1], keypoint2[1]])
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint[0], keypoint[1]
        x = int(x)
        y = int(y)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas
    

def draw(data_sequence = None, canvas_width = 512, canvas_height = 512,scale_factor=0.7):
    """
    Draws a sequence of keypoints on a canvas.
        data_sequence: List of width, height, keypoints, where each keypoint is a list of (x, y) coordinates.
        canvas_width: Width of the canvas.
        canvas_height: Height of the canvas.
    """ 
    num = len(data_sequence)
    if num == 0:
        return
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    region_boxes = []
    for i in range(num):
        region_box = (i * canvas_width // num, 0, canvas_width // num, canvas_height)
        region_boxes.append(region_box)

    for i, (width, height, keypoints) in enumerate(data_sequence):
        target_box = region_boxes[i]
        scaled_keypoints = denormalize_and_scale(keypoints, width, height, target_box, scale_factor)
        canvas = draw_canvas(canvas, scaled_keypoints)
        
    pose_image = Image.fromarray(canvas)
    return pose_image


