import cv2
import numpy as np
import os

# ========== 非几何攻击 ==========
# 高斯噪声
def add_gaussian_noise(img, mean=0, var=0.001):
    if img is None:
        raise FileNotFoundError("image is None.")
    h, w = img.shape[:2]
    noise = np.random.normal(mean, var**0.5, img.shape)
    noisy = img / 255.0 + noise
    noisy = np.clip(noisy, 0, 1)
    return (noisy * 255).astype(np.uint8)

# 椒盐噪声
def add_salt_pepper_noise(img, amount=0.001):
    if img is None:
        raise FileNotFoundError("image is None.")
    h, w = img.shape[:2]
    output = np.copy(img)
    probs = np.random.rand(*img.shape[:2])
    output[probs < amount] = 0
    output[probs > 1 - amount] = 255
    return output

# 斑点噪声
def add_speckle_noise(img, mean=0, var=0.01):
    if img is None:
        raise FileNotFoundError("image is None.")
    h, w = img.shape[:2]
    noise = np.random.normal(mean, var**0.5, img.shape)
    noisy = img / 255.0 + (img / 255.0) * noise
    noisy = np.clip(noisy, 0, 1)
    return (noisy * 255).astype(np.uint8)

# 中值滤波
def median_filter(img, filter_size=3):
    if img is None:
        raise FileNotFoundError("image is None.")
    h, w = img.shape[:2]
    return cv2.medianBlur(img, filter_size)

# 均值滤波
def mean_filter(img, filter_size=3):
    if img is None:
        raise FileNotFoundError("image is None.")
    h, w = img.shape[:2]
    return cv2.blur(img, (filter_size, filter_size))

# 高斯低通滤波
def gaussian_filter(img, filter_size=3):
    if img is None:
        raise FileNotFoundError("image is None.")
    h, w = img.shape[:2]
    return cv2.GaussianBlur(img, (filter_size, filter_size), 0)

# JPEG压缩
def jpeg_compression(img, quality):
    if img is None:
        raise FileNotFoundError("image is None.")
    h, w = img.shape[:2]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

# Gamma校正
def gamma_correction(img, gamma):
    if img is None:
        raise FileNotFoundError("image is None.")
    h, w = img.shape[:2]
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

# 缩放
def scaling(img, ratio):
    if img is None:
        raise FileNotFoundError("image is None.")
    h, w = img.shape[:2]
    scaled = cv2.resize(img, (int(w * ratio), int(h * ratio)))
    return cv2.resize(scaled, (w, h))  # 恢复原始尺寸

# ========== 几何攻击 ==========
# 居中裁剪
def center_crop(img, ratio):
    if img is None:
        raise FileNotFoundError("image is None.")
    h, w = img.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    return img[ch:h-ch, cw:w-cw]

# 边缘裁剪（模拟从边缘裁掉）
def edge_crop(img, ratio):
    if img is None:
        raise FileNotFoundError("image is None.")
    h, w = img.shape[:2]
    return img[0:h-int(h * ratio), int(w * ratio):]

# 旋转
def rotate(img, angle):
    if img is None:
        raise FileNotFoundError("image is None.")
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

# 平移
def translate(img, tx, ty):
    if img is None:
        raise FileNotFoundError("image is None.")
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))
