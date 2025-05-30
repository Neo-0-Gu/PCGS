from controlnet_aux import OpenposeDetector
import cv2
import os
import sys
from attack import *
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decoder import openpose_decode
from concurrent.futures import ThreadPoolExecutor, as_completed

def test(label,binary_code):
    correct = sum(p == g for p, g in zip(binary_code, label))
    accuracy = correct / len(binary_code)
    return accuracy

openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cuda:1")
image_folder = "/home/haoyi/vitton-hd"

# 所有攻击函数的字典，方便统一调用
attack_methods = {
    # "gaussian_noise": lambda img: add_gaussian_noise(img,0,0.005),
    # "gaussian_noise_2": lambda img: add_gaussian_noise(img,0,0.01),
    # "salt_pepper_noise": lambda img: add_salt_pepper_noise(img,0.001),
    # "salt_pepper_noise_2": lambda img: add_salt_pepper_noise(img,0.005),
    # "speckle_noise": lambda img: add_speckle_noise(img,0.01),
    # "speckle_noise_2": lambda img: add_speckle_noise(img,0.05),
    # "speckle_noise_3": lambda img: add_speckle_noise(img,0.1),
    # "median_filter": lambda img: median_filter(img, 3),
    # "median_filter_2": lambda img: median_filter(img, 5),
    # "median_filter_3": lambda img: median_filter(img, 7),
    # "mean_filter": lambda img: mean_filter(img, 3),
    # "mean_filter_2": lambda img: mean_filter(img, 5),
    # "mean_filter_3": lambda img: mean_filter(img, 7),
    # "gaussian_filter": lambda img: gaussian_filter(img, 3),
    # "gaussian_filter_2": lambda img: gaussian_filter(img, 5),
    # "gaussian_filter_3": lambda img: gaussian_filter(img, 7),
    # "jpeg_compression_50": lambda img: jpeg_compression(img, 50),
    # "jpeg_compression_90": lambda img: jpeg_compression(img, 90),
    # "gamma_correction": lambda img: gamma_correction(img, 0.8),
    "scaling": lambda img: scaling(img, 1.5),
    "center_crop_0.1": lambda img: center_crop(img, 0.1),
    # "center_crop_0.2": lambda img: center_crop(img, 0.2),
    "edge_crop_0.1": lambda img: edge_crop(img, 0.1),
    # "edge_crop_0.2": lambda img: edge_crop(img, 0.2),
    # "rotate_10": lambda img: rotate(img, 10),
    # "rotate_20": lambda img: rotate(img, 30),
    # "rotate_30": lambda img: rotate(img, 50),
    "translate_(16,10)": lambda img: translate(img, 16, 10),
    # "translate_(36,20)": lambda img: translate(img, 36, 20),
}

# 存储每种攻击的准确率
attack_accuracies = {}

# for attack_name, attack_fn in attack_methods.items():
#     accuracy_list = []
#     print(f"Running attack: {attack_name}")
#     for filename in tqdm(os.listdir(image_folder)):
#         if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
#             continue
#         img_path = os.path.join(image_folder, filename)
#         # print(img_path)
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         if img is None:
#             continue
#         attacked_img = attack_fn(img)
#         predicted_code = openpose_decode(attacked_img,openpose)
#         ground_truth_code = openpose_decode(img,openpose)
#         if(ground_truth_code is None):
#             continue
#         if(predicted_code is None):
#             accuracy_list.append(0)
#             continue
#         if len(predicted_code) != len(ground_truth_code):
#             predicted_code = openpose_decode(attacked_img,openpose,True)
#             ground_truth_code = openpose_decode(img,openpose,True)
#         acc = test(ground_truth_code, predicted_code)
#         accuracy_list.append(acc)

#     if accuracy_list:
#         acc = sum(accuracy_list) / len(accuracy_list)
#     attack_accuracies[attack_name] = acc
#     print(f"Accuracy for {attack_name}: {acc:.3f}")
def process_image(filename, attack_fn):
    img_path = os.path.join(image_folder, filename)
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return None

    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    attacked_img = attack_fn(img)

    # # 检测并编码
    # try:
    ground_truth_code = openpose_decode(img, openpose)
    predicted_code = openpose_decode(attacked_img, openpose)
    # except:
    #     return 0
    if ground_truth_code is None:
        return -1
    if predicted_code is None:
        return -1

    # if len(predicted_code) == len(ground_truth_code):
    #     len_t = len(predicted_code)
    #     predicted_code = openpose_decode(attacked_img, openpose, True, len_t+1)
    #     ground_truth_code = openpose_decode(img, openpose, True, len_t+1)
    # else:
    #     predicted_code = openpose_decode(attacked_img, openpose, True)
    #     ground_truth_code = openpose_decode(img, openpose, True)
    # if ground_truth_code is None:
    #     return -1
    # if predicted_code is None:
    #     return -1
    if len(predicted_code) != len(ground_truth_code):
        return -1
        # ground_truth_code = openpose_decode(img, openpose, True, 27)
        # predicted_code = openpose_decode(attacked_img, openpose, True, 27)
    if ground_truth_code is None:
        return -1
    if predicted_code is None:
        return -1
    acc = test(ground_truth_code, predicted_code)
    return acc


# 每种攻击并发处理
for attack_name, attack_fn in attack_methods.items():
    print(f"Running attack: {attack_name}")
    accuracy_list = []

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(process_image, filename, attack_fn)
            for filename in os.listdir(image_folder)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            acc = future.result()
            if acc is None or acc == -1:
                continue
            else:
                accuracy_list.append(acc)

    if accuracy_list:
        print(len(accuracy_list))
        acc = sum(accuracy_list) / len(accuracy_list)
        attack_accuracies[attack_name] = acc
        print(f"Accuracy for {attack_name}: {acc:.4f}")



# gaussian_noise:       0.870 # 0.992  0.2-0.9905   0.5-0.9912 0-0.9910  0.3-0.9903 0-0.9886 0.9934 0.9941 22-0.9939 23-0.9940 24-0.9937 25-0.9940 26 - 0.9935 27-0.99443  28-0.9935
# salt_pepper_noise:    0.879 # 0.992
# speckle_noise:        0.921 # 0.995
# median_filter:        0.865 # 0.994
# mean_filter:          0.850 # 0.994
# gaussian_filter:      0.876 # 0.994/2
# jpeg_compression_50:  0.847 # 0.989  
# jpeg_compression_90:  0.905   0.994
# gamma_correction:     0.882   0.990
# scaling:              0.910   0.996
# center_crop_0.1:      0.667   0.885
# center_crop_0.2:      0.476   0.565
# edge_crop_0.1: 0.917  0.721   0.917
# edge_crop_0.2:        0.600   0.767
# rotate_10:            0.751   0.956
# rotate_20:            0.714   0.934
# rotate_30:            0.683   0.897
# translate_(16,10):    0.768   0.958
# translate_(36,20):    0.755   0.956



