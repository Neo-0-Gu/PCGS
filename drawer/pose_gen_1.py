# pose generated from coco labels
# example usage:
import ast
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import find_samples, adjust_kp
from draw_pose import draw

tgt_code_1 = "0001000101010100111011"
tgt_code_2 = "0100010100101101011111"
file_path = "../data/pose_code_sequences.json"
save_path = "../data/pose/hybrid_pose.png"

samples_1 = find_samples(file_path, tgt_code_1)
samples_2 = find_samples(file_path, tgt_code_2)

if not samples_1:
    raise ValueError("No samples found for the given target codes 1.")
if not samples_2:
    raise ValueError("No samples found for the given target codes 2.")

sample_1 = samples_1[0]
width_1 = int(sample_1['width'])
height_1 = int(sample_1['height'])
keypoints_1 = ast.literal_eval(sample_1['keypoints'])

sample_2 = samples_2[0]
width_2 = int(sample_2['width'])
height_2 = int(sample_2['height'])
keypoints_2 = ast.literal_eval(sample_2['keypoints'])

draw([[width_1, height_1, keypoints_1],[width_2, height_2, keypoints_2]], 1024, 1024, 0.7).save(save_path)

