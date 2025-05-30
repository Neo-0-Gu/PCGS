import os
import json
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decoder import decode
from utils import adjust_kp

def process_pose_to_json(folder_path, output_json, image_folder_path):
    grouped_data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            # 获取图片路径（假设图片与txt文件同名但扩展名为.jpg）
            image_path = os.path.join(image_folder_path, filename.replace('.txt', '.jpg'))
            # 读取图片以获取其尺寸
            with Image.open(image_path) as img:
                width, height = img.size
            with open(os.path.join(folder_path, filename), 'r') as infile:
                for line in infile:
                    numbers = list(map(float, line.strip().split()))
                    seq = numbers[5:]
                    # 提取 (x, y) 坐标对

                    keypoints = []

                    for i in range(0, len(seq), 3):
                        if i + 2 < len(seq) and seq[i + 2] > 0:
                            keypoints.append([seq[i], seq[i + 1]])
                        else:
                            keypoints.append(None)

                    kp = adjust_kp([width, height, keypoints])
                    
                    code_sequence = decode(kp)

                    if(code_sequence == None):
                        continue

                    # 加入 json 数据结构中
                    keypoints = str(keypoints).replace('\n', '')  # 转为字符串，作为分组键    

                    sample = {
                        "image_index": filename.replace('.txt', '.jpg'),
                        "width": width,
                        "height": height,
                        "keypoints": keypoints
                    }

                    # 将 sample 添加到对应 pose_code 分组中
                    if code_sequence not in grouped_data:
                        grouped_data[code_sequence] = []
        
                    grouped_data[code_sequence].append(sample)
     # 构造最终 JSON 数据结构

    final_data = []
    for pose_code_str in sorted(grouped_data.keys()):
        final_data.append({
            "pose_code": pose_code_str,
            "samples": grouped_data[pose_code_str]
        })

    # 写入 JSON 文件
    with open(output_json, 'w') as outjson:
        json.dump(final_data, outjson, indent=2)

if __name__ == "__main__":
    folder_path = './coco_pose_label'
    output_json = 'pose_code_sequences.json'
    image_folder_path = './coco_pose_images'
    process_pose_to_json(folder_path, output_json, image_folder_path)
