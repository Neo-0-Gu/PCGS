import requests
import base64
import os
import json

# === 配置 ===
API_URL = "http://127.0.0.1:7860/sdapi/v1/txt2img"
PROMPT_FILE = "sd_prompts_1000.txt"
A_DETAILER_ARGS_FILE = "adetailer_args.json"
OUTPUT_DIR = "source_pics"
SECONDARY_OUTPUT_DIR = "stego_pics"
NEGATIVE_PROMPT = (
    "blurry, cropped body, missing hands, missing legs, extra limbs, distorted limbs, bad anatomy, "
    "poor lighting, low quality, monochrome, anime style, cartoon, painting, sketch, unrealistic, "
    "text, watermark, lowres, artifacts"
)

# === 准备输出目录 ===
os.makedirs(SECONDARY_OUTPUT_DIR, exist_ok=True)

# === 加载 prompts ===
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# === 加载 ADetailer 参数 ===
with open(A_DETAILER_ARGS_FILE, "r") as f:
    adetailer_args = json.load(f)

# === 逐图作为 controlnet 控制图进行生成 ===
for i, prompt in enumerate(prompts):
    control_image_path = os.path.join(OUTPUT_DIR, f"image_{i+1:04d}.png")

    if not os.path.exists(control_image_path):
        print(f"[!] 控制图 {control_image_path} 不存在，跳过。")
        raise ValueError(f"Control image not found: {control_image_path}")

    with open(control_image_path, "rb") as image_file:
        control_image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

    print(f"[*] 正在使用控制图 {control_image_path} 生成新图...")

    payload = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": 20,
        "cfg_scale": 7,
        "seed": 865123200 + i,
        "width": 512,
        "height": 512,
        "sampler_index": "DPM++ 2M",
        "scheduler": "Karras",
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "enabled": True,
                        "image": control_image_b64,
                        "pixel_perfect": True,
                        "model": "control_v11p_sd15_openpose [cab727d4]",
                        "module": "openpose",
                        "weight": 1.0,
                        "start_control_step": 0,
                        "end_control_step": 1,
                        "control_mode": "Balanced",
                        "resize_mode": "Resize and Fill",
                        "use_controlnet_in_highres": True
                    }
                ]
            },
            "ADetailer": {
                "args": adetailer_args
            }
        }
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        image_data = response.json()["images"][0]
        image_base64 = image_data.split(",", 1)[1] if image_data.startswith("data:image") else image_data
        image_bytes = base64.b64decode(image_base64)

        image_path = os.path.join(SECONDARY_OUTPUT_DIR, f"controlled_{i+1:04d}.png")
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        print(f"[✓] 控制图生成成功：{image_path}")
    else:
        print(f"[X] 生成失败：{response.status_code} - {response.text}")