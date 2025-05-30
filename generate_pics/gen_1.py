import requests
import base64
import os
import json

# === 配置 ===
API_URL = "http://127.0.0.1:7860/sdapi/v1/txt2img"
PROMPT_FILE = "sd_prompts_1000.txt"
A_DETAILER_ARGS_FILE = "adetailer_args.json"
OUTPUT_DIR = "source_pics"
NEGATIVE_PROMPT = (
    "blurry, cropped body, missing hands, missing legs, extra limbs, distorted limbs, bad anatomy, "
    "poor lighting, low quality, monochrome, anime style, cartoon, painting, sketch, unrealistic, "
    "text, watermark, lowres, artifacts"
)

# === 准备输出目录 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 加载 prompts ===
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# === 加载 ADetailer 参数 ===
with open(A_DETAILER_ARGS_FILE, "r") as f:
    adetailer_args = json.load(f)

# === 循环生成图像 ===
for i, prompt in enumerate(prompts):
    print(f"[*] 正在生成图像 {i+1}/{len(prompts)}")

    alwayson_scripts = {
        "ADetailer": {"args": adetailer_args}
    }

    payload = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": 20,
        "cfg_scale": 7,
        "seed": -1,
        "width": 512,
        "height": 512,
        "sampler_index": "DPM++ 2M",
        "scheduler": "Karras",
        "alwayson_scripts": alwayson_scripts
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        image_data = response.json()["images"][0]
        image_base64 = image_data.split(",", 1)[1] if image_data.startswith("data:image") else image_data
        image_bytes = base64.b64decode(image_base64)

        image_path = os.path.join(OUTPUT_DIR, f"image_{i+1:04d}.png")
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        print(f"[✓] 生成成功：{image_path}")
    else:
        print(f"[X] 生成失败：{response.status_code} - {response.text}")