import os
from PIL import Image
import torch
import torchvision.transforms as T
from Aesthtics import AestheticScorer  # 你自己的 NIMA 实现
import pandas as pd
import pyiqa  # ✅ 用于 NIQE、BRISQUE 等质量评估

# ==== 1. 参数设置 ====
image_folder = "../data/sd"  # 👈 修改为你的图像目录
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 2. 初始化模型 ====
scorer = AestheticScorer()
niqe_model = pyiqa.create_metric('niqe').to(device)  # ✅ 初始化 NIQE 模型

# ==== 3. 图像预处理 ====
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

results = []

# ==== 4. 遍历图像 ====
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            path = os.path.join(image_folder, filename)
            img_pil = Image.open(path).convert("RGB")
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            # 4.1 NIQE 评分（无参考）
            with torch.no_grad():
                niqe_score = niqe_model(img_tensor).item()

            # 4.2 NIMA 美学评分
            nima_score = scorer.score_image(img_pil)

            results.append({
                'image': filename,
                'NIQE': round(niqe_score, 3),
                'NIMA': round(nima_score, 3),
            })

        except Exception as e:
            print(f"⚠️ Failed to process {filename}: {e}")

# ==== 5. 保存或打印结果 ====
df = pd.DataFrame(results)
print(df)

# df.to_csv("image_quality_results.csv", index=False)
# print("✅ 图像质量评分已保存为 image_quality_results.csv")