import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import argparse
import os
from XuNet import XuNet  # 你需要准备好 XuNet 类定义

def generate_gradcam(model, image_path, save_path="gradcam_result.png"):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # 读取图像
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).requires_grad_()

    # 注册钩子获取梯度和激活
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Hook 在 conv3 上
    handle_fw = model.conv3.register_forward_hook(forward_hook)
    handle_bw = model.conv3.register_full_backward_hook(backward_hook)

    # 前向 + 反向
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    loss = output[0, pred_class]
    loss.backward()

    # 计算 Grad-CAM 权重和热图
    grads = gradients[0][0].detach().numpy()
    acts = activations[0][0].detach().numpy()
    weights = np.mean(grads, axis=(1, 2))
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() > 0 else cam
    cam = cv2.resize(cam, (128, 128))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # 原图叠加可视化
    original = np.array(image.resize((128, 128)))
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    # 保存图像
    cv2.imwrite(save_path, overlay[:, :, ::-1])  # RGB → BGR
    print(f"✅ Grad-CAM saved to: {save_path}")

    handle_fw.remove()
    handle_bw.remove()

# ===== 3. CLI 命令行接口 =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to xunet_weights.pth")
    parser.add_argument("--image", type=str, required=True, help="Path to stego image")
    parser.add_argument("--output", type=str, default="gradcam_result.png", help="Save path for overlay image")
    args = parser.parse_args()

    # 加载模型
    model = XuNet()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    # 生成 Grad-CAM
    generate_gradcam(model, args.image, args.output)