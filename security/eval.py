import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from XuNet import XuNet  # 你需要准备好 XuNet 类定义'
from YeNet import YeNet  # 你需要准备好 YeNet 类定义

class StegoOnlyDataset(Dataset):
    def __init__(self, stego_dir):
        self.paths = [os.path.join(stego_dir, f) for f in os.listdir(stego_dir) if f.endswith(('.jpg', '.png'))]
        self.label = 1  # 所有图片都是 stego
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.label
    
def evaluate_stego_accuracy(model_path, stego_dir):
    model = YeNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    dataset = StegoOnlyDataset(stego_dir)
    loader = DataLoader(dataset, batch_size=32)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)  # 取最大概率类别
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ YeNet Accuracy on stego-only dataset: {acc:.4f}")

# 使用示例
evaluate_stego_accuracy("best_yenet.pth", "../data/sd")