import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from XuNet import XuNet 
from YeNet import YeNet  

# ======= 数据集定义 =======
class StegoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        for label, folder in enumerate(["source", "stego"]):
            folder_path = os.path.join(root_dir, folder)
            for img_name in os.listdir(folder_path):
                self.image_paths.append(os.path.join(folder_path, img_name))
                self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# ======= 数据增强定义 =======
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.RandomResizedCrop((128, 128), scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ======= 加载数据 =======
dataset = StegoDataset("../data", transform=None)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ======= 模型、损失、优化器 =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = XuNet().to(device)
model = YeNet().to(device)
model_path = "best_yenet.pth"

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ======= 训练参数 =======
num_epochs = 30
patience = 5
best_auc = 0
patience_counter = 0

train_losses = []
val_aucs = []

# ======= 训练循环 =======
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 每轮评估 AUC
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1]  # stego概率
            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    current_auc = roc_auc_score(all_labels, all_scores)
    val_aucs.append(current_auc)

    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, AUC = {current_auc:.4f}")

    # 保存最佳模型
    if current_auc > best_auc:
        best_auc = current_auc
        patience_counter = 0
        torch.save(model.state_dict(), model_path)
        print("✅ Saved best model.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break
        continue

# ======= 可视化 Loss 和 AUC 曲线 =======
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig("loss_curve.png")

plt.figure()
plt.plot(val_aucs, label="Validation AUC", color='orange')
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Validation AUC")
plt.legend()
plt.savefig("auc_curve.png")

print("🎉 Training complete. Best AUC:", best_auc)

def test_model(model, test_loader, model_path="best_yenet.pth"):
    print("🔍 Loading best model for final test...")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_labels = []
    all_scores = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    final_auc = roc_auc_score(all_labels, all_scores)
    return final_auc

# 执行测试
test_auc = test_model(model, test_loader)
print(f"🧪 Final Test AUC: {test_auc:.4f}")