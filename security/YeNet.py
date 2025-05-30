import torch
import torch.nn as nn
import torch.nn.functional as F

class YeNet(nn.Module):
    def __init__(self):
        super(YeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2, bias=False)
        self.pool = nn.AvgPool2d(5, stride=2, padding=2)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 适用于输入128x128
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x