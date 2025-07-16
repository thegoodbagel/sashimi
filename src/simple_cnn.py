import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSushiCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # input RGB image
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # assuming input images are 224x224
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 + relu + maxpool
        x = self.pool(F.relu(self.conv2(x)))  # conv2 + relu + maxpool
        x = torch.flatten(x, 1)                # flatten all except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                        # output logits
        return x
