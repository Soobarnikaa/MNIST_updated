import torch
import torch.nn as nn
from Conv2d import CustomConv2D  # Assuming this is defined in another file

class MNISTModel(nn.Module):
    def __init__(self, custom_conv=False):
        super(MNISTModel, self).__init__()
        if custom_conv:
            # Use custom conv layer for inference
            self.conv1 = CustomConv2D(1, 32, kernel_size=3, padding=1)
            self.conv2 = CustomConv2D(32, 64, kernel_size=3, padding=1)
        else:
            # Use PyTorch's Conv2d for training
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
