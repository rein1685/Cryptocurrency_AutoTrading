import torch
import torch.nn as nn
import torch.nn.functional as F

class testModel(nn.Module):
    def __init__(self):
        super(testModel, self).__init__()
        self.conv1d_1 = nn.Conv1d(10, 64, kernel_size=1)
        self.conv1d_2 = nn.Conv1d(64, 32, kernel_size=1)
        self.fc = nn.Linear(32*128, 128)
        self.out = nn.Linear(128, 21)

    def forward(self, x):
        x = self.conv1d_1(x)
        x = F.relu(self.conv1d_2(x))

        x = x.view(-1, 32*128)

        x = self.fc(x)
        out = F.softmax(self.out(x), -1)
        return out