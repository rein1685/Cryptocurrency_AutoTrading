import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gym

class Conv1d_Net(nn.Module):
    def __init__(self, c_in=13, seq_len=128, c_out=9):
        super(Conv1d_Net, self).__init__()
        self.memory = []

        self.conv1d_1 = nn.Conv1d(c_in, 64, kernel_size=1)
        self.conv1d_2 = nn.Conv1d(64, 32, kernel_size=1)
        self.fc = nn.Linear(32 * seq_len, 128)

        self.fc_pi = nn.Linear(128, c_out)
        self.fc_v = nn.Linear(128, c_out)

    def pi(self, x):
        x = self.conv1d_1(x)
        x = F.relu(self.conv1d_2(x))

        x = x.view(-1, 32 * 128)

        x = F.relu(self.fc(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob

    def v(self, x):
        x = self.conv1d_1(x)
        x = F.relu(self.conv1d_2(x))

        x = x.view(-1, 32 * 128)

        x = F.relu(self.fc(x))
        v = self.fc_v(x)
        return v