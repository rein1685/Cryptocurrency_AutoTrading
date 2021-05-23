import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class ActorCritic():
    def __init__(self, Model, optimizer):
        self.model = Model
        self.optimizer = optimizer
        self.memory = []

    def put_data(self, data):
        self.memory.append(data)

    def train(self, gamma):
        R = 0
        self.optimizer.zero_grad()

        for r, log_prob in self.memory[::-1]:
            R = r + R*gamma
            loss = -log_prob*R
            loss.backward()
        self.optimizer.step()
        self.memory = []

    def act(self, x):
        return self.model(x)
        if x.ndim == 3:
            prob = self.model(x)
            action = Categorical(prob).sample()

            action_type = action // 10
            ratio = (action % 10) + 1

            return action_type, ratio
        elif x.ndim == 1:
            prob = self.model(x)
            action = Categorical(prob).sample()

            return action