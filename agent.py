import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Network(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim = 1)


class Agent():
    
    def __init__(self, input_dim, output_dim):
        self.network = Network(input_dim, output_dim)
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)

    def learn(self, rewards):
        loss = (-rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, input_values):
        action = self.network(input_values)

        return action

