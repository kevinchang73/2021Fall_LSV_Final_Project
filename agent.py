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
        hid = self.fc1(state)
        hid = self.fc2(hid)
        return torch.tanh(self.fc3(hid))


class Agent():
    
    def __init__(self, input_dim, output_dim):
        self.network = Network(input_dim, output_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

    def learn(self, CrossEntropy):

        self.optimizer.zero_grad()
        CrossEntropy.backward()
        self.optimizer.step()

    def sample(self, input_values):
        action = self.network(torch.FloatTensor(input_values))
        print(action)
        return action

