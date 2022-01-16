import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Network(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, state):
        hid = self.fc1(state)
        hid = self.fc2(hid)
        return self.fc3(hid)


class Agent():
    
    def __init__(self, input_dim, output_dim):
        self.network = Network(input_dim, output_dim)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.network = self.network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)

    def learn(self, loss):
        # loss = torch.mean(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, output_values):
        # input_values = torch.FloatTensor(input_values)
        output_values = output_values.to(self.device)
        weight = self.network(output_values)
        # print(action)
        return weight

