import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Network(nn.Module):
    def __init__(self): #Todo: convert TLN to NN
        super(Network, self).__init__()

    def forward(self, x):

    def cal_loss(self, pred, target):
        