from agent import Agent
from TLN_env.env_tln import *
import sys
from tqdm import tqdm
import torch
import math
import random
import matplotlib.pyplot as plt;
import torch.utils.data import Dataset
import torch.utils.data import DataLoader


class TLNDateset(Dataset):
    def __init__(self, X):
        self.data = torch.tensor(X, dtype = torch.float)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

input_file = sys.argv[2]
env = Tln_env(input_file + ".tln")
fi = open(input_file + ".funct2", "r")
lines = fi.readlines()[1:]
lines = [list(map(int, l.strip().split(" "))) for l in lines]
print("Number of functions in training set: ", len(lines))
train_set = TLNDateset(lines)
BATCH_SIZE = 5
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)

input_dim = len(lines[0])*BATCH_SIZE
output_dim = len(env.TLN.edges)
newAgent = Agent(input_dim, output_dim)

newAgent.network.train()
x = []
total_loss = []
NUM_EPOCH = 5
for j in range(NUM_EPOCH):
    train_loss = 0
    for i, data in enumerate(train_loader):
        newAgent.optimizer.zero_grad()
        # output_values = random.choice(lines)
        # output_values = lines[i]
        # output_values = torch.tensor(output_values, dtype = torch.float)
        data.requires_grad = True
        weight = newAgent.sample(data)
        loss = env.step(weight, data)
        newAgent.learn(loss)
        train_loss += loss.item()
    x.append(j + 1)
    total_loss.append(train_loss/len(train_set)*BATCH_SIZE)
    
plt.plot(x, total_loss)
plt.savefig(input_file + ".jpg")
