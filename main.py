from agent import Agent
from TLN_env.env_tln import *
import sys
from tqdm import tqdm
import torch
import math
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

NUM_EPOCH = 5
BATCH_SIZE = 5
class TLNDateset(Dataset):
    def __init__(self, X):
        self.data = torch.tensor(X, dtype = torch.float)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

input_file = sys.argv[2]
env = Tln_env(input_file + ".tln", BATCH_SIZE)
fi = open(input_file + ".funct2", "r")
lines = fi.readlines()[1:]
lines = [list(map(int, l.strip().split(" "))) for l in lines]
print("Number of functions in training set: ", len(lines))
train_set = TLNDateset(lines)
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)

input_dim = len(lines[0])*BATCH_SIZE
output_dim = len(env.TLN.edges)
newAgent = Agent(input_dim, output_dim)

newAgent.network.train()
x = []
total_loss = []
prg_bar = tqdm(enumerate(train_loader))

#CONFIRM MODEL RUNNING 
# for i in range(1000):
#     newAgent.optimizer.zero_grad()
#     output_values = torch.tensor(lines[0], dtype = torch.float)
#     output_values.requires_grad = True
#     weight = newAgent.sample(output_values)
#     loss = env.step(weight, output_values)
#     print(loss)
#     newAgent.learn(loss)

# Start Training
for epoch in range(NUM_EPOCH):
    train_loss = 0
    for i, data in prg_bar:
        newAgent.optimizer.zero_grad()
        # output_values = random.choice(lines)
        # output_values = lines[i]
        # output_values = torch.tensor(output_values, dtype = torch.float)
        output_values = data[0]
        for k in range(1, BATCH_SIZE):
            output_values = torch.cat((output_values, data[k]), 0)
        output_values.requires_grad = True
        
        weight = newAgent.sample(output_values)
        # print(weight)
        loss = env.step(weight, output_values)
        print(loss.item())
        newAgent.learn(loss)
        train_loss += loss.item()
        # print(train_loss)

        # print("###############")
        # for name, params in newAgent.network.named_parameters():
        #     print("params: ", params)
        #     print("params grad: ", params.grad)


        prg_bar.set_description(f"loss:  {loss.item(): .6f}")
    x.append(epoch + 1)
    total_loss.append(train_loss/len(train_set)*BATCH_SIZE)
    
plt.plot(x, total_loss)
plt.savefig(input_file + ".jpg")
