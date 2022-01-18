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
import time

model_path = "./model.ckpt"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
NUM_EPOCH = 10
BATCH_SIZE = 50
TRAINING_DATA_RATIO = 0.8
class TLNDateset(Dataset):
    def __init__(self, X):
        self.data = torch.tensor(X, dtype = torch.float)
        self.data.requires_grad = True

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
random.shuffle(lines)
lines = lines[:3000]
train_lines = lines[:int(len(lines)*TRAINING_DATA_RATIO)]
test_lines = lines[int(len(lines)*TRAINING_DATA_RATIO):]
train_set = TLNDateset(train_lines)
test_set = TLNDateset(test_lines)
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = True)

input_dim = len(lines[0])
output_dim = len(env.TLN.edges)
newAgent = Agent(input_dim, output_dim)

t = time.asctime(time.localtime(time.time()))
f = open("./" + input_file + " " + t + " train", "w")

newAgent.network.train()
x = []
total_loss = []
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
    train_loss = 0.0
    prg_bar = tqdm(enumerate(train_loader))
    for i, data in prg_bar:
        batch_loss = torch.tensor(0.0, dtype = torch.float).to(device)
        newAgent.optimizer.zero_grad()
        for b in range(BATCH_SIZE):
            # output_values = random.choice(lines)
            # output_values = lines[i]
            # output_values = torch.tensor(output_values, dtype = torch.float)
            output_values = data[b]
            # output_values.requires_grad = True
            output_values_device = output_values.to(device)
            weight = newAgent.sample(output_values_device)
            # print(weight)
            # print(weight)
            loss = env.step(weight, output_values_device)
            # print(loss.item())
            batch_loss = torch.add(batch_loss, loss)
            # print(train_loss)

            # print("###############")
            # for name, params in newAgent.network.named_parameters():
            #     print("params: ", params)
            #     print("params grad: ", params.grad)


        newAgent.learn(batch_loss)
        train_loss += batch_loss.item()/BATCH_SIZE
        prg_bar.set_description(f"loss:  {batch_loss.item()/BATCH_SIZE: .6f}")
    x.append(epoch + 1)
    total_loss.append(train_loss/len(train_set))
    print("Average training loss: ", train_loss/len(train_set))
    f.write(str(train_loss/len(train_set)) + '\n')

print("x: ", x)
print("total_loss: ", total_loss)
torch.save(newAgent.network.state_dict(), model_path)
print("Traning Done")
f.close()
f = open("./" + input_file + " " + t + " test", "w")
print("Start Testing")
newAgent.network.eval()
env.TLN.set_tests(True)
total_loss = []
with torch.no_grad():
    for epoch in range(NUM_EPOCH):
        test_loss = 0.0
        prg_bar = tqdm(enumerate(test_loader))
        for i, data in prg_bar:
            batch_loss = torch.tensor(0.0, dtype = torch.float).to(device)
            for b in range(BATCH_SIZE):
                # output_values = random.choice(lines)
                # output_values = lines[i]
                # output_values = torch.tensor(output_values, dtype = torch.float)
                output_values = data[b]
                # output_values.requires_grad = True
                output_values_device = output_values.to(device)
                weight = newAgent.sample(output_values_device)
                # print(weight)
                # print(weight)
                loss = env.step(weight, output_values_device)
                # print(loss.item())
                batch_loss = torch.add(batch_loss, loss)
                # print(train_loss)

                # print("###############")
                # for name, params in newAgent.network.named_parameters():
                #     print("params: ", params)
                #     print("params grad: ", params.grad)

            test_loss += batch_loss.item()/BATCH_SIZE
            prg_bar.set_description(f"loss:  {batch_loss.item()/BATCH_SIZE: .6f}")
        x.append(epoch + 1)
        total_loss.append(test_loss/len(test_set))
        print("Average testing loss: ", test_loss/len(test_set))
        f.write(str(test_loss/len(test_set)) + '\n')
print("Testing Done")


f.close()
