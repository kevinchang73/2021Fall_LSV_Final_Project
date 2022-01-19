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

torch.manual_seed(0)
model_path = "./model.ckpt"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
NUM_EPOCH = 15
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
random.shuffle(lines)
lines = lines[:3000]
print("Number of functions in training set: ", len(lines))
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
f1 = open("./" + input_file + " " + t + " loss", "w")
# f2 = open("./" + input_file + " " + t + " error rate", "w")

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
    newAgent.network.train()
    env.TLN.set_tests(False)
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
        prg_bar.set_description(f"epoch: {epoch} loss:  {batch_loss.item()/BATCH_SIZE: .6f}")
    x.append(epoch + 1)
    total_loss.append(train_loss/len(train_set)*BATCH_SIZE)
    print("Average training loss: ", train_loss/len(train_set)*BATCH_SIZE)
    f1.write(str(train_loss/len(train_set)*BATCH_SIZE) + '\n')

    # Testing
    # newAgent.network.eval()
    # env.TLN.set_tests(True)
    # with torch.no_grad():
    #     test_loss = 0.0
    #     prg_bar = tqdm(enumerate(train_loader))
    #     for i, data in prg_bar:
    #         batch_loss = torch.tensor(0.0, dtype = torch.float).to(device)
    #         for b in range(BATCH_SIZE):
    #             output_values = data[b]
    #             output_values_device = output_values.to(device)
    #             weight = newAgent.sample(output_values_device)
    #             loss = env.step(weight, output_values_device)
    #             batch_loss = torch.add(batch_loss, loss)
    #         test_loss += batch_loss.item()/BATCH_SIZE
    #     f2.write(str(test_loss/len(test_set)*BATCH_SIZE) + '\n')

print("x: ", x)
print("total_loss: ", total_loss)
torch.save(newAgent.network.state_dict(), model_path)
print("Traning Done")
print("Start Testing")
newAgent.network.eval()
env.TLN.set_tests(True)
total_loss = []
with torch.no_grad():
    test_loss = 0.0
    prg_bar = tqdm(enumerate(test_loader))
    for i, data in prg_bar:
        batch_loss = torch.tensor(0.0, dtype = torch.float).to(device)
        for b in range(BATCH_SIZE):
            output_values = data[b]
            output_values_device = output_values.to(device)
            weight = newAgent.sample(output_values_device)
            loss = env.step(weight, output_values_device)
            batch_loss = torch.add(batch_loss, loss)
            # print("###############")
            # for name, params in newAgent.network.named_parameters():
            #     print("params: ", params)
            #     print("params grad: ", params.grad)

        test_loss += batch_loss.item()/BATCH_SIZE
        prg_bar.set_description(f"error rate:  {batch_loss.item()/BATCH_SIZE: .6f}")
    print("Testing error rate: ", test_loss/len(test_set)*BATCH_SIZE)
    f1.write(str(test_loss/len(test_set)*BATCH_SIZE) + '\n')
print("Testing Done")

f1.close()
