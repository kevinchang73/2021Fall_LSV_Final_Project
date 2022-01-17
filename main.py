from agent import Agent
from TLN_env.env_tln import *
import sys
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import math
import random
import matplotlib.pyplot as plt;

input_file = sys.argv[2]
env = Tln_env(input_file + ".tln")
fi = open(input_file + ".funct", "r")
lines = fi.readlines()[1:]
lines = [list(map(int, l.strip().split(" "))) for l in lines]
print("Number of functions in training set: ", len(lines))

input_dim = len(lines[0])
output_dim = len(env.TLN.edges)
# print("output_dim: ", output_dim)
newAgent = Agent(input_dim, output_dim)

newAgent.network.train()

# EPISODE_PER_BATCH = 5  # 每蒐集 5 個 episodes 更新一次 agent
NUM_BATCH = 400        # 總共更新 400 次


prg_bar = tqdm(range(NUM_BATCH))
x = [i for i in range(int(NUM_BATCH/10))]
y = []
i = 0;
for batch in prg_bar:
    newAgent.optimizer.zero_grad()
    # output_values = random.choice(lines)
    output_values = lines[0]
    output_values = torch.tensor(output_values, dtype = torch.float)
    output_values.requires_grad = True
    weight = newAgent.sample(output_values)
    # print("Weight: ", weight)
    # weight_sum = weight.sum()
    # weight_sum.retain_grad()
    # weight_sum.backward()
    # newAgent.optimizer.step()
    # print("##################")
    # for name, params in newAgent.network.named_parameters():
    #     # params.retain_grad()
    #     # print("name: ", name)
    #     print("para: ", params)
    #     # print("required_grad: ", params.requires_grad)
    #     print("grad: ", params.grad)
    # print(weight_sum.grad)
    # print(output_values.grad)
    loss = env.step(weight, output_values)
    # prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")
    # print(weight)
    print("loss: ", loss)
    i += 1
    if(i%10 == 0):
        y.append(loss)
    # for name, params in newAgent.network.named_parameters():
    #     params.retain_grad()
    #     # print("name: ", name)
    #     # print("para: ", params)
    newAgent.learn(loss)


plt.plot(x, y)
plt.savefig("Case2.jpg")
