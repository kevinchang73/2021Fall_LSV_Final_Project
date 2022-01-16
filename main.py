# from importlib.metadata import requires
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
NUM_BATCH = 40000        # 總共更新 400 次

avg_total_rewards, avg_final_rewards = [], []

prg_bar = tqdm(range(NUM_BATCH))
x = [i for i in range(int(NUM_BATCH/10))]
y = []
i = 0;
for batch in prg_bar:


    output_values = random.choice(lines)
    # output_values = lines[1000]
    output_values = torch.tensor(output_values, dtype = torch.float)
    action = newAgent.sample(output_values)
    # print(action)
    # print("action: ", action)
    loss = env.step(action, output_values)
    print(loss)
    i += 1
    if(i%10 == 0):
        y.append(loss)
    newAgent.learn(Variable(loss, requires_grad = True))
    # newAgent.learn(loss)

plt(x, y)
