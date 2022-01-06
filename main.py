from agent import Agent
from TLN_env.env_tln import *
import sys
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import math
import random

input_file = sys.argv[2]
env = Tln_env(input_file + ".tln")
fi = open(input_file + ".funct", "r")
lines = fi.readlines()[1:]
lines = [list(map(int, l.strip().split(" "))) for l in lines]

input_dim = len(lines[0])
output_dim = len(env.TLN.nodes) + len(env.TLN.edges) - len(env.TLN.pis)
# print("output_dim: ", output_dim)
newAgent = Agent(input_dim, output_dim)

newAgent.network.train()

EPISODE_PER_BATCH = 5  # 每蒐集 5 個 episodes 更新一次 agent
NUM_BATCH = 400        # 總共更新 400 次

avg_total_rewards, avg_final_rewards = [], []

prg_bar = tqdm(range(NUM_BATCH))
for batch in prg_bar:

    rewards = []
    total_rewards, final_rewards = [], []

    # 蒐集訓練資料
    for episode in range(EPISODE_PER_BATCH):
        
        state = env.reset()
        total_reward, total_step = 0, 0

        while True:
            input_values = random.choice(lines)
            action = newAgent.sample(input_values)
            reward, done = env.step(action, input_values)
            total_reward += reward
            total_step += 1

            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                rewards.append(total_reward)  # 設定同一個 episode 每個 action 的 reward 都是 total reward
                break

    # 紀錄訓練過程
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    prg_bar.set_description(f"Total: {avg_total_reward: 0.6f}, Final: {avg_final_reward: 0.6f}")

    # 更新網路
    print(rewards)
    # rewards = np.concatenate(rewards, axis=0)
    # rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
    newAgent.learn(Variable(torch.from_numpy(rewards), requires_grad = True))
