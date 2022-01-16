# import gym
# from gym import spaces
# from gym.utils import seeding
import numpy as np
from TLN_env.TLN.TLN import Tln
import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class Tln_env():

    def __init__(self, inputFile):
        # self.range = 1000  # +/- value the randomly select guess_number can be between
        # self.weight_bound = 1  # Action space bounds
        # self.threshold_bound = 1  # Action space bounds
        # self.count = 0
        # self.max_count = 200
        self.TLN = Tln(inputFile)
        # low, high = self.init_weight_and_threshold();
        # self.action_space = spaces.Box(low=np.array(low), high=np.array(high))
        # self.observation_space = spaces.Box(low = np.array(-1.0), high=np.array(1.0))
        # self.prev_reward = 0

        # self.seed()

    # def init_weight_and_threshold(self):
    #     """
    #     TLN-weight: [w1, w2]
    #     TLN-weight: [t1, t2, t3]
    #     weight_bound = 3
    #     threshold_bound = 5
    #     low = [-3, -3, -5]
    #     high = [3, 3, 5]
    #     return lower bound and upper bound of action_space (1-dim array)
    #     """
    #     low = [-self.weight_bound]*len(self.TLN.edges) + [-self.threshold_bound]*(len(self.TLN.nodes) - 2)
    #     high = [self.weight_bound]*len(self.TLN.edges) + [self.threshold_bound]*(len(self.TLN.nodes) - 2)
    #     return low, high

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]
    
    def step(self, action, output_values):
        # assert self.action_space.contains(action)
        # action = list(action)
        self.TLN.set_weights(action[0:len(self.TLN.edges)])
        self.TLN.print_weights()
        # self.TLN.set_thresholds([0]*len(self.TLN.nodes))
        # outputs = torch.empty(0, dtype = torch.float)
        outputs = torch.empty(len(output_values), dtype = torch.float)
        for i in range(int(math.pow(2, len(self.TLN.pis)))):
            input_values = "{0:b}".format(i).zfill(len(self.TLN.pis))
            input_values = torch.tensor(list(map(int, list(input_values))), dtype = torch.float)
            self.TLN.propagate(input_values)
            #CrossEntropy
            # SE.extend(self.TLN.collect_outputs())

            #MSELoss
            outputs[i*len(self.TLN.pos):(i + 1)*len(self.TLN.pos)] = torch.tensor(self.TLN.collect_outputs(), dtype = torch.float)
            # SE.extend(self.TLN.collect_outputs())
            # SE.append(s)

        #CrossEntropy
        # outputs = torch.tensor([SE, output_values], dtype = torch.float)
        # loss = nn.CrossEntropyLoss()
        # outputs.requires_grad = True
        # return loss(outputs, torch.tensor([0, 1], dtype = torch.long))

        #MSELoss
        # outputs = torch.tensor(SE, dtype = torch.float)
        outputs.requires_grad = True
        print("outputs: ", outputs)
        target = torch.tensor(output_values, dtype = torch.float)
        target.requires_grad = True
        print("target: ", target)
        # MSE = torch.from_numpy(MSELoss)
        # MSE.requires_grad = True
        return nn.MSELoss()(outputs, target)


    # def reset(self):
        # self.prev_reward = 0
        # self.observation = 0
        # self.count = 0
        # return self.observation

if __name__ == "__main__":
    if sys.argv[1] == 'read':
        input_file = sys.argv[2]
        model = Tln_env(input_file)
        model.step([1, -0.5, 1], [0])
        print("done")
