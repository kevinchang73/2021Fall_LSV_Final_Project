import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from TLN import Tln
import sys
import math


class Tln_env(gym.Env):

    def __init__(self, inputFile):
        self.range = 1000  # +/- value the randomly select guess_number can be between
        self.weight_bound = 3  # Action space bounds
        self.threshold_bound = 5  # Action space bounds
        self.count = 0
        self.max_count = 200
        self.TLN = Tln(inputFile)
        low, high = self.init_weight_and_threshold();
        self.action_space = spaces.Box(low=np.array(low), high=np.array(high))
        self.observation_space = spaces.Box(low = np.array(-1.0), high=np.array(1.0))
        # self.prev_reward = 0

        # self.seed()

    def init_weight_and_threshold(self):
        """
        TLN-weight: [w1, w2]
        TLN-weight: [t1, t2, t3]
        weight_bound = 3
        threshold_bound = 5
        low = [-3, -3, -5]
        high = [3, 3, 5]
        return lower bound and upper bound of action_space (1-dim array)
        """
        low = [-self.weight_bound]*len(self.TLN.edges) + [-self.threshold_bound]*(len(self.TLN.nodes) - 2)
        high = [self.weight_bound]*len(self.TLN.edges) + [self.threshold_bound]*(len(self.TLN.nodes) - 2)
        return low, high

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def step(self, action, output_values):
        # assert self.action_space.contains(action)

        self.TLN.set_weights(action[0:len(self.TLN.edges)])
        self.TLN.set_thresholds([0, 0] + action[len(self.TLN.edges):])
        reward = 0.0
        for i in range(int(math.pow(2, len(self.TLN.pis)))):
            input_values = "{0:b}".format(i).zfill(len(self.TLN.pis))
            self.TLN.propagate(list(map(int, list(input_values))))
            reward += 1.0 - self.TLN.evaluate(output_values)/len(output_values)

        reward /= int(math.pow(2, len(self.TLN.pis)))
        # self.observation = reward - self.prev_reward if self.prev_reward else 0.0
        # self.prev_reward = reward
        self.count += 1
        # print(reward)
        done = self.count >= self.max_count

        return reward, done
    def reset(self):
        # self.prev_reward = 0
        # self.observation = 0
        self.count = 0
        # return self.observation

if __name__ == "__main__":
    if sys.argv[1] == 'read':
        input_file = sys.argv[2]
        model = Tln_env(input_file)
        model.step([1, -0.5, 1], [0])
