import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from TLN import TLN


class tln(gym.Env):

    def __init__(self):
        self.range = 1000  # +/- value the randomly select guess_number can be between
        self.weight_bounds = 3  # Action space bounds
        self.TLN = TLN(topology)
        low, high = self.init_weight_and_threshold(topology);
        self.action_space = spaces.Box(low=np.array(low), high=np.array(high)))
        self.observation_space = spaces.Discrete(4)

        self.guess_number = 0
        self.guess_count = 0
        self.guess_max = 200
        self.observation = 0

        self.seed()
        self.reset()

    def init_weight_and_threshold(self, topology):
        #TLN-weight: [[[w11, w12], [w13, w14]], [[w21, w22]]]
        #TLN-weight: [[t11, t12], [t2]]
        #weight_bound = 3
        #threshold_bound = 5
        #low = [-3, -3, -5, -3, -3, -5, -3, -3, -5]
        #high = [3, 3, 5, 3, 3, 5, 3, 3, 5]
        #return lower bound and upper bound of action_space (1-dim array)
    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def step(self, action):
        # assert self.action_space.contains(action)

        if action < self.guess_number:
            self.observation = 1

        elif action == self.guess_number:
            self.observation = 2

        elif action > self.guess_number:
            self.observation = 3

        reward = ((min(action, self.guess_number) + self.bounds) / (max(action, self.guess_number) + self.bounds)) ** 2

        self.guess_count += 1
        done = self.guess_count >= self.guess_max

        return self.observation, reward, done, {"guess_number": self.guess_number, "guesses": self.guess_count}

    def reset(self):
        self.guess_number = self.np_random.uniform(-self.range, self.range)
        print('guess number = ', self.guess_number)
        self.guess_count = 0
        self.observation = 0
        return self.observation
