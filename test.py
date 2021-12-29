import gym

import TLN_env.env

env = gym.make('TLNENV-v0')

obs = env.reset()

for step in range(10000):
    action = env.action_space.sample()
    print(action)
    obs, reward, done, info = env.step(action)
