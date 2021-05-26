import pandas as pd
import torch
from env.BitcoinTradingEnv import BitcoinTradingEnv
import numpy as np
import torch.optim as optim
from agent.ACER import ACER
from model.GRUFCN.models.RNN_FCN import MGRU_FCN as MODEL
from agent.ActorCritic import ActorCritic
from utils.ReplayMemory import ReplayBuffer
import gym
from torch.distributions.categorical import Categorical
import os

df = pd.read_csv('./data/1 Dec 2019 - 1 Dec 2020.csv')

test_env = BitcoinTradingEnv(df, serial=True)

print('observation space:', test_env.observation_space.shape)
print('action space:', test_env.action_space)

model = MODEL(c_in=test_env.observation_space.shape[0],
            c_out=test_env.action_space.n,
            seq_len=test_env.observation_space.shape[1])

#acer = ACER(model=model, memory=memory, config=acer_config)

if os.path.exists('./save/model.m5'):
    model.load_state_dict(torch.load('./save/model.m5'))

avg_t = 0
avg_r = 0

obs = test_env.reset()
done = False

while True:
    prob = model.pi(torch.FloatTensor(obs).unsqueeze(0))
    action = Categorical(prob).sample().item()
    obs_prime, reward, done, _ = test_env.step(action)

    if reward is None:
        continue

    print("current price : {}".format(test_env._get_current_price()))
    print("action : {}".format(action))
    print("prob : {}".format(prob))

    test_env.render(mode='human')

    obs = obs_prime

print('Finished.')