import pandas as pd
import torch
from env.BitcoinTradingEnv import BitcoinTradingEnv
import numpy as np
import torch.optim as optim
from agent.ACER import ACER
from model.models.RNN_FCN import MGRU_FCN
from agent.ActorCritic import ActorCritic
from utils.ReplayMemory import ReplayBuffer
import gym
from torch.distributions.categorical import Categorical
import os

#Hyperparameters
buffer_limit  = 6000
rollout_len   = 10
batch_size    = 4     # Indicates 4 sequences per mini-batch (4*rollout_len = 40 samples total)
c             = 1.0   # For truncating importance sampling ratio
n_episodes    = 1000
print_interval = 10

acer_config = {
    'lr':         0.0005,
    'gamma':      0.99,
    'lmbda':      0.95,
    'c':          1.0, # For truncating importance sampling ratio
}

df = pd.read_csv('./data/1 Dec 2019 - 1 Dec 2020.csv')

test_env = BitcoinTradingEnv(df, serial=True)

print('observation space:', test_env.observation_space.shape)
print('action space:', test_env.action_space)

memory = ReplayBuffer(buffer_limit, batch_size)

model = MGRU_FCN(c_in=test_env.observation_space.shape[0],
                 c_out=test_env.action_space.n,
                 seq_len=test_env.observation_space.shape[1])

acer = ACER(model=model, memory=memory, config=acer_config)

if os.path.exists('./save/test'):
    acer.model.load_state_dict(torch.load('./save/test'))

avg_t = 0
avg_r = 0

for epi in range(1, n_episodes + 1):
    obs = test_env.reset()
    done = False

    while not done:
        try:
            prob = acer.model.pi(torch.FloatTensor(obs).unsqueeze(0))
            action = Categorical(prob).sample().item()
            obs_prime, reward, done, _ = test_env.step(action)

            test_env.render(mode='file')

            obs = obs_prime

            if done:
                break
        except:
            pass

print('Finished.')