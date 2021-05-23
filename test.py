import pandas as pd
import torch
from env.BitcoinTradingEnv import BitcoinTradingEnv
import numpy as np
import torch.optim as optim
from model.PPO import PPO
from agent.ActorCritic import ActorCritic
import gym
from torch.distributions.categorical import Categorical

df = pd.read_csv('./data/1 Dec 2017 - 1 Dec 2018.csv')

test_env = BitcoinTradingEnv(df, serial=True, commission=0.0)
ppo = PPO()
ppo.load_state_dict(torch.load('./save/test'))

for epi in range(1, 300 + 1):
    obs = test_env.reset()

    while True:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        with torch.no_grad():
            prob = ppo.pi(torch.FloatTensor(obs).unsqueeze(0))
            action = Categorical(prob).sample().item()
            obs, reward, done, _ = test_env.step(action)

            test_env.render(mode='human')

torch.save(ppo.state_dict(), './save/test')
print('Finished.')