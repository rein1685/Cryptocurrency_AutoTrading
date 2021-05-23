import pandas as pd
import torch
from env.BitcoinTradingEnv import BitcoinTradingEnv
import numpy as np
import torch.optim as optim
from model.PPO import PPO
from agent.ActorCritic import ActorCritic
import gym
from torch.distributions.categorical import Categorical
import os

df = pd.read_csv('./data/1 Dec 2017 - 1 Dec 2018.csv')

test_env = BitcoinTradingEnv(df, serial=True, max_len=100)
ppo = PPO()

if os.path.exists('./save/test'):
    ppo.load_state_dict(torch.load('./save/test'))

print('observation space:', test_env.observation_space)
print('action space:', test_env.action_space)

gamma = 0.9

n_episodes = 1000

T_horizon     = 20

avg_t = 0
avg_r = 0

for epi in range(1, n_episodes + 1):
    obs = test_env.reset()
    done = False

    while not done:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        for t in range(T_horizon):
            prob = ppo.pi(torch.FloatTensor(obs).unsqueeze(0))
            action = Categorical(prob).sample().item()
            action_type = action // 10
            ratio = (action % 10) + 1
            obs_prime, reward, done, _ = test_env.step((action_type, ratio))

            ppo.put_data((obs, action, reward, obs_prime, prob.squeeze()[action].item(), done))

            obs = obs_prime

            avg_r += reward
            avg_t += 1

            if done:
                break

        ppo.train()

    if epi % 10 == 0 and epi != 0:
        print("# of episode : {}, Avg timestep : {}, Avg Reward : {}".format(epi, avg_t/10, avg_r/10))
        torch.save(ppo.state_dict(), './save/test')
        avg_t = 0
        avg_r = 0

print('Finished.')