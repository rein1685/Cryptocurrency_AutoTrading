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

print(torch.cuda.get_device_name(0))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print("device : {}".format(device))

#Hyperparameters
buffer_limit  = 6000
rollout_len   = 10
batch_size    = 32     # Indicates 4 sequences per mini-batch (4*rollout_len = 40 samples total)
c             = 1.0   # For truncating importance sampling ratio
n_episodes    = 1000
print_interval = 10

acer_config = {
    'lr':         0.0005,
    'gamma':      0.99,
    'lmbda':      0.95,
    'c':          1.0, # For truncating importance sampling ratio
}

df = pd.read_csv('./data/1 Dec 2017 - 1 Dec 2018.csv')

test_env = BitcoinTradingEnv(df, serial=True)

print('observation space:', test_env.observation_space.shape)
print('action space:', test_env.action_space)

memory = ReplayBuffer(buffer_limit, batch_size)

model = MGRU_FCN(c_in=test_env.observation_space.shape[0],
                 c_out=test_env.action_space.n,
                 seq_len=test_env.observation_space.shape[1])
model = model.to(device)

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
            seq_data = []
            for t in range(rollout_len):
                prob = acer.model.pi(torch.FloatTensor(obs).unsqueeze(0).to(device))
                action = Categorical(prob).sample().item()
                obs_prime, reward, done, _ = test_env.step(action)

                seq_data.append((obs, action, reward, prob.squeeze().cpu().detach().numpy(), done))

                obs = obs_prime

                avg_r += reward
                avg_t += 1

                if done:
                    break

            memory.put(seq_data)
            if memory.size() > 500:
                acer.train(on_policy=False)
        except Exception as E:
            #print("Wrong Action: {}".format("BUY" if action % 4 <1 else "SELL"))
            pass

    if epi % print_interval == 0:
        print("# of episode : {}, Avg timestep : {}, Avg Reward : {:.4f}%".format(epi, avg_t/print_interval, avg_r/print_interval/avg_t))
        torch.save(acer.model.state_dict(), './save/test')
        avg_t = 0
        avg_r = 0

print('Finished.')