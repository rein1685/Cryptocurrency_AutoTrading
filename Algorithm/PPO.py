import pandas as pd
import torch
from env.BitcoinTradingEnv import BitcoinTradingEnv
from agent.PPO import PPO
from model.GRUFCN.models.RNN_FCN import MGRU_FCN as MODEL
from utils.ReplayMemory import Memory
from torch.distributions.categorical import Categorical
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print("device : {}".format(device))

#Hyperparameters
T_horizon   = 30
n_episodes    = 10000
print_interval = 2

config = {
    'lr':           0.0005,
    'gamma':        0.90,
    'lmbda':        0.95,
    'eps_clip':     0.1,
    'K_epoch':      3,
}

df = pd.read_csv('./data/1 Dec 2017 - 1 Dec 2018.csv')

test_env = BitcoinTradingEnv(df, serial=True)

print('observation space:', test_env.observation_space.shape)
print('action space:', test_env.action_space)

memory = Memory()

model = MODEL(c_in=test_env.observation_space.shape[0],
                   c_out=test_env.action_space.n,
                   seq_len=test_env.observation_space.shape[1])
model = model.to(device)

agent = PPO(model=model, memory=memory, config=config, device=device)

if os.path.exists('./save/model.m5'):
    agent.model.load_state_dict(torch.load('./save/model.m5'))

avg_t = 0
avg_r = 0

for epi in range(1, n_episodes + 1):
    print("episode {} start!".format(epi))
    obs = test_env.reset()
    done = False

    while not done:
        t = 0
        action_list = []

        while t < T_horizon:
            prob = agent.model.pi(torch.FloatTensor(obs).unsqueeze(0).to(device))
            action = Categorical(prob).sample().item()
            obs_prime, reward, done, _ = test_env.step(action)

            if reward is None:
                continue

            action_list.append(action)

            memory.put((obs, action, reward, obs_prime, prob.squeeze()[action].detach().item(), done))

            obs = obs_prime

            avg_r += reward
            avg_t += 1

            t += 1

            if done:
                break

        print(action_list)
        agent.train()

    if epi % print_interval == 0:
        print("# of episode : {}, Avg timestep : {}, Avg Reward : {:.4f}%".format(epi, avg_t/print_interval, avg_r/print_interval/avg_t))
        torch.save(agent.model.state_dict(), './save/model.m5')
        avg_t = 0
        avg_r = 0

print('Finished.')