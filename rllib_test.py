import gym, ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

import pandas as pd
from env.BitcoinTradingEnv import BitcoinTradingEnv
import shutil

df = pd.read_csv('./data/1 Jan 2017 - 31 Dec 2020.csv')

def env_create(env_config):
    return BitcoinTradingEnv(df, serial=True)

register_env("my_env", env_creator=env_create)

ray.init()

config = ppo.DEFAULT_CONFIG.copy()
config['num_gpus'] = 0
config['num_workers'] = 3

trainer = ppo.PPOTrainer(env="my_env", config=config)
trainer.restore('save/ppo/chpt_train/checkpoint_000102/checkpoint-102')
shutil.rmtree('save/ppo/chpt_train/', ignore_errors=True, onerror=None)

for epi in range(1000):
    result = trainer.train()
    print(result)

    if epi != 0 and epi % 20 == 0:
        trainer.save('save/ppo/chpt_train')