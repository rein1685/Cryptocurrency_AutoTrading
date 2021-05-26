import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn import preprocessing
from stockstats import StockDataFrame
import talib
from talib.abstract import *
import random

from render.BitcoinTradingGraph import BitcoinTradingGraph

MAX_TRADING_SESSION = 100000


class BitcoinTradingEnv(gym.Env):
    """A Bitcoin trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human', 'system', 'file', 'none']}
    scaler = preprocessing.MinMaxScaler()
    viewer = None

    def __init__(self, df, lookback_window_size=127, initial_balance=10000, commission=0.00075, serial=False):
        super(BitcoinTradingEnv, self).__init__()

        self.df = df.dropna().reset_index()
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.serial = serial

        self.file_history = ""

        #Add column from OHLCV using TA-Lib Library
        scaled_df = self.df[['Open', 'High', 'Low', 'Close', 'Volume']].astype('float64')
        scaled_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'},
                         inplace=True)

        stock = StockDataFrame.retype(scaled_df)
        indicators_df = stock[['macd', 'vr']]
        indicators_df = pd.concat([indicators_df, CCI(scaled_df), OBV(scaled_df), RSI(scaled_df), STOCHRSI(scaled_df)], axis=1)
        indicators_df.rename(columns={0: 'cci', 1: 'obv', 2: 'rsi'}, inplace=True)

        self.df = pd.concat([self.df, indicators_df], axis=1)
        self.df = self.df.dropna()

        # Actions of the format Buy 1/10, Sell 3/10, Hold (amount ignored), etc.
        self.action_space = spaces.Discrete(9)

        # Observes the OHCLV values, net worth, and trade history
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(13, lookback_window_size + 1), dtype=np.float16)

    def _next_observation(self):
        end = self.current_step + self.lookback_window_size + 1

        scaled_df = self.active_df.values[:end].astype('float64')
        scaled_df = pd.DataFrame(scaled_df, columns=self.df.columns)

        obs = np.array([
            scaled_df['Close'].values[self.current_step:end],
            scaled_df['macd'].values[self.current_step:end],
            scaled_df['vr'].values[self.current_step:end],
            scaled_df['cci'].values[self.current_step:end],
            scaled_df['obv'].values[self.current_step:end],
            scaled_df['rsi'].values[self.current_step:end],
            scaled_df['fastk'].values[self.current_step:end],
            scaled_df['fastd'].values[self.current_step:end],
        ])

        obs = np.append(
            obs, self.account_history[:, -(self.lookback_window_size + 1):], axis=0)

        obs = self.scaler.fit_transform(obs)

        return obs

    def _reset_session(self):
        #self.current_step = int(len(self.df)*random.random())
        #if len(self.df) - self.current_step <= 500:
        #    self.current_step -= 500
        self.current_step = 0

        if self.serial:
            self.steps_left = len(self.df) - self.lookback_window_size - 1
            self.frame_start = self.lookback_window_size
        else:
            self.steps_left = np.random.randint(1, MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(
                self.lookback_window_size, len(self.df) - self.steps_left)

        self.active_df = self.df[self.frame_start - self.lookback_window_size:
                                 self.frame_start + self.steps_left]

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.btc_held = 0

        self._reset_session()

        self.account_history = np.repeat([
            [self.balance],
            [0],
            [0],
            [0],
            [0]
        ], self.lookback_window_size + 1, axis=1)
        self.trades = []

        return self._next_observation()

    def _get_current_price(self):
        return self.df['Close'].values[self.frame_start + self.current_step]

    def _take_action(self, action, current_price):
        #return_value = 0

        action_type = action // 4
        amount = ((action % 4) + 1)/ 4

        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0

        if (action_type == 0 and self.balance < self.initial_balance * 0.01) or \
            (action_type == 1 and self.btc_held <= 0) or action_type == 2:
            return -1

        if action_type < 1: #BUY
            btc_bought = (self.balance * amount)/ current_price
            cost = btc_bought * current_price * (1 + self.commission)

            self.btc_held += btc_bought
            self.balance -= cost

        elif action_type < 2: #SELL
            btc_sold = self.btc_held * amount
            sales = btc_sold * current_price * (1 - self.commission)

            self.btc_held -= btc_sold
            self.balance += sales

        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({'step': self.frame_start + self.current_step,
                                'amount': btc_sold if btc_sold > 0 else btc_bought, 'total': sales if btc_sold > 0 else cost,
                                'type': "sell" if btc_sold > 0 else "buy"})

        self.net_worth = self.balance + self.btc_held * current_price

        self.account_history = np.append(self.account_history, [
            [self.balance],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]
        ], axis=1)

        return 0

    def step(self, action):
        current_price = self._get_current_price()

        prev_net_worth = self.net_worth

        if (self._take_action(action, current_price) < 0):
            return None, None, True, {}

        self.steps_left -= 1
        self.current_step += 1

        if self.steps_left == 0:
            self.balance += self.btc_held * current_price
            self.btc_held = 0

            self._reset_session()

        obs = self._next_observation()
        reward = ((self.net_worth - prev_net_worth) / prev_net_worth) * 100
        done = self.current_step + self.lookback_window_size + 3 > len(self.df) or self.net_worth <= self.initial_balance*0.8 or \
               self.net_worth >= self.initial_balance * 1.2 or self.current_step > 200


        if action // 4 == 0:
            amount = ((action % 4) + 1)/ 4
            self.file_history = "{}%({:.1f}) BUY ({:.1f} -> {:.1f})".format(amount*100, self.trades[-1]['total'], prev_net_worth, self.net_worth)
        elif action // 4 == 1:
            amount = ((action % 4) + 1) / 4
            self.file_history = "{}%({:.1f}) SELL ({:.1f} -> {:.1f})".format(amount*100, self.trades[-1]['total'], prev_net_worth, self.net_worth)
        else:
            self.file_history = "HOLD ({} -> {})".format(prev_net_worth, self.net_worth)

        return obs, reward, done, {}

    def render(self, mode='human', **kwargs):
        if mode == 'system':
            print('Price: ' + str(self._get_current_price()))
            print(
                'Bought: ' + str(self.account_history[2][self.current_step + self.frame_start]))
            print(
                'Sold: ' + str(self.account_history[4][self.current_step + self.frame_start]))
            print('Net worth: ' + str(self.net_worth))

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = BitcoinTradingGraph(
                    self.df, kwargs.get('title', None))

            self.viewer.render(self.frame_start + self.current_step,
                               self.net_worth,
                               self.trades,
                               window_size=self.lookback_window_size)

        elif mode == 'file':
            with open('log/hisotry.txt', 'a') as f:
                f.write("{}\n".format(self.file_history))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None