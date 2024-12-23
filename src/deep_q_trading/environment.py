import numpy as np

class TradingEnvironment:
    def __init__(self, data, lookback, close):
        self.data = data
        self.close = close
        self.reward = 0
        self.done = False
        self.current_step = lookback
        self.inventory = [0] * lookback
        self.take_profit = 0.4
        self.stop_loss = 0.2
        self.total_reward = 0
        self.returns_list = []
        self.lookback = lookback

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data)
        next_state = np.zeros((self.lookback, self.data.shape[1])) if done else \
                     self.data.iloc[self.current_step - self.lookback:self.current_step].values
        reward = 0
        return next_state, reward, done

    def reset(self):
        self.reward = 0
        self.done = False
        self.current_step = self.lookback
        self.total_reward = 0
        initial_state = self.data[:self.lookback].values
        self.returns_list = []
        return initial_state
