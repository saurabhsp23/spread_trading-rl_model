import numpy as np


class TradingEnvironment:
    """
    A trading environment that simulates market conditions for training a reinforcement learning agent.
    """
    def __init__(self, data, lookback, close):
        """
        Initializes the environment with data and trading parameters.

        Args:
            data (DataFrame): Historical market data.
            lookback (int): Number of time steps to include in the state.
            close (Series): Closing prices of the market data.
        """
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
        """
        Executes a step in the environment based on the action provided.

        Args:
            action (int): The action to take.

        Returns:
            tuple: (next_state, reward, done)
                - next_state (ndarray): The next state representation.
                - reward (float): The reward received for the action.
                - done (bool): Whether the episode has ended.
        """
        self.current_step += 1
        done = self.current_step >= len(self.data)
        next_state = np.zeros((self.lookback, self.data.shape[1])) if done else \
                     self.data.iloc[self.current_step - self.lookback:self.current_step].values
        reward = 0
        return next_state, reward, done

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
            ndarray: The initial state of the environment.
        """
        self.reward = 0
        self.done = False
        self.current_step = self.lookback
        self.total_reward = 0
        initial_state = self.data[:self.lookback].values
        self.returns_list = []
        return initial_state
