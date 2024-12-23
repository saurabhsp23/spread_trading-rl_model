import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from transformer_model import Transformer

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, eps_start, eps_end, eps_decay, load_model=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.model = Transformer(state_dim, state_dim, 1, 2, action_dim).to("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(10000)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.stack(dones)

    def __len__(self):
        return len(self.buffer)
