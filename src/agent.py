import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from transformer_model import TransformerModel

class DQNAgent:
    """
    Implements a Deep Q-Network (DQN) agent using a Transformer-based model.
    The agent interacts with the environment to learn an optimal policy.
    """
    def __init__(self, state_dim, action_dim, lr, gamma, eps_start, eps_end, eps_decay, load_model=False):
        """
        Initializes the DQNAgent with its parameters and model.

        Args:
            state_dim (int): Dimensionality of the input state.
            action_dim (int): Number of possible actions.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            eps_start (float): Starting value of epsilon for exploration.
            eps_end (float): Minimum value of epsilon.
            eps_decay (float): Rate at which epsilon decays.
            load_model (bool, optional): Whether to load a pre-trained model. Defaults to False.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.model = TransformerModel(state_dim, state_dim, 1, 2, action_dim).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(10000)

class ReplayBuffer:
    """
    Implements a circular buffer for experience replay, storing transitions
    from the agent's interaction with the environment.
    """
    def __init__(self, capacity):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity (int): Maximum number of transitions the buffer can hold.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Stores a transition in the buffer.

        Args:
            state (np.array): The state before taking the action.
            action (int): The action taken by the agent.
            reward (float): The reward received for the action.
            next_state (np.array): The state after taking the action.
            done (bool): Whether the episode is done.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: Arrays of states, actions, rewards, next_states, and done flags.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.stack(dones)

    def __len__(self):
        """
        Returns the current size of the buffer.

        Returns:
            int: Number of transitions currently stored.
        """
        return len(self.buffer)
