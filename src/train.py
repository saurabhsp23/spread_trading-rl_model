import os
import torch

class Trainer:
    """
    Manages the training process of the agent in the given environment.
    """
    def __init__(self, agent, train_env, batch_size, num_episodes, output_path):
        """
        Initializes the Trainer.

        Args:
            agent: The agent to be trained.
            train_env: The environment in which the agent is trained.
            batch_size (int): The batch size used for updating the agent.
            num_episodes (int): The number of training episodes.
            output_path (str): Path to save training results and models.
        """
        self.agent = agent
        self.train_env = train_env
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.output_path = output_path

    def train(self):
        """
        Executes the training process.

        Trains the agent over the specified number of episodes, periodically
        saving models and training results.

        Returns:
            None
        """
        for episode in range(self.num_episodes):
            print(f"Training Episode {episode}")
            state = self.train_env.reset()

            while True:
                action = self.agent.get_action(state)  # Select action
                next_state, reward, done = self.train_env.step(action)  # Take action
                self.agent.memory.push(state, action, reward, next_state, done)  # Store experience

                if self.train_env.current_step % 100 == 0:
                    self.agent.update(self.batch_size)  # Update agent periodically

                if done:
                    save_training_results(self.output_path, episode, self.train_env)  # Save episode results
                    break

                state = next_state  # Update state

            if episode % 5 == 0:
                torch.save(self.agent.model.state_dict(), f"model-{episode}.pth")  # Save model
