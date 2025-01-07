class Validator:
    """
    Handles the validation phase of the agent in the environment.
    """
    def __init__(self, agent, valid_env, output_path, best_valid_reward, patience):
        """
        Initializes the Validator.

        Args:
            agent: The agent to be validated.
            valid_env: The environment used for validation.
            output_path (str): Path to save validation results.
            best_valid_reward (float): The best reward observed so far.
            patience (int): The maximum number of validation episodes without improvement.
        """
        self.agent = agent
        self.valid_env = valid_env
        self.output_path = output_path
        self.best_valid_reward = best_valid_reward
        self.patience = patience

    def validate(self):
        """
        Runs the validation process.

        Executes the agent in the validation environment, monitors performance,
        and stops early if no improvement is observed beyond the patience threshold.

        Returns:
            tuple: Updated best reward and the number of episodes without improvement.
        """
        state = self.valid_env.reset()
        no_improve_counter = 0

        while True:
            action = self.agent.exploit(state)  # Use exploitation policy
            next_state, reward, done = self.valid_env.step(action)

            if done:
                save_validation_results(self.output_path, self.valid_env)  # Save validation results
                if self.valid_env.total_reward > self.best_valid_reward:
                    self.best_valid_reward = self.valid_env.total_reward
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1

                if no_improve_counter > self.patience:
                    print("Early stopping due to lack of improvement.")
                    break

                return self.best_valid_reward, no_improve_counter

            state = next_state  # Update state for the next step
