class Tester:
    """
    Handles the testing phase of the agent in the environment.
    """
    def __init__(self, agent, test_env, output_path):
        """
        Initializes the Tester with an agent, environment, and output path.

        Args:
            agent: The trained agent to be tested.
            test_env: The environment in which testing is performed.
            output_path (str): Path to save test results.
        """
        self.agent = agent
        self.test_env = test_env
        self.output_path = output_path

    def test(self):
        """
        Executes the testing process.

        Runs the agent in the test environment, collecting actions and saving results.

        Returns:
            None
        """
        state = self.test_env.reset()
        actions = []
        while True:
            action = self.agent.exploit(state)  # Choose action using exploitation
            next_state, reward, done = self.test_env.step(action)  # Take the action
            actions.append(action)

            if done:
                save_test_results(self.output_path, self.test_env, actions)  # Save results when episode ends
                break

            state = next_state  # Move to the next state
