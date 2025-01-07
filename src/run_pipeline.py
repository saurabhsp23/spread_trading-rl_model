# run_pipeline.py
from train import Trainer
from validate import Validator
from test import Tester
from data_preprocessing import DataPreprocessor
from environment import TradingEnvironment
from agent import DQNAgent
from indicators import Indicators

def load_and_preprocess_data(filepaths):
    """
    Loads data from the specified filepaths, calculates indicators, and preprocesses it.

    Args:
        filepaths (list): List of filepaths to data files.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    print("Loading data...")
    data = DataPreprocessor.load_data(filepaths)
    print("Calculating indicators...")
    data['MACD'] = Indicators.calculate_macd(data['JPY1M'])
    data['RSI'] = Indicators.calculate_rsi(data['JPY1M'])
    print("Preprocessing data...")
    return DataPreprocessor.preprocess_data(data)

def setup_environment(data, lookback):
    """
    Splits the data into train, validation, and test sets and creates environments.

    Args:
        data (pd.DataFrame): The preprocessed data.
        lookback (int): The number of time steps in the lookback window.

    Returns:
        tuple: Training, validation, and testing environments.
    """
    train_data, valid_data, test_data = split_data(data)
    close = data[['JPY1M']].rename(columns={"JPY1M": "Spread"})
    train_env = TradingEnvironment(train_data, lookback, close)
    valid_env = TradingEnvironment(valid_data, lookback, close)
    test_env = TradingEnvironment(test_data, lookback, close)
    return train_env, valid_env, test_env

def setup_agent(lookback):
    """
    Initializes the DQN agent.

    Args:
        lookback (int): The state dimension for the agent.

    Returns:
        DQNAgent: The initialized agent.
    """
    return DQNAgent(
        state_dim=lookback, action_dim=3, lr=0.001, gamma=0.99,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995
    )


if __name__ == '__main__':
    # Filepaths for input data
    filepaths = ["../data/JPY.xlsx", "../data/EUR.xlsx", "../data/AUD.xlsx", "../data/CAD.xlsx"]

    # Load and preprocess data
    data = load_and_preprocess_data(filepaths)

    # Environment setup
    lookback = 30
    train_env, valid_env, test_env = setup_environment(data, lookback)

    # Agent setup
    agent = setup_agent(lookback)

    # Output filepaths
    train_output_path = 'train_output.csv'
    valid_output_path = 'valid_output.csv'
    test_output_path = 'test_output.csv'

    # Run pipeline
    print("Starting training...")
    trainer = Trainer(agent, train_env, batch_size=217, num_episodes=500, output_path=train_output_path)
    trainer.train()

    print("Starting validation...")
    validator = Validator(agent, valid_env, valid_output_path, best_valid_reward=float('-inf'), patience=100)
    validator.validate()

    print("Starting testing...")
    tester = Tester(agent, test_env, test_output_path)
    tester.test()
