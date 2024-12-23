from train import train
from validate import validate
from test import test
from data_preprocessing import load_data, preprocess_data
from environment import TradingEnvironment
from agent import DQNAgent
from indicators import calculate_macd, calculate_rsi

if __name__ == '__main__':
    # Data loading and preprocessing
    filepaths = ["../data/JPY.xlsx", "../data/EUR.xlsx", "../data/AUD.xlsx", "../data/CAD.xlsx"]
    data = load_data(filepaths)
    data['MACD'] = calculate_macd(data['JPY1M'])
    data['RSI'] = calculate_rsi(data['JPY1M'])
    data = preprocess_data(data)

    # Environment setup
    lookback = 30
    close = data[['JPY1M']].rename(columns={"JPY1M": "Spread"})
    train_data, valid_data, test_data = split_data(data)
    train_env = TradingEnvironment(train_data, lookback, close)
    valid_env = TradingEnvironment(valid_data, lookback, close)
    test_env = TradingEnvironment(test_data, lookback, close)

    # Agent setup
    agent = DQNAgent(
        state_dim=lookback, action_dim=3, lr=0.001, gamma=0.99,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995
    )

    # Run training, validation, and testing
    train_output_path = 'train_output.csv'
    valid_output_path = 'valid_output.csv'
    test_output_path = 'test_output.csv'

    train(agent, train_env, batch_size=217, num_episodes=500, output_path=train_output_path)
    best_valid_reward = float('-inf')
    best_valid_reward, _ = validate(agent, valid_env, valid_output_path, best_valid_reward, patience=100)
    test(agent, test_env, test_output_path)
