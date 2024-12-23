from environment import TradingEnvironment
from agent import DQNAgent
from indicators import calculate_macd, calculate_rsi
from data_processing import load_data, preprocess_data

if __name__ == '__main__':
    # Load and preprocess data
    filepaths = ["../data/JPY.xlsx", "../data/EUR.xlsx", "../data/AUD.xlsx", "../data/CAD.xlsx"]
    data = load_data(filepaths)
    data['MACD'] = calculate_macd(data['Close'])
    data['RSI'] = calculate_rsi(data['Close'])
    data = preprocess_data(data)

    # Initialize environment and agent
    lookback = 30
    train_env = TradingEnvironment(data, lookback, data['Close'])
    agent = DQNAgent(state_dim=lookback, action_dim=3, lr=0.001, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995)

    # Training loop (simplified)
    for episode in range(10):
        state = train_env.reset()
        while True:
            action = agent.get_action(state)
            next_state, reward, done = train_env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            if done:
                break
            state = next_state

