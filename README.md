
# RL Trading System

This project implements a deep reinforcement learning-based trading system using a modular design for maintainability and scalability.

## Source code Explanation

- **`environment.py`**: Contains the `TradingEnvironment` class for simulating the trading environment.
- **`agent.py`**: Implements the `DQNAgent` and `ReplayBuffer` classes for reinforcement learning.
- **`transformer_model.py`**: Defines the Transformer model architecture.
- **`indicators.py`**: Provides technical indicator calculations such as MACD and RSI.
- **`data_processing.py`**: Handles data loading and preprocessing.
- **`main.py`**: Orchestrates the entire workflow, including training, validation, and testing.

## How to Run

1. **Install Dependencies**:
   Ensure you have the required Python libraries installed. Use the following command:
   ```bash
   pip install pandas numpy torch matplotlib
   ```

2. **Prepare Data**:
   Place your financial data files (e.g., `JPY.xlsx`, `EUR.xlsx`) in the `../data/` directory.

3. **Run the Code**:
   Execute `main.py` to start the training process:
   ```bash
   python main.py
   ```

## Features

1. **Trading Environment**:
   - Simulates a trading environment with reward calculation and state management.

2. **Deep Q-Learning Agent**:
   - Uses a Transformer-based architecture for predicting actions in the trading environment.
   - Includes epsilon-greedy exploration for learning.

3. **Technical Indicators**:
   - Computes technical indicators like MACD and RSI to enhance state representation.

4. **Data Preprocessing**:
   - Loads and preprocesses financial data for training.

5. **Modular Design**:
   - Easy to extend and maintain with clearly defined modules.

## Project Workflow

1. **Data Loading**:
   - Load financial and macroeconomic data using `data_processing.py`.
   
2. **Feature Engineering**:
   - Generate MACD and RSI indicators for enhanced features.

3. **Environment Initialization**:
   - Create a `TradingEnvironment` with historical data.

4. **Agent Training**:
   - Use the `DQNAgent` class to train a Transformer-based model.

5. **Validation and Testing**:
   - Evaluate the agent's performance using unseen data.

