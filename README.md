# Project: Trading Environment with DQN Agent

## Overview
This project implements a trading environment using Deep Q-Learning (DQN) with a modularized structure. It includes data preprocessing, a custom trading environment, a DQN agent with a Transformer-based model, and utilities for indicators like MACD and RSI. The code is organized for clarity and scalability.

## Project Structure

```
src/
│   ├── train.py          # Handles the training loop.
│   ├── validate.py       # Validation logic and early stopping.
│   ├── test.py           # Testing and result logging.
│   ├── run_pipeline.py   # Main orchestrator for the workflow.
│   ├── transformer_model.py  # Transformer-based model definition.
│   ├── trading_environment.py # Custom trading environment.
│   ├── dqn_agent.py      # DQN agent and replay buffer.
│   ├── indicators.py     # MACD and RSI calculation.\
├── data/
│   ├── JPY.xlsx          # Sample JPY data.
│   ├── EUR.xlsx          # Sample EUR data.
│   ├── AUD.xlsx          # Sample AUD data.
│   ├── CAD.xlsx          # Sample CAD data.
```

## Key Features

1. **Custom Trading Environment**
   - Simulates trading using financial data.
   - Implements step, reset, and reward mechanisms.

2. **DQN Agent with Transformer**
   - Utilizes a Transformer model for feature extraction.
   - Includes epsilon-greedy exploration, replay buffer, and gradient updates.

3. **Data Preprocessing and Indicators**
   - Supports MACD and RSI calculations for feature engineering.
   - Preprocesses data to handle missing values and normalize inputs.

4. **Pipeline Workflow**
   - Training: Optimizes the DQN agent on a trading environment.
   - Validation: Monitors performance and applies early stopping.
   - Testing: Evaluates agent performance and logs results.

## Setup Instructions

### Prerequisites

- Python 3.8+
- Required Libraries: Install via `pip install -r requirements.txt`
  ```
  pandas
  numpy
  torch
  scikit-learn
  matplotlib
  ```

### Steps to Run the Project

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd project
   ```

2. Prepare data files:
   - Place JPY.xlsx, EUR.xlsx, AUD.xlsx, CAD.xlsx in the `data/` directory.

3. Run the pipeline:
   ```bash
   python scripts/run_pipeline.py
   ```

## Modularized Workflow

### Data Loading and Preprocessing
- **File:** `data_preprocessing.py`
- **Description:**
  - Load data from Excel files.
  - Calculate indicators (MACD, RSI).
  - Normalize and handle missing values.

### Training
- **File:** `train.py`
- **Description:**
  - Implements the training loop for the DQN agent.
  - Saves intermediate results and models.

### Validation
- **File:** `validate.py`
- **Description:**
  - Evaluates the agent’s performance on validation data.
  - Implements early stopping logic.

### Testing
- **File:** `test.py`
- **Description:**
  - Runs the agent on test data and evaluates performance metrics like cumulative returns.

### Utility Modules
- **File:** `file_io.py`
- **Description:**
  - Handles saving/loading data, results, and model checkpoints.
- **File:** `indicators.py`
- **Description:**
  - Contains MACD and RSI calculation functions for feature engineering.

## Customization
- Update `run_pipeline.py` to modify parameters like:
  - `lookback`: Number of past timesteps.
  - `eps_decay`: Rate at which exploration decreases.
  - `lr`: Learning rate for optimization.

## Outputs
- Training, validation, and test results are saved as CSV files in the project directory.
- Model checkpoints are saved every 5 episodes in `model-{episode}.pth`.

## Future Improvements
- **Add More Indicators:** Enhance feature engineering with Bollinger Bands or stochastic oscillators.
- **Hyperparameter Tuning:** Automate tuning using tools like Optuna.
- **Advanced Models:** Experiment with attention mechanisms or LSTMs for improved performance.
- **Visualization:** Add plots for cumulative returns and Sharpe ratios.


