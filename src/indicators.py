class Indicators:
    """
    A utility class for calculating financial indicators used in trading strategies.

    """

    @staticmethod
    def calculate_macd(data, short_window=12, long_window=26):
        """
        Calculates the Moving Average Convergence Divergence (MACD) for a given time series.

        The MACD is computed as the difference between a short-term Exponential Moving Average (EMA)
        and a long-term EMA. It is a momentum indicator that shows the relationship between two EMAs.

        Returns:
            pd.Series: The MACD values.
        """

        short_ema = data.ewm(span=short_window, adjust=False).mean()
        long_ema = data.ewm(span=long_window, adjust=False).mean()
        return short_ema - long_ema

    @staticmethod
    def calculate_rsi(data, period=14):
        """
        Calculates the Relative Strength Index (RSI) for a given time series.

        The RSI measures the magnitude of recent price changes to evaluate overbought or oversold
        conditions in the price of a stock or other asset.


        Returns:
            pd.Series: The RSI values ranging from 0 to 100.
        """

        delta = data.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0  # Retain only positive changes for gains
        down[down > 0] = 0  # Retain only negative changes for losses
        average_gain = up.rolling(window=period).mean()
        average_loss = abs(down.rolling(window=period).mean())
        rs = average_gain / average_loss
        return 100 - (100 / (1 + rs))
