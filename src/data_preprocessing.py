import pandas as pd
import numpy as np

class DataPreprocessor:
    """
    Handles loading and preprocessing of financial data.
    """

    @staticmethod
    def load_data(filepaths):
        """
        Loads and combines data from multiple Excel files.

        Args:
            filepaths (list): List of file paths to Excel files.

        Returns:
            pd.DataFrame: Combined and cleaned data.
        """
        data_frames = [pd.read_excel(fp) for fp in filepaths]
        for df in data_frames:
            df.set_index('Date', inplace=True)
        return pd.concat(data_frames, axis=1).dropna()

    @staticmethod
    def preprocess_data(data):
        """
        Prepares data by calculating percentage changes.

        Args:
            data (pd.DataFrame): Raw financial data.

        Returns:
            pd.DataFrame: Data with percentage changes applied.
        """
        return data.pct_change().dropna()
