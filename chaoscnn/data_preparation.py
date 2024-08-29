import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_data(data, lookback=10):
    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    # Check if we have enough data points
    if len(data) < lookback:
        raise ValueError(f"Not enough data points. Expected at least {lookback}, but got {len(data)}.")

    # Separate features and target
    features = data.iloc[:, :-1]  # All columns except the last
    target = data.iloc[:, -1]     # Last column

    # Normalize features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    X = []
    y = []
    for i in range(len(data) - lookback):
        X.append(features_scaled[i:i+lookback])
        y.append(target.iloc[i+lookback])

    return np.array(X), np.array(y), scaler

# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('data/accident_data.csv')

    # Prepare data
    X, y, scaler = prepare_data(data, lookback=10)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First X sample:")
    print(X[0])
    print("First y value:", y[0])