# this file file handles: target, split, scaling, LSTM sequences, 
# and provide preprocess_data for airflow.

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# ===============================
# Configuration
# ===============================
FEATURES = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]

# ===============================
# Target creation
# ===============================
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Target"] = df["Adj_Close"].shift(-1)
    return df.dropna()

# ===============================
# Train / Val / Test split
# ===============================
def split_data(df, train_ratio=0.8, val_ratio=0.1):
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)

    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size + val_size]
    test = df.iloc[train_size + val_size:]

    return train, val, test

# ===============================
# Scaling
# ===============================
def scale_data(train, val, test):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train[FEATURES])
    y_train = scaler_y.fit_transform(train[["Target"]])

    X_val = scaler_X.transform(val[FEATURES])
    y_val = scaler_y.transform(val[["Target"]])

    X_test = scaler_X.transform(test[FEATURES])
    y_test = scaler_y.transform(test[["Target"]])

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        scaler_X, scaler_y
    )

# ===============================
# LSTM sequence creation
# ===============================
def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []

    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])

    return np.array(Xs), np.array(ys)

# ===============================
# MAIN ENTRYPOINT (FOR train.py)
# ===============================
def preprocess_data(df, time_steps=60):
    """
    Returns exactly what train.py expects
    """

    df = create_target(df)
    train, val, test = split_data(df)

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        _, scaler_y
    ) = scale_data(train, val, test)

    X_train, y_train = create_sequences(X_train, y_train, time_steps)
    X_val, y_val = create_sequences(X_val, y_val, time_steps)
    X_test, y_test = create_sequences(X_test, y_test, time_steps)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        scaler_y
    )

