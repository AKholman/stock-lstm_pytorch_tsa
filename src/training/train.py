import os
import torch
import numpy as np
import torch.nn as nn
from math import sqrt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import mlflow
import mlflow.pytorch

from src.data.fetch import fetch_data
from src.data.preprocess import preprocess_data
from src.models.lstm import LSTMModel

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, X_test, y_test, scaler_y, device):
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test.to(device)).cpu().numpy()

    y_true = scaler_y.inverse_transform(y_test.cpu().numpy())
    y_pred = scaler_y.inverse_transform(preds_scaled)

    return {
        "rmse": sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

# -----------------------------
# Training
# -----------------------------
def train_model(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    scaler_y,
    input_dim,
    model_path="models/best_lstm_model.pth",
    scaler_path="models/scaler_y.pkl",
    epochs=60,
    batch_size=32,
    patience=50,
    lr=1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    wait = 0

    # -----------------------------
    # MLflow tracking start
    # -----------------------------
    mlflow.set_experiment("stock_lstm")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "input_dim": input_dim
        })

        for epoch in range(epochs):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val.to(device)), y_val.to(device)).item()

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {np.mean(train_losses):.6f} | Val Loss: {val_loss:.6f}")
            mlflow.log_metric("val_loss", val_loss, step=epoch+1)
            mlflow.log_metric("train_loss", np.mean(train_losses), step=epoch+1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), model_path)
                joblib.dump(scaler_y, scaler_path)
                mlflow.pytorch.log_model(model, "model")
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    break

        metrics = evaluate(model, X_test, y_test, scaler_y, device)
        print("Test metrics:", metrics)
        # Log test metrics
        mlflow.log_metrics(metrics)

    return model, metrics

# -----------------------------
# Airflow entrypoint
# -----------------------------

def run_training():
    # 1. Get data
    df = fetch_data("AAPL") 
    # Ensure you keep the original dataframes or convert the arrays back to DF
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_y = preprocess_data(df)
    
    # --- ADD THIS: Save files for Evidently ---
    os.makedirs("data/processed", exist_ok=True)
    # If preprocess_data returns arrays, convert them to DF or save the original 'df'
    df.iloc[:len(X_train)].to_csv("data/processed/train.csv", index=False)
    df.iloc[len(X_train):].to_csv("data/processed/test.csv", index=False)
    # ------------------------------------------

    input_dim = X_train.shape[2]
    train_model(X_train, y_train, X_val, y_val, X_test, y_test, scaler_y, input_dim)

    # Now call monitoring
    from src.monitoring.run_monitoring import run_monitoring
    run_monitoring()

# -----------------------------
# Direct run
# -----------------------------

if __name__ == "__main__":
    # 1. This starts the data fetching, saving, and training
    run_training()
    
    # 2. This triggers the monitoring after training is finished
    try:
        from src.monitoring.run_monitoring import run_monitoring
        run_monitoring()
    except ImportError as e:
        print(f"Monitoring failed to start: {e}")

