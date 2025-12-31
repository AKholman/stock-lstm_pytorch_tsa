import torch
import joblib
import numpy as np

def predict(model, X, scaler_path="models/scaler_y.pkl", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    with torch.no_grad():
        X = X.to(device)
        pred_scaled = model(X).cpu().numpy()

    scaler_y = joblib.load(scaler_path)
    return scaler_y.inverse_transform(pred_scaled)
