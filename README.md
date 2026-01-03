#📈 Stock Price Time Series Analysis & Forecasting (MLOps Project)

This project implements an end-to-end production-ready time series forecasting pipeline for stock prices.
It combines classical time series methods, machine learning, and deep learning, wrapped with modern MLOps tools including Airflow, MLflow, Evidently, and Streamlit for deployment and monitoring.

# ==========================
# Project: AAPL Next-Day Close Price Prediction
# Type: Daily Regression
# Full MLSD Pipeline
# ==========================

# 1. Goal Definition
# ------------------
goal = "Predict next-day close price for AAPL using daily OHLCV data"
task = Regression

# 3. Data Collection
# ------------------
data_source = "Yahoo Finance"
ticker = "AAPL"
frequency = "Daily"
lookback_period = "Last 5 years (~1250 rows)"

____________________________________________
✅ Project Quick Checklist (MLOps)
--------------------------------------------
✅ Problem: Next-day AAPL price forecasting (time-series regression)
✅ Metrics: RMSE, MAE, MAPE (offline evaluation; no SLAs)
✅ Data: Yahoo Finance (yfinance) ingestion with reproducible preprocessing and time-based splits
✅ Models: SARIMA, Random Forest, XGBoost, LSTM (Keras), LSTM (PyTorch – best)
✅ Training & Orchestration: Dockerized Apache Airflow pipeline
✅ Experiment Tracking: MLflow (params, metrics, artifacts)
✅ Serving: Streamlit app deployed on Render
✅ Monitoring: Evidently for data drift & performance (manual runs)
✅ Tradeoffs: Live data re-fetch; retraining triggered manually

___________________________________
🧱 Project Structure (High-Level)
-----------------------------------

stock_lstm_TSA/
│
├── airflow/                  # Airflow (Dockerized)
│   └── dags/stock_lstm_pipeline.py
│
├── src/                      # Core ML logic
│   ├── data/                 # Data fetch & preprocessing
│   ├── models/               # PyTorch LSTM model
│   ├── training/             # Training + evaluation
│   ├── inference/            # Prediction logic
│   └── monitoring/           # Evidently monitoring
│
├── data/processed/           # Train/Test CSVs (Evidently)
├── models/                   # Trained models & scalers
├── mlruns/                   # MLflow artifacts
├── my_monitoring_data/       # Evidently workspace
├── streamlit/                # Streamlit app (Render)
└── requirements.txt

__________________________________
🧠 Models Trained & Evaluated
----------------------------------

The following models were implemented, trained, and evaluated:

SARIMA – classical statistical time series model
Random Forest Regressor – tree-based machine learning model
XGBoost Regressor – gradient boosting model
LSTM (TensorFlow / Keras) – deep learning sequence model
LSTM (PyTorch) – deep learning sequence model

✅ Best-performing model:
LSTM implemented in PyTorch, selected based on validation and test performance.

______________________________
🚀 How to Run the Project
------------------------------

python -m venv stock_env
source stock_env/bin/activate   # macOS/Linux

pip install -r requirements.txt

______________________________________
2️⃣ Run Training (local or via Airflow)
--------------------------------------

python src/training/train.py

cd airflow/docker
docker-compose up

http://localhost:8080

_________________________________
📊 MLflow – Experiment Tracking
---------------------------------

Start MLflow UI:
mlflow ui

Open in browser:
http://127.0.0.1:5000

Tracked items:

Parameters (epochs, learning rate, batch size)
Metrics (RMSE, MAE, MAPE)
Artifacts (model weights, scaler)

_______________________________________
🔍 Evidently – Data & Model Monitoring
---------------------------------------
Run monitoring script:
python src/monitoring/run_monitoring.py

Start Evidently UI:
evidently ui --workspace my_monitoring_data

Open in browser:
http://127.0.0.1:8000

Monitored aspects:

Data drift
Feature statistics
Regression performance (when predictions available)

______________________________
🌐 Streamlit App (Inference)
------------------------------

Run locally:
streamlit run streamlit/app.py

Run on render: 

Github:  https://github.com/AKholman/stock-lstm-pytorch-streamlit-render
Render: https://streamlit-render-lstm-pytorch.onrender.com
Currently, it is suspended to keep free tier sources. It can be actvated at any time. 

Deployment:

The Streamlit app is deployed on Render, loading:
Render: https://streamlit-render-lstm-pytorch.onrender.com
Saved scaler for inverse transformation

____________________________
🧩 Key Technologies
----------------------------

PyTorch – deep learning
Airflow – workflow orchestration
MLflow – experiment tracking
Evidently – monitoring & drift detection
Streamlit – UI & deployment
Docker – Airflow isolation

______________________________
✅ Project Highlights
------------------------------

Clean separation between training, inference, and monitoring
Production-style MLOps workflow
Reproducible experiments
Model and data monitoring ready for real-world usage



**APPENDICES** 

A) LSTM MODEL BUILDING:

Input shape: (60 timesteps, 6 features)
      ▼
┌─────────────────────────------────-┐
│ LSTM(64, return_sequences=True).   │
│ → outputs 64 features per timestep │
│ Output shape: (60, 64)             │
└────────────────────────────-------─┘
      ▼
 Dropout(0.2)
      ▼
┌────────────────────────---------─────┐
│ LSTM(32, return_sequences=False)     │
│ → outputs only final timestep vector │
│ Output shape: (32,)                  │
└──────────────────────────---------───┘
      ▼
 Dropout(0.2)
      ▼
┌─────────────────────────---------------------------────┐
│ Dense(16, activation='relu')                           │
│ → fully connected layer, learns nonlinear combinations │
│ Output shape: (16,)                                    │
└─────────────────────---------------------------────────┘
      ▼
┌─────────────────────────----────┐
│ Dense(1)                        │
│ → final prediction (regression) │
│ Output shape: (1,)              │
└────────────────────────────----─┘


Some details of Pytorch LSTM model traning and testing. 

1.  Using Min–Max scaling we scaled (normalized) the features (X) and the target (y). The scalers are fit only on the training data, then applied to validation and test — which prevents information leakage.

2. Sequence creation - 'def create_sequences(X, y, time_steps=60)' : 
Goal of this step - Transform each continuous 1D timeline of features into overlapping time windows (sequences).
Each sequence of time_steps = 60 days becomes one sample for the LSTM, and the label is the target value right after that window. 
Output: we have NumPy arrays (X_train_seq, y_train_seq, etc.).

But PyTorch models can only work with PyTorch tensors with GPU acceleration and automatic differentiation (autograd).

3. So, Step 4 converts all the NumPy arrays (matrix) into PyTorch tensors and prepares them for efficient mini-batch training.
X(rows, features) -> X(rows, timestep, features), i.e. 2D data (matrix) → 3D sequences (tensor)	LSTM needs (batch, time, features). 
DataLoader (a PyTorch utility) breaks the dataset into mini-batches of 32 samples.


4. MODEL DEFINITION:

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

🔹 Step 4. Data as it flows:

Step	Layer	      Input shape	     Output shape
1	    Input	       (60, 6)	   →     (60, 6).      usually this is not considered as a layer
2	    LSTM(64)	   (60, 6)	   →     (60, 64)
3	    Dropout(0.2)   (60, 64)    →     (60, 64)
4	    LSTM(32)	   (60, 64)	   →     (32,)
5	    Dropout(0.2)	(32,)	   →     (32,)
6	    Dense(16, ReLU)	(32,)	   →     (16,)
7	    Dense(1)	    (16,)	   →     (1,)

SUMMARY:
Total layers: 7
Input layer: (60, 6) → 6 features × 60 timesteps
First LSTM layer: 64 neurons (each learning a temporal pattern)
Output layer: 1 neuron (final continuous prediction)


5. One full batch cycle of training: 
    🔹 Step 1 — Forward pass:
        Input batch → LSTM1 → LSTM2 → fc1 → ReLU → fc2 → Output → Compute prediction.

    🔹 Step 2 — Compute loss:
        Loss = criterion(output, true_y). (e.g., Mean Squared Error for regression)

    🔹 Step 3 — Backpropagation:
        Call loss.backward(): 
            → PyTorch automatically computes gradients ∂loss/∂weight for all layers.

    🔹 Step 4 — Optimizer update:
        optimizer.step()
            → All layer parameters are adjusted based on gradients.

    🔹 Step 5 — Next batch:
        LSTM starts fresh with new sequence inputs.
            Gradients from the previous batch are cleared (optimizer.zero_grad()).
                The updated weights now slightly better fit the data → model improves.

