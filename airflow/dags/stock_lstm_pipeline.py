# this is the Airflow DAG definition file for scheduling the LSTM model training pipeline:
# Airflow runs run_training() once per schedule

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from src.training.train import run_training

with DAG(
    dag_id="stock_lstm_training",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    train_task = PythonOperator(
        task_id="train_lstm_model",
        python_callable=run_training,
    )

    train_task
