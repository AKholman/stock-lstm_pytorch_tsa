# Dockerfile create one image (=docker container folder) with all dependencies installed.

FROM apache/airflow:2.9.1-python3.10

USER root
RUN pip install --no-cache-dir torch yfinance pandas numpy scikit-learn matplotlib

USER airflow

