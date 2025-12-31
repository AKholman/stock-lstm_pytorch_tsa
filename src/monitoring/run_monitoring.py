# src/monitoring/run_monitoring.py 
#(merged: evidently_report.py + run_evidently.py)

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, RegressionPreset, DataSummaryPreset
from evidently.ui.workspace import Workspace


def run_monitoring():
    # 1. SETUP WORKSPACE
    ws = Workspace.create("my_monitoring_data")
    project = next((p for p in ws.list_projects() if p.name == "Model Monitoring Project"), None)
    if not project:
        project = ws.create_project("Model Monitoring Project")

    # 2. LOAD DATA
    df_ref = pd.read_csv("data/processed/train.csv")
    df_curr = pd.read_csv("data/processed/test.csv")
    
    try:
        df_preds = pd.read_csv("data/predictions/test_predictions.csv")["prediction"]
        df_curr["prediction"] = df_preds
        report = Report(metrics=[RegressionPreset(), DataDriftPreset(), DataSummaryPreset()])
    except (FileNotFoundError, KeyError):
        report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])

    # 3. RUN & SAVE
    run_result = report.run(reference_data=df_ref, current_data=df_curr)
    ws.add_run(project.id, run_result)
    print(f"Monitoring complete for project: {project.id}")

if __name__ == "__main__":
    run_monitoring()