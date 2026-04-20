import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, ColumnDistributionMetric
from evidently.pipeline.column_mapping import ColumnMapping

SEED = 42
DATA_DIR = "data/processed"
REPORT_DIR = "reports/drift"
os.makedirs(REPORT_DIR, exist_ok=True)


def load_data():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_raw.npy"))
    return X, y


def build_dataframe(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "mean":  X.mean(axis=1),
        "std":   X.std(axis=1),
        "min":   X.min(axis=1),
        "max":   X.max(axis=1),
        "range": X.max(axis=1) - X.min(axis=1),
        "rms":   np.sqrt((X ** 2).mean(axis=1)),
    })


def main():
    print("Loading processed data...")
    X, y = load_data()
    print(f"Total samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"Reference (train): {len(X_train)} samples")
    print(f"Current (test):    {len(X_test)} samples")

    reference_df = build_dataframe(X_train)
    current_df   = build_dataframe(X_test)

    print("Running drift analysis...")

    column_mapping = ColumnMapping(
        target="label",
        numerical_features=["mean", "std", "min", "max", "range", "rms"],
    )

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])

    report.run(reference_data=reference_df, current_data=current_df)

    report_path = os.path.join(REPORT_DIR, "drift_report.html")
    report.save_html(report_path)
    print(f"Drift report saved to: {report_path}")


if __name__ == "__main__":
    main()
