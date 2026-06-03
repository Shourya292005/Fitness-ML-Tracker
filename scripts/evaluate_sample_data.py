#!/usr/bin/env python3
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_PATH = os.path.join(BASE_DIR, "data", "sample", "health_sample_anonymized.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(BASE_DIR, "model", "weight_predictor.joblib")

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

FEATURES = ["Net_Calorie_7Day_Avg", "Weight_7Day_Avg"]
TARGET = "Target_Weight_Next_Week"


def prepare_sample_features(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Basal Energy"] = df["Basal Energy"].interpolate(method="linear")
    df["Active Energy"] = df["Active Energy"].interpolate(method="linear")
    df["Calories_Imputed"] = df["Calories"].fillna(df["Calories"].median())
    df = df.ffill().bfill()

    df["Tracker_TDEE"] = df["Basal Energy"] + df["Active Energy"]
    df["Net_Calorie_Balance"] = df["Calories_Imputed"] - df["Tracker_TDEE"]
    df["Weight_7Day_Avg"] = df["Body Weight"].rolling(window=7, min_periods=1).mean()
    df["Net_Calorie_7Day_Avg"] = df["Net_Calorie_Balance"].rolling(window=7, min_periods=1).mean()
    df[TARGET] = df["Weight_7Day_Avg"].shift(-7)
    return df.dropna(subset=[TARGET, *FEATURES]).reset_index(drop=True)


def main():
    df = prepare_sample_features(SAMPLE_PATH)
    model = joblib.load(MODEL_PATH)
    predictions = pd.Series(model.predict(df[FEATURES]), index=df.index)
    actual = df[TARGET]
    errors = (predictions - actual).abs()
    r2 = "n/a" if actual.nunique() < 2 else f"{r2_score(actual, predictions):.4f}"

    print("Anonymized Sample Smoke Test")
    print(f"  Rows      : {len(df)}")
    print(f"  MAE       : {mean_absolute_error(actual, predictions):.4f} kg")
    print(f"  RMSE      : {mean_squared_error(actual, predictions) ** 0.5:.4f} kg")
    print(f"  R2        : {r2}")
    print(f"  <=0.5 kg  : {(errors <= 0.5).mean() * 100:.1f}%")
    print(f"  <=1.0 kg  : {(errors <= 1.0).mean() * 100:.1f}%")
    print(
        "  Note      : this is a scaled anonymized smoke test, not the validation "
        "set used for final model accuracy."
    )


if __name__ == "__main__":
    main()
