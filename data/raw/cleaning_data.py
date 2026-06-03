import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

input_path = os.path.join(DATA_RAW, "health_wide_full.csv")
processed_path = os.path.join(DATA_PROCESSED, "health_features_engineered.csv")
cleaned_path = os.path.join(DATA_RAW, "health_modeling_ready.csv")

df = pd.read_csv(input_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

df["Basal Energy"] = df["Basal Energy"].interpolate(method="linear")
df["Active Energy"] = df["Active Energy"].interpolate(method="linear")

df["Calories_Imputed"] = df["Calories"].fillna(df["Calories"].median())
df["Protein_Imputed"] = df["Protein"].fillna(df["Protein"].median())
df = df.ffill().bfill()

df["Tracker_TDEE"] = df["Basal Energy"] + df["Active Energy"]
df["Net_Calorie_Balance"] = df["Calories_Imputed"] - df["Tracker_TDEE"]
df["Weight_7Day_Avg"] = df["Body Weight"].rolling(window=7, min_periods=1).mean()
df["Net_Calorie_7Day_Avg"] = df["Net_Calorie_Balance"].rolling(window=7, min_periods=1).mean()
df["Target_Weight_Next_Week"] = df["Weight_7Day_Avg"].shift(-7)
df["Target_Weight_Class_Next_Week"] = (
    df["Target_Weight_Next_Week"] > df["Weight_7Day_Avg"]
).astype(int)

if "Steps" in df.columns:
    df["Steps_7Day_Avg"] = df["Steps"].rolling(window=7, min_periods=1).mean()

os.makedirs(DATA_PROCESSED, exist_ok=True)
df.to_csv(processed_path, index=False)
df.to_csv(cleaned_path, index=False)
print(f"Pipeline complete. Saved {processed_path} and {cleaned_path}.")
