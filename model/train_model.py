import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# ── Load data ────────────────────────────────────────────────────────────────
# Resolve path relative to this script so it works on any machine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "health_modeling_ready.csv")

df = pd.read_csv(DATA_PATH)

# Drop rows where the target or key features are missing
# (last 7 rows have no target; a handful of early rows may also be incomplete)
df = df.dropna(subset=["Target_Weight_Next_Week", "Net_Calorie_7Day_Avg", "Weight_7Day_Avg"])

# ── Features & target ────────────────────────────────────────────────────────
FEATURES = ["Net_Calorie_7Day_Avg", "Weight_7Day_Avg"]
TARGET   = "Target_Weight_Next_Week"

X = df[FEATURES].reset_index(drop=True)   # reset so positional slicing is safe
y = df[TARGET].reset_index(drop=True)     # reset index to match X exactly

# ── Train / test split (chronological — no shuffle for time-series) ───────────
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ── Model ────────────────────────────────────────────────────────────────────
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
preds = model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)

print(f"Model Training Complete.")
print(f"  Train rows : {len(X_train)}  |  Test rows : {len(X_test)}")
print(f"  MAE        : {mae:.4f} kg")

results = X_test.copy()
results["Actual"]    = y_test.values
results["Predicted"] = preds
results["Error"]     = (results["Predicted"] - results["Actual"]).abs()
print("\nPer-row test results:")
print(results.to_string(index=False))

# ── Save model ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "weight_predictor.joblib")
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")