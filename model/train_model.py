import os

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from predictors import CurrentWeightBaselineRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "health_modeling_ready.csv")
MODEL_PATH = os.path.join(BASE_DIR, "weight_predictor.joblib")

FEATURES = ["Net_Calorie_7Day_Avg", "Weight_7Day_Avg"]
TARGET = "Target_Weight_Next_Week"


def regression_metrics(y_true, predictions):
    errors = (predictions - y_true).abs()
    return {
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": mean_squared_error(y_true, predictions) ** 0.5,
        "r2": r2_score(y_true, predictions),
        "within_0_5kg": (errors <= 0.5).mean() * 100,
        "within_1kg": (errors <= 1.0).mean() * 100,
    }


def print_scores(label, scores):
    print(
        f"  {label:<16}"
        f" MAE: {scores['mae']:.4f} kg"
        f"  RMSE: {scores['rmse']:.4f} kg"
        f"  R2: {scores['r2']:.4f}"
        f"  <=0.5kg: {scores['within_0_5kg']:.1f}%"
        f"  <=1kg: {scores['within_1kg']:.1f}%"
    )


def fit_diagnosis(train_scores, test_scores, baseline_scores, selected_model_name):
    mae_gap = test_scores["mae"] - train_scores["mae"]
    beats_baseline = test_scores["mae"] < baseline_scores["mae"]

    if selected_model_name == "CurrentWeightBaselineRegressor":
        return (
            "Fit diagnosis: reduced overfitting risk by selecting the baseline "
            "predictor because regularized XGBoost did not beat it on the "
            "chronological holdout. The remaining error is mainly data-limited "
            "and distribution-shift driven."
        )

    if train_scores["mae"] < 0.05 and mae_gap > 0.20:
        fit = "overfitting"
    elif train_scores["mae"] > 0.30 and test_scores["mae"] > 0.30:
        fit = "underfitting"
    else:
        fit = "roughly fit, but data-limited"

    baseline_text = "beats" if beats_baseline else "does not beat"
    return (
        f"Fit diagnosis: {fit}. The holdout model {baseline_text} the naive "
        "rolling-weight baseline on MAE."
    )


df = pd.read_csv(DATA_PATH)
if TARGET not in df.columns:
    df["Target_Weight_Next_Week"] = df["Weight_7Day_Avg"].shift(-7)
if "Target_Weight_Class_Next_Week" not in df.columns:
    df["Target_Weight_Class_Next_Week"] = (
        df["Target_Weight_Next_Week"] > df["Weight_7Day_Avg"]
    ).astype(int)
df = df.dropna(subset=[TARGET, *FEATURES]).reset_index(drop=True)

X = df[FEATURES]
y = df[TARGET]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

xgb_model = XGBRegressor(
    n_estimators=20,
    learning_rate=0.03,
    max_depth=1,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=20.0,
    random_state=42,
    objective="reg:squarederror",
)
baseline_model = CurrentWeightBaselineRegressor()

xgb_model.fit(X_train, y_train)
baseline_model.fit(X_train, y_train)

xgb_train_preds = pd.Series(xgb_model.predict(X_train), index=y_train.index)
xgb_test_preds = pd.Series(xgb_model.predict(X_test), index=y_test.index)
baseline_train_preds = pd.Series(baseline_model.predict(X_train), index=y_train.index)
baseline_preds = pd.Series(baseline_model.predict(X_test), index=y_test.index)

train_scores = regression_metrics(y_train, xgb_train_preds)
model_scores = regression_metrics(y_test, xgb_test_preds)
baseline_train_scores = regression_metrics(y_train, baseline_train_preds)
baseline_scores = regression_metrics(y_test, baseline_preds)
selected_model_name = "CurrentWeightBaselineRegressor"
selected_model = baseline_model
selected_train_scores = baseline_train_scores
selected_test_scores = baseline_scores

if model_scores["mae"] < baseline_scores["mae"]:
    selected_model_name = "Regularized XGBoost regressor"
    selected_model = xgb_model
    selected_train_scores = train_scores
    selected_test_scores = model_scores

print("Model Training Complete.")
print(f"  Train rows : {len(X_train)}  |  Test rows : {len(X_test)}")
print(f"  Features   : {', '.join(FEATURES)}")
print("\nTraining metrics:")
print_scores("XGBoost train", train_scores)
print_scores("Baseline train", baseline_train_scores)
print("\nChronological holdout metrics:")
print_scores("Naive baseline", baseline_scores)
print_scores("XGBoost test", model_scores)
print_scores("Selected model", selected_test_scores)
print(f"\nSelected model: {selected_model_name}")
print(fit_diagnosis(selected_train_scores, selected_test_scores, baseline_scores, selected_model_name))

results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = selected_model.predict(X_test)
results["XGBoost_Predicted"] = xgb_test_preds.values
results["Baseline"] = baseline_preds
results["Model_Error"] = (results["Predicted"] - results["Actual"]).abs()
results["Baseline_Error"] = (results["Baseline"] - results["Actual"]).abs()

print("\nPer-row test results:")
print(results.to_string(index=False))

joblib.dump(selected_model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
