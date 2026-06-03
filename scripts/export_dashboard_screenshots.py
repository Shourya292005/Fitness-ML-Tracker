#!/usr/bin/env python3
import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DATA_PATH = os.path.join(BASE_DIR, "model", "health_modeling_ready.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "weight_predictor.joblib")
OUTPUT_DIR = os.path.join(BASE_DIR, "docs", "screenshots")

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)


def load_dashboard_data():
    df = pd.read_csv(MODEL_DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def add_header(fig, title, latest_row, filtered_df):
    fig.text(0.06, 0.94, "Smart Calories Tracker - Dashboard", fontsize=28, weight="bold")
    fig.text(0.06, 0.895, title, fontsize=14, color="#555555")

    weight = latest_row["Body Weight"]
    calories = latest_row["Calories"] if pd.notna(latest_row["Calories"]) else 0
    net_balance = filtered_df["Net_Calorie_Balance"].tail(7).mean()

    metric_x = [0.06, 0.38, 0.70]
    labels = ["Current Weight", "Logged Calories", "7-Day Avg Net Balance"]
    values = [f"{weight:.1f} kg", f"{int(calories)} kcal", f"{int(net_balance)} kcal"]
    for x, label, value in zip(metric_x, labels, values):
        fig.text(x, 0.82, label, fontsize=12, color="#555555")
        fig.text(x, 0.765, value, fontsize=24, weight="bold", color="#172033")


def export_trends(df):
    filtered = df.tail(31).copy()
    latest = df.iloc[-1]

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    add_header(fig, "Interactive Trends", latest, filtered)

    ax = fig.add_axes([0.06, 0.13, 0.88, 0.52])
    ax.plot(filtered["Date"], filtered["Weight_7Day_Avg"], color="#2ECC71", linewidth=3, label="7-day average")
    ax.scatter(filtered["Date"], filtered["Body Weight"], color="#E74C3C", alpha=0.55, label="Daily weigh-in")
    ax.set_title("7-Day Smoothed Body Weight Trend (kg)", loc="left", fontsize=16, weight="bold", pad=18)
    ax.set_ylabel("Weight (kg)")
    ax.set_xlabel("Date")
    ax.grid(axis="y", color="#E6E8EF")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.autofmt_xdate(rotation=0)

    fig.savefig(os.path.join(OUTPUT_DIR, "dashboard-trends.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def export_ai(df):
    filtered = df.tail(31).copy()
    latest = df.iloc[-1]

    model = joblib.load(MODEL_PATH)
    recent = df.dropna(subset=["Net_Calorie_7Day_Avg", "Weight_7Day_Avg"]).tail(1)
    prediction = model.predict(recent[["Net_Calorie_7Day_Avg", "Weight_7Day_Avg"]])[0]
    expected_change = prediction - latest["Body Weight"]

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    add_header(fig, "AI Predictions", latest, filtered)

    fig.text(0.06, 0.62, "Next Week Forecast", fontsize=18, weight="bold")
    fig.text(0.06, 0.54, "Predicted Target Weight (In 7 Days)", fontsize=12, color="#555555")
    fig.text(0.06, 0.46, f"{prediction:.2f} kg", fontsize=38, weight="bold", color="#172033")
    fig.text(
        0.06,
        0.39,
        f"Expected weekly change: {expected_change:+.2f} kg",
        fontsize=14,
        color="#0B7A3B" if expected_change <= 0 else "#9A4B00",
    )

    ax = fig.add_axes([0.50, 0.18, 0.38, 0.45], polar=True)
    min_weight = latest["Body Weight"] - 5
    max_weight = latest["Body Weight"] + 5
    angle = (prediction - min_weight) / (max_weight - min_weight) * 180
    angle = max(0, min(180, angle))
    theta = [i * 3.14159 / 180 for i in range(181)]
    ax.plot(theta, [1] * len(theta), color="#DDE3EA", linewidth=24)
    gauge_points = int(angle) + 1
    ax.plot(theta[:gauge_points], [1] * gauge_points, color="#31333F", linewidth=24)
    ax.set_theta_offset(3.14159)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.2)
    ax.set_axis_off()
    ax.text(0, 0.28, f"{prediction:.2f} kg", ha="center", va="center", fontsize=22, weight="bold")

    fig.savefig(os.path.join(OUTPUT_DIR, "dashboard-ai.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def export_ledger(df):
    filtered = df.tail(15).copy()
    latest = df.iloc[-1]

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    add_header(fig, "Historical Data Ledger", latest, df.tail(31))

    table_df = filtered[
        ["Date", "Body Weight", "Calories", "Protein", "Active Energy", "Net_Calorie_Balance"]
    ].copy()
    table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m-%d")
    for col in ["Body Weight", "Calories", "Protein", "Active Energy", "Net_Calorie_Balance"]:
        table_df[col] = table_df[col].map(lambda value: "" if pd.isna(value) else f"{value:.1f}")

    ax = fig.add_axes([0.04, 0.08, 0.92, 0.58])
    ax.axis("off")
    table = ax.table(
        cellText=table_df.iloc[::-1].values,
        colLabels=["Date", "Weight", "Calories", "Protein", "Active Energy", "Net Balance"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#E6E8EF")
        if row == 0:
            cell.set_text_props(weight="bold", color="#172033")
            cell.set_facecolor("#F4F6F8")
        else:
            cell.set_facecolor("white")

    fig.savefig(os.path.join(OUTPUT_DIR, "dashboard-ledger.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_dashboard_data()
    export_trends(df)
    export_ai(df)
    export_ledger(df)
    print(f"Dashboard screenshots exported to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
