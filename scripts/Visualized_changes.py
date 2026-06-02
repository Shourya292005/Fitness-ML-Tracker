#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

#Loading  the same dataset you used for training
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
df = pd.read_csv(os.path.join(DATA_RAW, "health_modeling_ready.csv"))

#Re-applying the same cleaning/filtering used in training
df = df.dropna(subset=['Target_Weight_Next_Week', 'Net_Calorie_7Day_Avg', 'Weight_7Day_Avg'])

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Weight on the primary axis
ax1.plot(df['Date'], df['Weight_7Day_Avg'], color='blue', label='Weight (kg)')
ax1.set_ylabel('Weight (kg)', color='blue')

# Create a second axis for Calories
ax2 = ax1.twinx()
ax2.bar(df['Date'], df['Net_Calorie_7Day_Avg'], color='orange', alpha=0.3, label='Net Calorie Balance')
ax2.set_ylabel('Calorie Balance', color='orange')

plt.title("Weight vs. Net Calorie Balance Over Time")
plt.show()