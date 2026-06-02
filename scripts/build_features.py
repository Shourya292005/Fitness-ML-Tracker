#!/usr/bin/env python3
from pandas._libs import interval
import pandas as pd
import numpy as np
import os


#  Load the wide dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
df = pd.read_csv(os.path.join(DATA_RAW, "health_wide_full.csv"))

# Ensure dates are sorted sequentially
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Note: Basic imputation is included here so mathematical signals are complete
df['Basal Energy'] = df['Basal Energy'].interpolate(method='linear')
df['Active Energy'] = df['Active Energy'].interpolate(method='linear')
df['Calories_Imputed'] = df['Calories'].fillna(df['Calories'].median())


# --------------------------------------------------------------------------
# PARADIGM 1: Mathematical Representation of Thermodynamics
# --------------------------------------------------------------------------
# Directly engineering (A - B) differences to give tree models an explicit signal
df['Tracker_TDEE'] = df['Basal Energy'] + df['Active Energy']
df['Net_Calorie_Balance'] = df['Calories_Imputed'] - df['Tracker_TDEE']


# --------------------------------------------------------------------------
# PARADIGM 2: Signal Smoothing (Denoising)
# --------------------------------------------------------------------------
# Applying a 7-day rolling window to smooth out water retention/glycogen noise
# min_periods=1 ensures we get averages even for the first week of data
df['Weight_7Day_Avg'] = df['Body Weight'].rolling(window=7, min_periods=1).mean()
df['Net_Calorie_7Day_Avg'] = df['Net_Calorie_Balance'].rolling(window=7, min_periods=1).mean()


# --------------------------------------------------------------------------
# PARADIGM 3: Feature-Label Realignment (Look-Ahead Shifting)
# --------------------------------------------------------------------------
# If we want to predict what happens 7 days from now using TODAY's lagging features, 
# we must pull the future weight data BACK into the current row.

# Regression Target: What will the smoothed weight be exactly 7 days into the future?
df['Target_Weight_Next_Week'] = df['Weight_7Day_Avg'].shift(-7)

# Classification Target: Will the smoothed weight go UP (1) or DOWN/FLAT (0) next week?
df['Target_Weight_Class_Next_Week'] = (df['Target_Weight_Next_Week'] > df['Weight_7Day_Avg']).astype(int)


# --------------------------------------------------------------------------
# VERIFICATION AND SAVING
# --------------------------------------------------------------------------
# Let's inspect a slice of rows to see the alignment in action
preview_columns = [
    'Date', 
    'Net_Calorie_7Day_Avg',    # Feature: Lagging 7-day average surplus/deficit
    'Weight_7Day_Avg',         # Feature: Current smoothed baseline weight
    'Target_Weight_Next_Week', # Target: Future ground-truth weight (Realignment)
    'Target_Weight_Class_Next_Week' # Target: Future binary direction (Realignment)
]

print("--- Realignment Matrix Preview ---")
print(df[preview_columns].head(10)) 



# Save the finalized dataset ready for model training (XGBoost, Random Forest, etc.)
df.to_csv(r"C:\Users\shour\OneDrive\Desktop\Smart calories tracker\data manipulation\health_modeling_ready.csv", index=False)
print("\nPipeline execution complete. Dataset saved to 'health_modeling_ready.csv'.")