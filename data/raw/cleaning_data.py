import pandas as pd
import numpy as np

# 1. Load the wide dataset
df = pd.read_csv(r'C:\Users\shour\OneDrive\Desktop\Smart calories tracker\data manipulation\health_wide_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 2. Impute Wearable Metrics (Linear Interpolation)
df['Basal Energy'] = df['Basal Energy'].interpolate(method='linear')
df['Active Energy'] = df['Active Energy'].interpolate(method='linear')

# 3. Impute Nutrition Logs 
cal_median = df['Calories'].median()
prot_median = df['Protein'].median()

df['Calories_Imputed'] = df['Calories'].fillna(cal_median)
df['Protein_Imputed'] = df['Protein'].fillna(prot_median)

# 4. Global Cleanup: Fill any remaining gaps (e.g., at the very start/end)
# We fill forward first, then backward to catch any edge cases.
df = df.ffill().bfill()

# 5. Feature Engineering (Metabolic and Composition Tracking Equations)
df['Tracker_TDEE'] = df['Basal Energy'] + df['Active Energy']
df['Net_Calorie_Balance'] = df['Calories_Imputed'] - df['Tracker_TDEE']

# 6. Noise Reduction (7-Day Rolling Window)
# Since we already filled NaNs above, these rolling calculations will now be stable.
df['Weight_7Day_Avg'] = df['Body Weight'].rolling(window=7, min_periods=1).mean()
df['Net_Calorie_7Day_Avg'] = df['Net_Calorie_Balance'].rolling(window=7, min_periods=1).mean()
df['Steps_7Day_Avg'] = df['Steps'].rolling(window=7, min_periods=1).mean()

# Export
df.to_csv("health_features_engineered.csv", index=False)
print("Pipeline complete! 'health_features_engineered.csv' saved.")