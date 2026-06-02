# Smart Calories Tracker

A comprehensive health and nutrition tracking application powered by machine learning.

## 📁 Project Structure

```
Smart calories tracker/
├── src/                          # Main application code
│   └── app.py                   # Streamlit web application
│
├── scripts/                      # Data processing & analysis scripts
│   ├── build_features.py        # Feature engineering pipeline
│   └── Visualized_changes.py    # Data visualization utilities
│
├── data/                         # Data storage
│   ├── raw/                     # Raw data files
│   │   ├── health_wide_full.csv
│   │   ├── health_modeling_ready.csv
│   │   └── cleaning_data.py
│   └── processed/               # Processed data
│       └── health_features_engineered.csv
│
├── model/                        # ML models
│   ├── artifacts/               # Trained model files
│   │   └── weight_predictor.joblib
│   ├── train_model.py           # Model training script
│   └── health_modeling_ready.csv
│
├── notebooks/                    # Jupyter notebooks (if any)
│
├── config/                       # Configuration files
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Run the Streamlit App
```bash
streamlit run src/app.py
```
This launches the interactive dashboard for tracking calories and health metrics.

#### Run Feature Engineering
```bash
python scripts/build_features.py
```
Processes raw data and creates engineered features for modeling.

#### Visualize Changes
```bash
python scripts/Visualized_changes.py
```
Generates visualizations of weight and calorie trends.

## 📊 Features

- **Dashboard**: Real-time tracking of calories, macronutrients, and health metrics
- **Predictions**: Machine learning model for weight predictions
- **Feature Engineering**: Automated data processing and feature creation
- **Visualizations**: Interactive charts and graphs

## 📝 Data Files

### Raw Data (`data/raw/`)
- `health_wide_full.csv` - Complete health dataset
- `health_modeling_ready.csv` - Cleaned data ready for modeling
- `cleaning_data.py` - Data cleaning script

### Processed Data (`data/processed/`)
- `health_features_engineered.csv` - Features for ML models

### Model Files (`model/artifacts/`)
- `weight_predictor.joblib` - Trained weight prediction model

## 🛠️ Dependencies

See `requirements.txt` for all Python package dependencies.

---

**Last Updated:** June 2026
