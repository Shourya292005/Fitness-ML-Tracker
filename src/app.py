#!/usr/bin/env python3
import os
import sys
import subprocess
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px  # Line 7: Fixed the hidden space typo
import plotly.graph_objects as go

# Paths (Windows-style safe)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "model", "artifacts")

WIDE_PATH = os.path.join(DATA_RAW, "health_wide_full.csv")
CLEANED_PATH = os.path.join(DATA_RAW, "health_modeling_ready.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "weight_predictor.joblib")
MODEL_CPY_PATH = os.path.join(DATA_RAW, "health_modeling_ready.csv")

st.set_page_config(page_title="Smart Calories Tracker", layout="wide")
st.title("Smart Calories Tracker — Dashboard")

# Sidebar: data entry
st.sidebar.header("Log daily data")
entry_date = st.sidebar.date_input("Date", datetime.today())
calories = st.sidebar.number_input("Total Calories", min_value=0.0, value=2000.0, step=1.0)
carbs = st.sidebar.number_input("Carbs (g)", min_value=0.0, value=250.0, step=1.0)
protein = st.sidebar.number_input("Protein (g)", min_value=0.0, value=75.0, step=1.0)
fat = st.sidebar.number_input("Fat (g)", min_value=0.0, value=70.0, step=1.0)
body_weight = st.sidebar.number_input("Body Weight (kg)", min_value=0.0, value=70.0, step=0.1)
basal_energy = st.sidebar.number_input("Basal Energy (kcal) — optional", min_value=0.0, value=0.0, step=1.0)
active_energy = st.sidebar.number_input("Active Energy (kcal) — optional", min_value=0.0, value=0.0, step=1.0)

if st.sidebar.button("Save entry"):
    # Assemble a row matching expected columns used by cleaning script
    row = {
        "Date": pd.to_datetime(entry_date).strftime("%Y-%m-%d"),
        "Calories": calories,
        "Carbs": carbs,
        "Protein": protein,
        "Fat": fat,
        "Body Weight": body_weight,
        "Basal Energy": basal_energy if basal_energy > 0 else pd.NA,
        "Active Energy": active_energy if active_energy > 0 else pd.NA,
    }

    # Ensure folder exists
    os.makedirs(os.path.dirname(WIDE_PATH), exist_ok=True)

    if os.path.exists(WIDE_PATH):
        df_wide = pd.read_csv(WIDE_PATH)
        df_wide = df_wide[df_wide['Date'] != row['Date']]
        df_wide = pd.concat([df_wide, pd.DataFrame([row])], ignore_index=True)
    else:
        df_wide = pd.DataFrame([row])

    df_wide.to_csv(WIDE_PATH, index=False)
    st.sidebar.success(f"Saved entry for {row['Date']}")

    st.sidebar.info("Executing Pipeline... Watch console logs below:")
    
    # Create a live log box in the sidebar so we can see output strings
    log_box = st.sidebar.empty()
    
    try:
        # 1. RUN CLEANING SCRIPT
        cleaning_script = os.path.join(BASE_DIR, "data manipulation", "cleaning_data.py")
        if os.path.exists(cleaning_script):
            log_box.text("Running data cleaning script...")
            result_clean = subprocess.run([sys.executable, cleaning_script], capture_output=True, text=True, check=False)
            
            if result_clean.returncode != 0:
                st.sidebar.error(f"Cleaning script failed!\nError: {result_clean.stderr}")
                st.stop()  # Stop Streamlit rendering immediately so you can see the error

        # Copy file over
        if os.path.exists(CLEANED_PATH):
            os.makedirs(MODEL_DIR, exist_ok=True)
            pd.read_csv(CLEANED_PATH).to_csv(MODEL_CPY_PATH, index=False)

        # 2. RUN TRAINING SCRIPT
        train_script = os.path.join(BASE_DIR, "model", "train_model.py")
        if os.path.exists(train_script):
            log_box.text("Running model training script...")
            result_train = subprocess.run([sys.executable, train_script], capture_output=True, text=True, check=False)
            
            if result_train.returncode != 0:
                st.sidebar.error(f"Training script failed!\nError: {result_train.stderr}")
                st.stop()

        st.sidebar.success("Pipeline executed successfully!")
        st.cache_data.clear()
        st.rerun()

    except Exception as e:
        st.sidebar.error(f"Pipeline error: {e}")

# ==========================================
# DATA ENGINE LAYER (WITH FALLBACK CALCULATIONS)
# ==========================================
@st.cache_data(ttl=15)
def load_and_process_metrics():
    if os.path.exists(MODEL_CPY_PATH):
        df = pd.read_csv(MODEL_CPY_PATH, parse_dates=["Date"])
    elif os.path.exists(CLEANED_PATH):
        df = pd.read_csv(CLEANED_PATH, parse_dates=["Date"])
    elif os.path.exists(WIDE_PATH):
        df = pd.read_csv(WIDE_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        return pd.DataFrame()

    df = df.sort_values('Date').reset_index(drop=True)
    
    # Generate structural columns if the backend processing pipeline hasn't run yet
    if 'Weight_7Day_Avg' not in df.columns and 'Body Weight' in df.columns:
        df['Weight_7Day_Avg'] = df['Body Weight'].rolling(window=7, min_periods=1).mean()
    if 'Net_Calorie_Balance' not in df.columns:
        bmr = df['Basal Energy'].fillna(1800)
        tdee = bmr + df['Active Energy'].fillna(400)
        df['Net_Calorie_Balance'] = df['Calories'] - tdee
    if 'Net_Calorie_7Day_Avg' not in df.columns:
        df['Net_Calorie_7Day_Avg'] = df['Net_Calorie_Balance'].rolling(window=7, min_periods=1).mean()
        
    return df

modeling_df = load_and_process_metrics()

if modeling_df.empty:
    st.info("👋 Welcome! Your metrics engine is ready. Log your first daily entry in the sidebar to populate the dashboard view.")
else:
    # ==========================================
    # INTERACTIVE DATE-RANGE TIMEFRAME FILTER
    # ==========================================
    min_date = modeling_df['Date'].min().to_pydatetime()
    max_date = modeling_df['Date'].max().to_pydatetime()
    
    selected_range = st.slider(
        "📅 Select Analytics Timeframe Window",
        min_value=min_date,
        max_value=max_date,
        value=(max(min_date, max_date - timedelta(days=30)), max_date),
        format="YYYY-MM-DD"
    )
    
    filtered_df = modeling_df[
        (modeling_df['Date'] >= selected_range[0]) & 
        (modeling_df['Date'] <= selected_range[1])
    ]

    # ==========================================
    # HIGH-LEVEL SUMMARY KPI CARDS
    # ==========================================
    latest_row = modeling_df.iloc[-1]
    
    if len(modeling_df) >= 8:
        prior_week_row = modeling_df.iloc[-8]
        weight_delta = latest_row['Body Weight'] - prior_week_row['Body Weight']
    else:
        weight_delta = 0.0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Current Weight", value=f"{latest_row['Body Weight']:.1f} kg", delta=f"{weight_delta:.1f} kg (vs last week)", delta_color="inverse")
    with m2:
        # Check if the calories value is missing (NaN) before converting to int
        raw_cal = latest_row['Calories']
        cal_value = int(raw_cal) if pd.notna(raw_cal) else 0
        st.metric(label="Logged Calories", value=f"{cal_value} kcal")
    with m3:
        avg_7d_net = filtered_df['Net_Calorie_Balance'].tail(7).mean()
        cal_bal = int(avg_7d_net) if pd.notna(avg_7d_net) else 0
        st.metric(
            label="7-Day Avg Net Balance", 
            value=f"{cal_bal} kcal", 
            delta="Surplus" if cal_bal > 0 else "Deficit", 
            delta_color="normal" if cal_bal < 0 else "inverse"
        )

    st.markdown("---")

    # ==========================================
    # TABBED NAVIGATION INTERFACE
    # ==========================================
    tab_trends, tab_ai, tab_ledger = st.tabs([
        "📊 Interactive Trends", 
        "🤖 AI Predictions", 
        "📋 Historical Data Ledger"
    ])

    # ==========================================
    # TAB 1: DYNAMIC ATTRIBUTE SELECTOR BOX
    # ==========================================
    with tab_trends:
        st.subheader("🎯 Custom Attribute Analytics")
        
        # This choice selector dynamically flips the visualization target lower down
        metric_options = ["Body Weight & 7-Day Avg", "Total Calories", "Protein (g)", "Carbs (g)", "Fat (g)", "Active Energy"]
        selected_metric = st.radio(
            "Select an attribute option to inspect trend charts:", 
            options=metric_options, 
            horizontal=True
        )
        
        st.markdown("---")
        
        if selected_metric == "Body Weight & 7-Day Avg":
            fig = px.line(filtered_df, x='Date', y='Weight_7Day_Avg', title='📈 7-Day Smoothed Body Weight Trend (kg)', line_shape='spline', color_discrete_sequence=['#2ECC71'])
            fig.add_scatter(x=filtered_df['Date'], y=filtered_df['Body Weight'], mode='markers', name='Daily Weigh-In', marker=dict(opacity=0.5, color='#E74C3C'))
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
        elif selected_metric == "Total Calories":
            filtered_df = filtered_df.copy()
            filtered_df['Balance_Type'] = filtered_df['Net_Calorie_Balance'].apply(lambda x: 'Surplus' if x >= 0 else 'Deficit')
            fig = px.bar(
                filtered_df, x='Date', y='Calories',
                color='Balance_Type',
                color_discrete_map={'Surplus': '#EF553B', 'Deficit': '#636EFA'},
                title='🔥 Daily Intended Calorie Consumption (kcal)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif selected_metric == "Protein (g)":
            if 'Protein' in filtered_df.columns:
                fig = px.line(filtered_df, x='Date', y='Protein', title='🍗 Daily Protein Intake History (g)', markers=True, line_shape='spline', color_discrete_sequence=['#FF9F43'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No explicit protein tracking history discovered.")
                
        elif selected_metric == "Carbs (g)":
            if 'Carbs' in filtered_df.columns:
                fig = px.line(filtered_df, x='Date', y='Carbs', title='🍞 Daily Carbohydrate Load History (g)', markers=True, line_shape='spline', color_discrete_sequence=['#54A0FF'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No explicit carbohydrate tracking data discovered.")
                
        elif selected_metric == "Fat (g)":
            if 'Fat' in filtered_df.columns:
                fig = px.line(filtered_df, x='Date', y='Fat', title='🥑 Daily Dietary Lipid Intake (g)', markers=True, line_shape='spline', color_discrete_sequence=['#10AC84'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No explicit fat intake data metrics found.")
                
        elif selected_metric == "Active Energy":
            if 'Active Energy' in filtered_df.columns:
                fig = px.bar(filtered_df, x='Date', y='Active Energy', title='🏃‍♂️ Tracked Daily Active Energy Expenditure (kcal)', color_discrete_sequence=['#00D2D3'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Log Active Energy metrics inside the optional sidebar container to populate activity profiles.")

    # ==========================================
    # TAB 2: AI ML WEIGHT FORECAST MODEL
    # ==========================================
    with tab_ai:
        st.subheader("🤖 Predictive Intelligence & XGBoost Projections")
        if os.path.exists(MODEL_PATH):
            try:
                import joblib
                model = joblib.load(MODEL_PATH)
                recent = modeling_df.dropna(subset=['Net_Calorie_7Day_Avg', 'Weight_7Day_Avg']).tail(1)
                
                if not recent.empty:
                    X_latest = recent[['Net_Calorie_7Day_Avg', 'Weight_7Day_Avg']]
                    pred = model.predict(X_latest)[0]
                    
                    c_left, c_right = st.columns([1, 2])
                    with c_left:
                        st.markdown("### Next Week Forecast")
                        st.metric(label="Predicted Target Weight (In 7 Days)", value=f"{pred:.2f} kg")
                        
                        expected_change = pred - latest_row['Body Weight']
                        if expected_change < -0.2:
                            st.success(f"📉 Your current energy balance is reducing your mass by roughly {abs(expected_change):.2f} kg per week. Excellent fat-loss velocity.")
                        elif expected_change > 0.2:
                            st.warning(f"📈 Current macro energy parameters indicate a weight gain pace of +{expected_change:.2f} kg. Optimal for lean tissue accrual.")
                        else:
                            st.info("⚖️ System flags your energy matrix as running near homeostasis maintenance limits.")
                            
                    with c_right:
                        fig_g = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = pred,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Predicted Weight Value Line (kg)"},
                            gauge = {
                                'axis': {'range': [latest_row['Body Weight'] - 5, latest_row['Body Weight'] + 5]},
                                'bar': {'color': "#31333F"},
                                'steps': [
                                    {'range': [latest_row['Body Weight'] - 5, latest_row['Body Weight']], 'color': "#E3F2FD"},
                                    {'range': [latest_row['Body Weight'], latest_row['Body Weight'] + 5], 'color': "#FFEBEE"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig_g, use_container_width=True)
                else:
                    st.info("Data features are missing rolling metric averages. Keep logging records to unlock prediction charts.")
            except Exception as e:
                st.error(f"Error compiling analytical prediction layers: {e}")
        else:
            st.info("No trained predictive model file found. Enter additional data entries to run your training pipeline and create your model.")

    # ==========================================
    # TAB 3: HISTORICAL RAW LEDGER GRID
    # ==========================================
    with tab_ledger:
        st.subheader("📋 Comprehensive Log History Database")
        st.dataframe(
            filtered_df.sort_values(by="Date", ascending=False),
            use_container_width=True,
            column_config={
                "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
                "Calories": st.column_config.NumberColumn("Energy (kcal)", format="%d"),
                "Body Weight": st.column_config.NumberColumn("Weight (kg)", format="%.2f"),
                "Protein": st.column_config.NumberColumn("Protein (g)"),
                "Carbs": st.column_config.NumberColumn("Carbs (g)"),
                "Fat": st.column_config.NumberColumn("Fat (g)")
            }
        )