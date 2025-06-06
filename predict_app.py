import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
model = joblib.load('car_price_xgboost_tuned_model.pkl')

# Load dataset to get unique model names
df = pd.read_csv('sub_dataset.csv')
models_list = sorted(df['Model'].dropna().unique())
manufacturers_list = sorted(df['Manufacturer'].dropna().unique())
stolen_options = ['Yes', 'No']
damage_severity_options = sorted(df['Damage Severity'].dropna().unique())

st.title("Car Price Prediction")

manufacturer = st.selectbox("Select Manufacturer", manufacturers_list)
model_name = st.selectbox("Select Model", models_list)
model_year = st.number_input("Enter Model Year", min_value=1990, max_value=2025, value=2015)
mileage = st.number_input("Enter Mileage", min_value=0, max_value=500000, value=50000)
stolen = st.selectbox("Is the car stolen?", stolen_options)
damage_severity = st.selectbox("Select Damage Severity", damage_severity_options)

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'Manufacturer': [manufacturer],
        'Model': [model_name],
        'Model Year': [model_year],
        'Mileage': [mileage],
        'Stolen': [stolen],
        'Damage Severity': [damage_severity],
        'Age': [2025 - model_year]  # Assuming current year is 2025
    })

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Price: ${prediction:,.2f}")
