import streamlit as st
import pandas as pd
import joblib
import yaml
import os

# Load configuration
def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load configuration from config.yaml
config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')  # Assuming config.yaml is at the root
config = load_config(config_path)

# Get model and dataset paths from the config
model_name = config['models']['streamlit_model']
print(f"Using model: {model_name}")
model_path = os.path.join(config['models']['model_save_dir'], config['models']['streamlit_model'])
print(f"Loading model from: {model_path}")
dataset_path = config['data']['dataset_file']

model = joblib.load(model_path)
df = pd.read_csv(dataset_path)
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
