import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import numpy as np
from datetime import datetime
import os

def train_regression_model(file_path, model_save_dir, show_plots=False):
    """Trains a regression model to predict car prices."""
    
    # Load dataset
    df = pd.read_csv(file_path)

    # Ensure 'Price' column values are numeric
    def is_numeric(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    non_numeric_prices = df[~df['Price'].apply(is_numeric)]
    if not non_numeric_prices.empty:
        print("Non-numeric values found in 'Price' column:")
        print(non_numeric_prices)
    else:
        print("All values in 'Price' column are numeric or convertible to numeric.")

    # Calculate 'Age' from 'Model Year'
    current_year = datetime.now().year
    df['Age'] = current_year - df['Model Year']
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df.dropna(subset=['Age'])
    df['Age'] = df['Age'].astype(int)

    # Feature matrix and target vector
    X = df[['Model', 'Mileage', 'Manufacturer', 'Stolen', 'Model Year', 'Damage Severity', 'Age']]
    y = df['Price'].astype(float)  # Ensure y is float

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Define preprocessing and model pipeline
    categorical = ['Model', 'Manufacturer', 'Stolen', 'Damage Severity']
    numerical = ['Mileage', 'Model Year', 'Age']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', 'passthrough', numerical)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Save the trained model
    os.makedirs(model_save_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(__file__))[0] + '.pkl'
    model_save_path = os.path.join(model_save_dir, model_name)
    joblib.dump(model, model_save_path)
    print(f"Model trained and saved to {model_save_path} as {model_name}")

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    if show_plots:
        # Plot actual vs predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs. Predicted Car Prices")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return model  # Return the trained model for potential further use

