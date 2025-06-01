import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import numpy as np
from xgboost import XGBRegressor  # Import XGBoost

# Load dataset
file_path = 'sub_dataset.csv'
df = pd.read_csv(file_path)

# Check if all values in 'Price' column are numeric
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

#Calculate 'Age' from 'Model Year'
from datetime import datetime
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

# Using XGBRegressor instead of RandomForestRegressor
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=6))
])

# Train model
model.fit(X_train, y_train)
joblib.dump(model, 'car_price_xgboost_model.pkl')
print("XGBoost model trained and saved to 'car_price_xgboost_model.pkl'.")

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Car Prices")
plt.grid(True)
plt.tight_layout()
plt.show()
