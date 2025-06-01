import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import numpy as np
from xgboost import XGBRegressor
from scipy.stats import uniform, randint

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

# XGBRegressor setup
xgboost_model = XGBRegressor(random_state=42)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgboost_model)
])

# Define parameter grid for RandomizedSearchCV
param_dist = {
    'regressor__n_estimators': randint(50, 500),  # Number of trees
    'regressor__learning_rate': uniform(0.01, 0.2),  # Learning rate
    'regressor__max_depth': randint(3, 15),  # Maximum depth of a tree
    'regressor__subsample': uniform(0.6, 0.4),  # Fraction of samples to use for each tree
    'regressor__colsample_bytree': uniform(0.6, 0.4),  # Fraction of features to use for each tree
    'regressor__gamma': uniform(0, 1),  # Regularization term
    'regressor__min_child_weight': randint(1, 10),  # Minimum child weight (complexity of the tree)
}

# RandomizedSearchCV setup
random_search = RandomizedSearchCV(
    model, 
    param_distributions=param_dist, 
    n_iter=50,  # Number of random combinations to try
    scoring='neg_mean_absolute_error',  # Optimization criterion (MAE)
    cv=3,  # 3-fold cross-validation
    verbose=2,  # Print progress
    random_state=42,
    n_jobs=-1  # Use all processors
)

# Perform randomized search
random_search.fit(X_train, y_train)

# Get the best model
best_model = random_search.best_estimator_
best_params = random_search.best_params_

# Save the best model
joblib.dump(best_model, 'car_price_xgboost_tuned_model.pkl')
print(f"Best model saved to 'car_price_xgboost_tuned_model.pkl'.")
print(f"Best Hyperparameters: {best_params}")

# Predict and evaluate the best model
y_pred = best_model.predict(X_test)
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
