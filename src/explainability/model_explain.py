import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from explainerdashboard import RegressionExplainer, ExplainerDashboard
import os

model = joblib.load('models/xgboost_hp_model.pkl')  
data_path = os.path.join("data", "processed", "registered_cars_dataset.csv")
data = pd.read_csv(data_path)

# === Preprocess data like in training ===
from datetime import datetime

current_year = datetime.now().year
data['Age'] = current_year - data['Model Year']
data = data.dropna(subset=['Age'])
data['Age'] = data['Age'].astype(int)
data['Price'] = data['Price'].astype(float)

X = data[['Model', 'Mileage', 'Manufacturer', 'Stolen', 'Model Year', 'Damage Severity', 'Age']]
y = data['Price']

# === Sample test data for explanation (optional, to keep it light) ===
X_sample = X.sample(n=100, random_state=42)
y_sample = y.loc[X_sample.index]

# === SHAP Explanation ===
# Extract preprocessed data (transform through pipeline's preprocessor)
X_transformed = model.named_steps['preprocessor'].transform(X_sample)

# Get the regressor from the pipeline
regressor = model.named_steps['regressor']

# Use TreeExplainer for XGBoost
explainer = shap.Explainer(regressor)
shap_values = explainer(X_transformed)

# Plot SHAP summary
plt.title("SHAP Feature Importance")
shap.plots.beeswarm(shap_values)
plt.savefig("src/explainability/shap_summary.png")
plt.show()

# === ExplainerDashboard (wrap full pipeline) ===
dashboard_explainer = RegressionExplainer(model, X_sample, y_sample)
ExplainerDashboard(dashboard_explainer).run()
