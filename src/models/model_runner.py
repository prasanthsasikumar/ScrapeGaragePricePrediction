# model_runner.py
from src.models.regression_model import train_regression_model
from src.models.xgboost_model import train_xgboost_model
from src.models.xgboost_hp_model import train_xgboost_with_tuning
import os
import yaml

# Load configuration
def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
print(f"Loading configuration from: {config_path}")
config = load_config(config_path)

def run_regression():
    """Function to run the regression model."""
    print("Training regression model...")
    model = train_regression_model(config['data']['dataset_file'],
                                   config['models']['model_save_dir'])
    return model

def run_xgboost():
    """Function to run the basic XGBoost model."""
    print("Training XGBoost model...")
    model = train_xgboost_model(config['data']['dataset_file'],
                                   config['models']['model_save_dir'])
    return model

def run_xgboost_with_hp():
    """Function to run the XGBoost model with hyperparameter tuning."""
    print("Training XGBoost model with hyperparameter tuning...")
    model = train_xgboost_with_tuning(config['data']['dataset_file'],
                                   config['models']['model_save_dir'])
    return model

# Main function to select and run a model
if __name__ == "__main__":
    # Example of choosing which model to run
    choice = input("Select model to run (1: Regression, 2: XGBoost, 3: XGBoost with HP): ")
    
    if choice == '1':
        run_regression()
    elif choice == '2':
        run_xgboost()
    elif choice == '3':
        run_xgboost_with_hp()
    else:
        print("Invalid choice.")
