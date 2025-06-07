# Car Auction Price Prediction

Welcome to the **Car Auction Price Prediction** repository! This project aims to estimate the price of cars based on auction data scraped from **Manheim NZ**. Using a simple regression model, we predict car prices based on several features like the car's **manufacturer**, **model**, **year**, **mileage**, **stolen status**, and **damage status**.

The app is deployed using Streamlit and can be accessed via this link: [Car Auction Price Prediction App](https://nz-car-auction.streamlit.app/).

## Overview

This is a continuation of the [ScrapeGarage project](https://github.com/prasanthsasikumar/ScrapeGarage), which collects car auction data from **Manheim NZ** every midnight. Over the course of two years, the dataset has grown to contain valuable information about cars, including their **make**, **model**, **year**, **mileage**, **stolen status**, **damage description**, and more.

The **trained model** uses this dataset to predict auction prices for **registered cars only**. This model is built for personal use and is not intended for large-scale commercial deployment.

## Features

* **Car Price Prediction**: Input details of a car (e.g., make, model, mileage, damage status) and get an estimated auction price.
* **Model Training**: The model is trained using historical auction data, with the main script likely located in `scripts/` or `src/models/` (e.g., `scripts/train_model.py`).
* **Data Cleaning and Preprocessing**: The data is cleaned and preprocessed in Jupyter Notebooks located in the `notebooks/` folder (e.g., `notebooks/01_data_cleaning.ipynb`).
* **Streamlit Interface**: A simple, user-friendly interface for making predictions, likely available in `src/app/predict_app.py` or `scripts/predict_app.py`.

## Dataset

The dataset used for training is available in the folder `data/raw/`. We have used only the **registered cars** in this dataset for the model. The data collected over the last two years includes:

* **Car Manufacturer**
* **Model**
* **Year**
* **Mileage**
* **Stolen Status**
* **Damage Severity**
* **Other relevant features**

### Data Collection

The dataset was scraped every midnight from **Manheim NZ** ([https://www.manheim.co.nz/](https://www.manheim.co.nz/)). You can view more details about the scraping process in the [ScrapeGarage repository](https://github.com/prasanthsasikumar/ScrapeGarage).

## Installation

### Prerequisites

To run the code locally, you will need Python 3.7+ and the following libraries:

* **pandas**
* **scikit-learn**
* **streamlit**
* **joblib**
* **matplotlib**

### Steps to Get Started

1. Clone the repository:
   ```bash
   git clone https://github.com/prasanthsasikumar/ScrapeGaragePricePrediction.git
   cd ScrapeGaragePricePrediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. To train the model, run the appropriate training script (e.g., located in `scripts/` or `src/models/`):

   ```bash
   ScrapeGaragePricePrediction>python -m src.models.model_runner
   or python scripts/train_model.py 
   ```

4. To launch the Streamlit app for predictions (e.g., located in `src/app/` or `scripts/`):

   ```bash
   streamlit run src/app/streamlit_car_price_prediction.py
   ```

   The app will be available at `http://localhost:8501` by default.

## Files

* `scripts/train_model.py` (example path): Script to train the regression model using the dataset.
* `src/app/predict_app.py` (example path): Streamlit app for interacting with the model and making predictions.
* `notebooks/01_data_cleaning.ipynb`: Jupyter notebook for cleaning and preprocessing the raw dataset.
* `notebooks/03_model_training.ipynb`: Jupyter notebook for training the model and performing initial experiments.
* `requirements.txt`: Python dependencies required for the project.
* `models/`: Folder containing trained model files (e.g., `xgboost_hp_model.pkl`).
* `data/raw/`: Folder containing raw car data.
* `data/processed/`: Folder containing processed datasets.

## Usage

1. Upload a CSV file with car details (must include columns like **Model**, **Mileage**, **Manufacturer**, **Stolen**, **Model Year**, **Damage Severity**).
2. The app will process the data and return the predicted auction price for each car in the dataset.

## Example

An example of the input CSV format:

| Model  | Mileage | Manufacturer | Stolen | Model Year | Damage Severity |
| ------ | ------- | ------------ | ------ | ---------- | --------------- |
| Honda  | 100000  | Honda        | No     | 2015       | Light           |
| Toyota | 80000   | Toyota       | Yes    | 2017       | Heavy           |

Once uploaded, you will get the predicted prices for the cars.

## Acknowledgments

* This project uses data scraped from **Manheim NZ** ([https://www.manheim.co.nz/](https://www.manheim.co.nz/)).
* The regression model was built using **scikit-learn**.
* Tuned XGBoost is the current model selected.
* The Streamlit app provides an easy way to interact with the model.

## Notes

* This project is **for personal/friends use only** and might not be suitable for use outside the NZ car auction community.
* The architecture and model can still be useful if you buy and sell cars from **repo** or **insurance auctions**.
* As this is a personal project, the documentation is limited. Apologies for the lack of detail.

## Contact

If you buy and fix cars from repo or insurance auctions, feel free to **reach out**! I would love to chat about cars, auctions, or potential improvements to this project. You can contact me via \[email or LinkedIn(https://www.linkedin.com/in/prasanth-sasikumar/)].

---

**Enjoy exploring the project!**

---

### Screenshot

![Screenshot](https://github.com/prasanthsasikumar/car_aution_prediction/blob/main/Screenshot.png?raw=true)

