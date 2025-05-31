import csv
import pandas as pd

file_path = 'sub_dataset.csv'

def print_csv_headers(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        print("Headers:", headers)

def check_nulls(file_path):
    df = pd.read_csv(file_path)
    null_counts = df[['Model', 'Mileage', 'Price']].isnull().sum()
    print("Null counts in 'Model', 'Mileage' & 'Price':")
    print(null_counts)

def remove_null_mileage(file_path):
    df = pd.read_csv(file_path)
    df_cleaned = df.dropna(subset=['Mileage'])
    df_cleaned.to_csv(file_path, index=False)
    print(f"Rows with NaN in 'Mileage' removed. Remaining rows: {len(df_cleaned)}")

def format_mileage_column(file_path):
    df = pd.read_csv(file_path)
    df['Mileage'] = df['Mileage'].replace({r'\D': ''}, regex=True).astype(float)
    df_cleaned = df[df['Mileage'] > 0]
    df_cleaned.to_csv(file_path, index=False)
    print(f"Rows with 'Mileage' as 0 removed. Remaining rows: {len(df_cleaned)}")

def remove_null_price(file_path):
    df = pd.read_csv(file_path)
    df_cleaned = df.dropna(subset=['Price'])
    df_cleaned.to_csv(file_path, index=False)
    print(f"Rows with NaN in 'Price' removed. Remaining rows: {len(df_cleaned)}")

def format_price_column(file_path):
    df = pd.read_csv(file_path)
    df['Price'] = df['Price'].replace({r'\$': '', ',': ''}, regex=True).astype(float)
    df_cleaned = df[df['Price'] > 0]
    df_cleaned.to_csv(file_path, index=False)
    print(f"Rows with 'Price' as 0 removed. Remaining rows: {len(df_cleaned)}")

def classify_stolen(file_path):
    df = pd.read_csv(file_path)
    df['Stolen'] = df['Damage description'].str.contains('stolen|ignition|vandalised', case=False, na=False).replace({True: 'Yes', False: 'No'})
    df.to_csv(file_path, index=False)
    print("Stolen classification added to the dataset.")

def format_model_year(file_path):
    df = pd.read_csv(file_path)
    last_segment = df['Link'].str.extract(r'/([^/]+)/?$')[0]
    df['Model Year'] = last_segment.str.extract(r'^(\d{4})')
    df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')
    df = df.dropna(subset=['Model Year'])
    df['Model Year'] = df['Model Year'].astype(int)
    df.to_csv(file_path, index=False)
    print(f"Model Year extracted and added. Remaining rows: {len(df)}")


if __name__ == "__main__":
    print_csv_headers(file_path)
    check_nulls(file_path)
    remove_null_mileage(file_path)
    format_mileage_column(file_path)
    remove_null_price(file_path)
    format_price_column(file_path)
    classify_stolen(file_path)
    format_model_year(file_path)
    print("Data cleaning completed successfully.")
