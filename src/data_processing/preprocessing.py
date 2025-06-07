import os
import glob
import pandas as pd
import yaml

# Load configuration
def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
print(f"Loading configuration from: {config_path}")
config = load_config(config_path)

# Dataset operations
def load_dataset(file_path):
    """Loads a dataset into a pandas DataFrame."""
    return pd.read_csv(file_path)

def save_dataset(df, file_path):
    """Saves the DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

# Null value reporting and handling
def report_nulls(df, columns=['Model', 'Mileage', 'Price']):
    """Reports the number of null values in specified columns."""
    null_counts = df[columns].isnull().sum()
    print("Null counts in specified columns:")
    print(null_counts)

def drop_nulls_in_column(df, column_name):
    """Drops rows with null values in a specific column."""
    df_cleaned = df.dropna(subset=[column_name])
    print(f"Removed rows with NaN in '{column_name}'. Remaining rows: {len(df_cleaned)}")
    return df_cleaned

# Column data cleaning functions
def clean_column_with_regex(df, column_name, regex_pattern, conversion_func=None):
    """Cleans a column using regex and optional conversion."""
    df = df.copy()
    df[column_name] = df[column_name].replace({regex_pattern: ''}, regex=True)
    if conversion_func:
        df[column_name] = df[column_name].apply(conversion_func)
    df_cleaned = df[df[column_name] > 0]
    print(f"Removed invalid entries in '{column_name}'. Remaining rows: {len(df_cleaned)}")
    return df_cleaned

def classify_stolen_vehicles(df):
    """Classifies vehicles as stolen based on damage description."""
    df['Stolen'] = df['Damage description'].str.contains('stolen|ignition|vandalised', case=False, na=False).replace({True: 'Yes', False: 'No'})
    print("Stolen vehicle classification added.")
    return df

def extract_model_year(df):
    """Extracts and adds the model year from the 'Link' column."""
    last_segment = df['Link'].str.extract(r'/([^/]+)/?$')[0]
    model_year = last_segment.str.extract(r'^(\d{4})')[0]  # Ensure this is a Series
    model_year = model_year.str.replace(r'\.$', '', regex=True)  # Perform replace on the Series
    model_year = pd.to_numeric(model_year, errors='coerce')

    df['Model Year'] = model_year
    df = df.dropna(subset=['Model Year'])
    df = df[df['Model Year'].apply(lambda x: isinstance(x, (int, float)) and x.is_integer())]
    df['Model Year'] = df['Model Year'].astype(int)
    
    print(f"Model Year extracted. Remaining rows: {len(df)}")
    return df

def extract_damage_severity(df):
    """Extracts damage severity from the 'Damage description' column. If it contains 'Light', assign 'Light', medium, assign 'Medium', and if it contains 'Heavy', assign 'Heavy'. If none of these are present, assign 'Light'. Not case sensitive."""
    df['Damage Severity'] = df['Damage description'].str.lower().str.extract(r'(light|medium|heavy)', expand=False).fillna('Light')
    df['Damage Severity'] = df['Damage Severity'].str.capitalize()  # Capitalize the first letter
    print("Damage severity extracted and assigned.")
    return df

# Dataset merging and deduplication
def load_and_merge_csv_files(raw_dir):
    """Loads all CSV files in the raw data directory and merges them."""
    csv_files = glob.glob(os.path.join(raw_dir, '*.csv'))
    all_data = []
    
    for file in csv_files:
        print(f"Loading file: {file}")
        df = pd.read_csv(file)
        df_filtered = df[df['Registration Status'] == 'Yes'].copy()  # Ensure it's a copy
        
        # Add date based on filename (format 'car_data_YYYY-MM-DD.csv')
        date_str = os.path.basename(file).split('_')[-1].replace('.csv', '')
        
        # Create a Series with the same index as df_filtered to avoid the ValueError
        df_filtered['Date'] = pd.Series(pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce'), index=df_filtered.index)
        df_filtered = df_filtered.dropna(axis=1, how='all')
        all_data.append(df_filtered)
    
    merged_data = pd.concat(all_data, ignore_index=True)
    return merged_data


def keep_highest_price_entries(df):
    """Keeps the highest priced entry for each unique 'Link'."""
    df = clean_column_with_regex(df, 'Price', r'[\$,]', float)
    df_sorted = df.sort_values(by=['Link', 'Price'], ascending=[True, False])
    return df_sorted.drop_duplicates(subset='Link', keep='first')

# Data cleaning workflow
def clean_dataset(df):
    """Performs various data cleaning operations."""
    # Report nulls in key columns
    report_nulls(df)

    # Clean and format columns
    df = drop_nulls_in_column(df, 'Mileage')
    df = clean_column_with_regex(df, 'Mileage', r'\D', float)
    df = drop_nulls_in_column(df, 'Price')

    # Additional cleaning tasks
    df = classify_stolen_vehicles(df)
    df = extract_model_year(df)
    df = extract_damage_severity(df)
    
    return df

# Main function
def main():
    """Main execution flow for loading, cleaning, and saving the dataset."""
    raw_dir = config['data']['raw_dir']
    output_file = config['data']['dataset_file']
    
    # Load and process data
    merged_data = load_and_merge_csv_files(raw_dir)
    cleaned_data = keep_highest_price_entries(merged_data)
    final_data = clean_dataset(cleaned_data)

    # Save the cleaned data
    save_dataset(final_data, output_file)
    print("Data cleaning completed successfully.")

if __name__ == "__main__":
    main()
