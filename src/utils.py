import pandas as pd

def save_dataframe(df, file_path):
    """
    Save a DataFrame to a CSV file.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to save.
    file_path : str
        The path where the CSV file will be saved.
    """
    if df is None:
        print("DataFrame is None. Cannot save.")
        return

    try:
        print(f"Saving DataFrame to: {file_path}")
        df.to_csv(file_path, index=False, encoding='utf-8')
        print("DataFrame saved successfully.")
    except Exception as e:
        print(f"Error saving DataFrame: {e}")

# Add other utility functions as needed, e.g., loading configuration
