import pandas as pd


def load_dataset(file_path, sample_size=None, nrows=None): # Add nrows parameter
    """Loads the dataset from a CSV file."""
    print(f"Loading dataset from: {file_path}")
    try:
        # Pass nrows directly to pd.read_csv
        df = pd.read_csv(file_path, nrows=nrows)

        if sample_size and sample_size < len(df):
            print(f"Sampling {sample_size} rows from the dataset.")
            df = df.sample(n=sample_size, random_state=42) # Use a fixed random state for reproducibility if needed

        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.ParserError as e:
         print(f"Error loading dataset: {e}") # More specific error capture
         return None
    except Exception as e:
        # Catch other potential loading errors (permissions, memory, etc.)
        print(f"An unexpected error occurred during dataset loading: {e}")
        return None


def explore_initial_data(df):
    """
    Perform initial exploration of the dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe.
    """
    if df is None:
        print("Dataframe is None. Cannot explore.")
        return

    print("\nDataset Information:")
    df.info()

    print("\nFirst 5 rows:")
    print(df.head())

    if 'language' in df.columns:
        print("\nLanguage Distribution:")
        print(df['language'].value_counts())
    else:
        print("\n'language' column not found.")

    if 'tag' in df.columns: # Assuming 'tag' column represents genre
        print("\nGenre (Tag) Distribution (Top 10):")
        print(df['tag'].value_counts().head(10))
    else:
        print("\n'tag' (genre) column not found.")

def filter_by_language(df, language_code='en'):
    """
    Filter the dataframe for a specific language.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe.
    language_code : str
        Language code to filter by (default: 'en').

    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe.
    """
    if df is None or 'language' not in df.columns:
        print("Cannot filter by language. Dataframe is None or 'language' column missing.")
        return df # Return original or None

    original_count = len(df)
    df_filtered = df[df['language'] == language_code].copy()
    print(f"\nFiltered for language '{language_code}'.")
    print(f"Original count: {original_count}, Filtered count: {len(df_filtered)}")
    return df_filtered
