# Example in main.py (or move to pipeline.py and import)
import os
import pandas as pd
from typing import Dict, Tuple, Optional # For type hinting
from src import data_loader, preprocessing, sentiment_analyzer, visualization, utils

# --- Keep your Configuration and Parameters sections ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_FILE = os.path.join(DATA_DIR, 'song_lyrics.csv') # Use the trimmed one if needed
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'processed_lyrics.csv')
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)
LOAD_NROWS = None # Or None if using full trimmed file

SAMPLE_SIZE = None
LANGUAGE_CODE = 'en'
GENRE_COLUMN = 'tag'
ARTIST_COLUMN = 'artist'
YEAR_COLUMN = 'year'
LYRICS_COLUMN = 'lyrics'
PROCESSED_LYRICS_COLUMN = 'processed_lyrics'

TEXTBLOB_SENTIMENT_COLUMN = 'sentiment_tb' # Output of TextBlob analysis (base name)
TEXTBLOB_POLARITY_COLUMN = f'{TEXTBLOB_SENTIMENT_COLUMN}_polarity' # Already defined
TEXTBLOB_SUBJECTIVITY_COLUMN = f'{TEXTBLOB_SENTIMENT_COLUMN}_subjectivity' 

# --- NEW FUNCTION ---
def run_full_analysis() -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, str]]]:
    """
    Executes the full analysis pipeline and returns results.

    Returns:
        Tuple containing:
        - pd.DataFrame: The final analyzed DataFrame (or None if failed).
        - Dict[str, str]: A dictionary mapping plot names to their file paths (or None if failed).
    """
    print("Executing analysis pipeline...")
    plot_paths = {}

    # 1. Load Data
    df_raw = data_loader.load_dataset(RAW_DATA_FILE, sample_size=SAMPLE_SIZE, nrows=LOAD_NROWS)
    if df_raw is None: return None, None

    # 2. Initial Exploration (optional for streamlit app, maybe just print shape)
    # data_loader.explore_initial_data(df_raw)
    print(f"Loaded data shape: {df_raw.shape}")


    # 3. Filter Language
    df_lang_filtered = data_loader.filter_by_language(df_raw, language_code=LANGUAGE_CODE)
    if df_lang_filtered is None or df_lang_filtered.empty: return None, None
    print(f"Shape after language filter: {df_lang_filtered.shape}")


    # 4. Preprocess Text
    preprocessing.download_nltk_data() # Ensure NLTK data is available
    df_processed = preprocessing.apply_preprocessing_to_column(
        df_lang_filtered, text_column=LYRICS_COLUMN, new_column=PROCESSED_LYRICS_COLUMN
    )
    if df_processed is None: return None, None
    print("Preprocessing complete.")

    # 5. Sentiment Analysis
    df_analyzed = sentiment_analyzer.apply_textblob_sentiment(
        df_processed, text_column=PROCESSED_LYRICS_COLUMN, sentiment_col_name=TEXTBLOB_SENTIMENT_COLUMN
    )
    if df_analyzed is None: return None, None
    print("Sentiment analysis complete.")


    # 6. Visualization - Capture the returned paths
    print("Generating visualizations...")
    if TEXTBLOB_SENTIMENT_COLUMN in df_analyzed.columns:
        path = visualization.plot_sentiment_distribution(
            df_analyzed, sentiment_column=TEXTBLOB_SENTIMENT_COLUMN, save_dir=PLOT_DIR
        )
        if path: plot_paths['distribution'] = path

    # Use POLARITY column for average plots
    if GENRE_COLUMN in df_analyzed.columns:
         path = visualization.plot_sentiment_by_category(
             df_analyzed, category_column=GENRE_COLUMN, sentiment_column=TEXTBLOB_POLARITY_COLUMN, top_n=10, save_dir=PLOT_DIR
         )
         if path: plot_paths['genre'] = path

    if ARTIST_COLUMN in df_analyzed.columns:
         path = visualization.plot_sentiment_by_category(
             df_analyzed, category_column=ARTIST_COLUMN, sentiment_column=TEXTBLOB_POLARITY_COLUMN, top_n=10, save_dir=PLOT_DIR
         )
         if path: plot_paths['artist'] = path

    if YEAR_COLUMN in df_analyzed.columns:
        # Check/convert year column type if needed (add robust check like before)
        if pd.api.types.is_numeric_dtype(df_analyzed[YEAR_COLUMN]):
            path = visualization.plot_sentiment_trends_over_time(
                df_analyzed, year_column=YEAR_COLUMN, sentiment_column=TEXTBLOB_POLARITY_COLUMN, save_dir=PLOT_DIR
            )
            if path: plot_paths['trends'] = path
        else:
             print(f"Year column '{YEAR_COLUMN}' not numeric, skipping trend plot.")


    # 7. Save Processed Data (Optional for Streamlit, but good practice)
    utils.save_dataframe(df_analyzed, PROCESSED_DATA_FILE)
    print("Pipeline finished.")


    return df_analyzed, plot_paths

# Remove or comment out the old __main__ block if app.py is the entry point
# if __name__ == "__main__":
#     run_analysis_pipeline() # Old function call
