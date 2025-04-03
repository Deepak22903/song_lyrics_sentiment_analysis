import os
from src import data_loader, preprocessing, sentiment_analyzer, visualization, utils

# --- Configuration ---
# Consider moving these to a separate config.py or using environment variables
LOAD_NROWS = 2667950
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets directory where main.py is
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_FILE = os.path.join(DATA_DIR, 'song_lyrics.csv') # CHANGE FILENAME IF NEEDED
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'processed_lyrics.csv')
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True) # Ensure the plot directory exists

# --- Parameters ---
SAMPLE_SIZE = None # Set to an integer (e.g., 5000) for testing, None to use full dataset
LANGUAGE_CODE = 'en' # Filter for English lyrics
GENRE_COLUMN = 'tag' # Column name for genre (adjust if different)
ARTIST_COLUMN = 'artist' # Column name for artist
YEAR_COLUMN = 'year' # Column name for year
LYRICS_COLUMN = 'lyrics' # Raw lyrics column
PROCESSED_LYRICS_COLUMN = 'processed_lyrics' # Output of preprocessing
TEXTBLOB_SENTIMENT_COLUMN = 'sentiment_tb' # Output of TextBlob analysis
TEXTBLOB_POLARITY_COLUMN = f'{TEXTBLOB_SENTIMENT_COLUMN}_polarity' # ADD THIS LINE
# TEXTBLOB_SUBJECTIVITY_COLUMN = f'{TEXTBLOB_SENTIMENT_COLUMN}_subjectivity' # Optional: if needed elsewhere

# --- Main Pipeline ---
def run_analysis_pipeline():
    """Executes the full lyrics analysis workflow."""

    print("Starting Lyrics Analysis Pipeline...")


    # 1. Load Data
    df_raw = data_loader.load_dataset(
        RAW_DATA_FILE,
        sample_size=SAMPLE_SIZE,
        nrows=LOAD_NROWS # Pass the nrows value here
    )
    if df_raw is None:
        print("Failed to load data. Exiting.")
        return

    # 2. Initial Exploration
    data_loader.explore_initial_data(df_raw)

    # 3. Filter Language
    df_lang_filtered = data_loader.filter_by_language(df_raw, language_code=LANGUAGE_CODE)
    if df_lang_filtered is None or df_lang_filtered.empty:
        print("No data after language filtering. Exiting.")
        return

    # 4. Preprocess Text
    # Ensure NLTK data is downloaded before calling preprocessing functions that need it
    preprocessing.download_nltk_data()
    df_processed = preprocessing.apply_preprocessing_to_column(
        df_lang_filtered,
        text_column=LYRICS_COLUMN,
        new_column=PROCESSED_LYRICS_COLUMN
    )
    if df_processed is None:
        print("Preprocessing failed. Exiting.")
        return

    # 5. Sentiment Analysis (Lexicon-Based: TextBlob)
    df_analyzed = sentiment_analyzer.apply_textblob_sentiment(
        df_processed,
        text_column=PROCESSED_LYRICS_COLUMN,
        sentiment_col_name=TEXTBLOB_SENTIMENT_COLUMN # Pass the desired column name
    )
    if df_analyzed is None:
        print("Sentiment analysis failed. Exiting.")
        return

    # 6. Visualization <--- START INDENTING HERE
    print("\nGenerating and Saving Visualizations...")
    if TEXTBLOB_SENTIMENT_COLUMN in df_analyzed.columns:
        visualization.plot_sentiment_distribution(
            df_analyzed,
            sentiment_column=TEXTBLOB_SENTIMENT_COLUMN,
            save_dir=PLOT_DIR # Pass the directory
        )
        if GENRE_COLUMN in df_analyzed.columns:
            visualization.plot_sentiment_by_category(
                df_analyzed,
                category_column=GENRE_COLUMN,
                sentiment_column=TEXTBLOB_POLARITY_COLUMN, 
                top_n=10,
                save_dir=PLOT_DIR
            )
        if ARTIST_COLUMN in df_analyzed.columns:
             visualization.plot_sentiment_by_category(
                 df_analyzed,
                 category_column=ARTIST_COLUMN,
                 sentiment_column=TEXTBLOB_POLARITY_COLUMN, 
                 top_n=10,
                 save_dir=PLOT_DIR
             )
        
        if YEAR_COLUMN in df_analyzed.columns:
             # Ensure year column is numeric before plotting trends
             if pd.api.types.is_numeric_dtype(df_analyzed[YEAR_COLUMN]):
                  visualization.plot_sentiment_trends_over_time(
                      df_analyzed,
                      year_column=YEAR_COLUMN,
                      sentiment_column=TEXTBLOB_POLARITY_COLUMN, # <--- CHANGE HERE
                      save_dir=PLOT_DIR
                  )
             else:
                 print(f"Warning: Year column '{YEAR_COLUMN}' is not numeric. Skipping sentiment trends plot.")

        else:
            print(f"Warning: Year column '{YEAR_COLUMN}' not found. Skipping sentiment trends plot.")
    else:
        print(f"Warning: Sentiment column '{TEXTBLOB_SENTIMENT_COLUMN}' not found. Skipping all visualizations.")


    # 7. Save Processed Data
    utils.save_dataframe(df_analyzed, PROCESSED_DATA_FILE)

    print("\nLyrics Analysis Pipeline Finished Successfully!") 

if __name__ == "__main__":
    # Need to import pandas if using it directly in main, e.g., for the year check
    try:
        import pandas as pd
    except ImportError:
        print("Pandas library not found. Please install it: pip install pandas")
        pd = None # Set to None if import fails

    run_analysis_pipeline()
