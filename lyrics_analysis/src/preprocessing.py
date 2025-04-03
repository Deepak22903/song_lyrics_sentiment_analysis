import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import ssl # Added for potential SSL certificate issues during download

def download_nltk_data():
    """
    Downloads necessary NLTK datasets if they are not found.
    Includes basic SSL context handling for potential download issues.
    """
    # --- Attempt to bypass SSL verification errors (use with caution) ---
    # Sometimes NLTK downloads fail due to SSL certificate issues.
    # Uncomment the following lines if you encounter SSL errors during download.
    # try:
    #     _create_unverified_https_context = ssl._create_unverified_context
    # except AttributeError:
    #     pass
    # else:
    #     ssl._create_default_https_context = _create_unverified_https_context
    # --------------------------------------------------------------------

    resources = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet'  # This is the one causing the error
    }
    all_downloaded = True
    for resource_path, download_name in resources.items():
        try:
            nltk.data.find(resource_path)
            # print(f"NLTK resource '{download_name}' found.") # Optional debug message
        except LookupError: # Correct exception to catch here
            print(f"NLTK resource '{download_name}' not found. Attempting download...")
            try:
                nltk.download(download_name, quiet=False) # Set quiet=False to see download progress/errors
                # Verify download worked
                nltk.data.find(resource_path)
                print(f"Successfully downloaded and verified '{download_name}'.")
            except Exception as e:
                print(f"ERROR: Failed to download NLTK resource '{download_name}'. Error: {e}")
                print("\nPlease try downloading manually:")
                print("1. Open a Python interpreter (type 'python' in your terminal).")
                print("2. Run the following commands:")
                print(f">>> import nltk")
                print(f">>> nltk.download('{download_name}')")
                all_downloaded = False
                # Optionally, raise an error to stop the script if a resource is critical
                # raise RuntimeError(f"Failed to download essential NLTK resource: {download_name}") from e

    if all_downloaded:
        print("All required NLTK resources are available.")
    else:
        print("One or more NLTK resources failed to download automatically. Please follow manual instructions above.")
        # Exit if downloads failed and are critical
        # exit(1)


# --- Text Cleaning Functions ---
def clean_text(text):
    """
    Comprehensive text cleaning: lowercase, remove brackets, special chars, extra spaces.

    Parameters:
    -----------
    text : str
        Input text.

    Returns:
    --------
    str
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""  # Return empty string for non-string inputs

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove content in brackets (e.g., [Chorus])
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with single space
    return text

def tokenize_text(text):
    """
    Tokenize text into words.

    Parameters:
    -----------
    text : str
        Input text.

    Returns:
    --------
    list
        List of word tokens.
    """
    if not isinstance(text, str):
        return []
    return word_tokenize(text)

def remove_stopwords_from_tokens(tokens):
    """
    Remove English stopwords from a list of tokens.

    Parameters:
    -----------
    tokens : list
        List of word tokens.

    Returns:
    --------
    list
        List of tokens with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens):
    """
    Lemmatize a list of tokens using WordNetLemmatizer.

    Parameters:
    -----------
    tokens : list
        List of word tokens.

    Returns:
    --------
    list
        List of lemmatized tokens.
    """
    lemmatizer = WordNetLemmatizer()
    # Note: Lemmatization can be improved by providing Part-of-Speech tags,
    # but for simplicity, we'll lemmatize assuming nouns by default.
    return [lemmatizer.lemmatize(word) for word in tokens]

# --- Main Preprocessing Pipeline ---
def preprocess_lyrics_text(text):
    """
    Apply the full preprocessing pipeline to a single text string.
    Includes cleaning, tokenization, stopword removal, and lemmatization.

    Parameters:
    -----------
    text : str
        Raw lyrics text.

    Returns:
    --------
    str
        Fully preprocessed text (tokens joined back into a string).
    """
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens_no_stopwords = remove_stopwords_from_tokens(tokens)
    lemmatized_tokens = lemmatize_tokens(tokens_no_stopwords)
    return ' '.join(lemmatized_tokens) # Return processed text as a single string

def apply_preprocessing_to_column(df, text_column='lyrics', new_column='processed_lyrics'):
    """
    Apply the preprocessing pipeline to a specific column in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame.
    text_column : str
        Name of the column containing the text to preprocess.
    new_column : str
        Name of the new column to store the processed text.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with the new preprocessed column.
    """
    if df is None or text_column not in df.columns:
        print(f"Error: DataFrame is None or '{text_column}' column not found.")
        return df

    print(f"Applying preprocessing to '{text_column}' column...")
    # Ensure NLTK data is available before applying functions that need it
    download_nltk_data()
    df[new_column] = df[text_column].apply(preprocess_lyrics_text)
    print(f"Preprocessing complete. Added '{new_column}' column.")
    return df
