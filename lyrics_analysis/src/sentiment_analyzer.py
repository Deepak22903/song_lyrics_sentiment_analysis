import pandas as pd
from textblob import TextBlob
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Uncomment if using VADER
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .preprocessing import preprocess_text # Use relative import if in same package

# --- Lexicon-Based Analysis (TextBlob) ---
def analyze_sentiment_textblob(text):
    """
    Determine sentiment using TextBlob.

    Parameters:
    -----------
    text : str
        Input text.

    Returns:
    --------
    dict
        Dictionary containing sentiment label, polarity, and subjectivity.
    """
    if not isinstance(text, str):
        return {'sentiment': 'Neutral', 'polarity': 0.0, 'subjectivity': 0.0}

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Define thresholds for sentiment categorization
    if polarity > 0.05:
        sentiment = 'Positive'
    elif polarity < -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return {
        'sentiment': sentiment,
        'polarity': polarity,
        'subjectivity': subjectivity
    }

# Define the function to calculate sentiment scores
def get_sentiment_scores(text):
    """Calculates polarity and subjectivity, handling non-strings."""
    if not isinstance(text, str):
        return (0.0, 0.0) # Neutral for non-strings/NaNs
    try:
        analysis = TextBlob(text)
        return (analysis.sentiment.polarity, analysis.sentiment.subjectivity)
    except Exception as e:
        # Log or print the error for debugging if needed
        # print(f"Error processing text snippet: {str(text)[:50]}... Error: {e}")
        return (0.0, 0.0) # Neutral on error

# Define the function to categorize polarity
def categorize_sentiment(polarity, threshold=0.05):
    """Categorizes polarity score into positive, negative, or neutral."""
    if polarity > threshold:
        return 'positive'
    elif polarity < -threshold:
        return 'negative'
    else:
        return 'neutral'

# --- MODIFY THIS FUNCTION ---
def apply_textblob_sentiment(df, text_column, sentiment_col_name='sentiment_tb'): # <-- Add parameter with default
    """
    Applies TextBlob sentiment analysis to a DataFrame column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing the text to analyze.
        sentiment_col_name (str): The base name for the output sentiment columns.
                                   Defaults to 'sentiment_tb'.

    Returns:
        pd.DataFrame: The DataFrame with added sentiment columns
                      (e.g., 'sentiment_tb_polarity', 'sentiment_tb_subjectivity', 'sentiment_tb').
                      Returns None if the text column doesn't exist or an error occurs.
    """
    if text_column not in df.columns:
        print(f"Error: Text column '{text_column}' not found in DataFrame.")
        return None
    if df[text_column].isnull().all():
         print(f"Warning: Text column '{text_column}' contains only null values. Adding empty sentiment columns.")
         df[f'{sentiment_col_name}_polarity'] = 0.0
         df[f'{sentiment_col_name}_subjectivity'] = 0.0
         df[sentiment_col_name] = 'neutral'
         return df

    print(f"Applying TextBlob sentiment analysis to '{text_column}' column...")
    try:
        # Calculate polarity and subjectivity
        sentiment_scores = df[text_column].apply(get_sentiment_scores)

        # Create the polarity and subjectivity columns using the provided name
        polarity_col = f'{sentiment_col_name}_polarity'
        subjectivity_col = f'{sentiment_col_name}_subjectivity'
        df[polarity_col] = sentiment_scores.apply(lambda x: x[0])
        df[subjectivity_col] = sentiment_scores.apply(lambda x: x[1])

        # Create the categorical sentiment column using the base name
        df[sentiment_col_name] = df[polarity_col].apply(categorize_sentiment)

        print(f"Sentiment analysis complete. Added '{polarity_col}', '{subjectivity_col}', and '{sentiment_col_name}' columns.")

    except Exception as e:
        print(f"An error occurred during TextBlob sentiment analysis: {e}")
        # Depending on desired behavior, you might return df or None
        return None # Return None on failure to indicate an issue

    return df

def train_ml_sentiment_model(df, text_column='processed_lyrics', label_column='sentiment_label'):
    """
    Train a simple ML model (e.g., Naive Bayes) for sentiment analysis.
    THIS IS A PLACEHOLDER - Requires a labeled dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with processed text and sentiment labels.
    text_column : str
        Column with processed text features.
    label_column : str
        Column with target sentiment labels (e.g., 'Positive', 'Negative', 'Neutral').

    Returns:
    --------
    tuple
        Trained model, vectorizer, and evaluation report (or None if fails).
    """
    if df is None or text_column not in df.columns or label_column not in df.columns:
        print(f"Error: DataFrame is None or required columns ('{text_column}', '{label_column}') not found.")
        return None, None, None

    print(f"\n--- Training ML Sentiment Model (Placeholder Example) ---")
    print(f"Using text from '{text_column}' and labels from '{label_column}'.")
    print("WARNING: This requires a reliable labeled dataset for meaningful results.")

    # Check if there are enough labels
    if df[label_column].isnull().sum() > 0:
        print(f"Warning: Found missing values in label column '{label_column}'. Dropping them.")
        df = df.dropna(subset=[label_column])

    if len(df) < 100 or len(df[label_column].unique()) < 2:
          print("Error: Not enough data or distinct labels to train a model.")
          return None, None, None

    X = df[text_column]
    y = df[label_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Vectorize text data using TF-IDF
    print("Vectorizing text data using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000) # Limit features
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a simple model (e.g., Multinomial Naive Bayes)
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    # Alternative: model = LinearSVC(dual=False) # Often performs well on text
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    print("Evaluating model on test set...")
    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    print("--- ML Model Training Complete ---")
    return model, vectorizer, report

def predict_ml_sentiment(texts, model, vectorizer):
    """
    Predict sentiment using a trained ML model.

    Parameters:
    -----------
    texts : list or pd.Series
        List or Series of texts to predict sentiment for.
    model : trained sklearn model
        The trained classifier.
    vectorizer : fitted sklearn vectorizer
        The TF-IDF vectorizer used during training.

    Returns:
    --------
    np.array
        Array of predicted sentiment labels.
    """
    if model is None or vectorizer is None:
        print("Error: Model or vectorizer is not available for prediction.")
        return None
    print("Predicting sentiment using ML model...")
    texts_tfidf = vectorizer.transform(texts)
    predictions = model.predict(texts_tfidf)
    print("Prediction complete.")
    return predictions


# --- NEW FUNCTION for single text analysis ---
def analyze_single_text_sentiment(text: str) -> dict:
    """
    Preprocesses and analyzes sentiment of a single text string using TextBlob.

    Args:
        text (str): The input text (e.g., lyrics).

    Returns:
        dict: Dictionary containing 'polarity', 'subjectivity', and 'sentiment' category.
              Returns default neutral values if input is invalid.
    """
    if not text or not isinstance(text, str):
        return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment': 'neutral'}

    processed_text = preprocess_text(text) # Preprocess the input
    if not processed_text:
        return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment': 'neutral'}

    try:
        analysis = TextBlob(processed_text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity

        # Same categorization logic as before
        threshold = 0.05
        if polarity > threshold:
            sentiment_category = 'positive'
        elif polarity < -threshold:
            sentiment_category = 'negative'
        else:
            sentiment_category = 'neutral'

        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment_category
        }
    except Exception as e:
        print(f"Error analyzing single text: {e}")
        return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment': 'neutral'}
