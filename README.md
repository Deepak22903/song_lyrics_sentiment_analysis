# Lyrics Sentiment Analysis Project

This project analyzes song lyrics to determine their sentiment (Positive, Negative, Neutral) using NLP techniques. It aims to understand emotional tones in music, identify trends, and potentially explore relationships between sentiment and features like genre, artist, or year.

**Team:**

- DEEPAK SHITOLE (642303019)
- SOMESHWAR GIRAM (642303009)
- OM BHUTKAR (642303005)

**(Project Deadline/Presentation: March 2025)**

## Project Structure

yrics_analysis/
├── .gitignore
├── README.md
├── requirements.txt
├── config.py # Optional: configuration settings
├── data/
│ └── song_lyrics.csv # Needs to be added by user
│ └── processed_lyrics.csv # Generated output
├── notebooks/
│ └── preprocessing_exploration.ipynb # Original notebook
├── src/
│ ├── init.py
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── sentiment_analyzer.py
│ ├── visualization.py
│ └── utils.py
└── main.py # Main execution script

## Features

- Loads lyrics data from a CSV file.
- Filters lyrics by language (default: English).
- Preprocesses lyrics: cleaning (lowercase, remove punctuation/brackets), tokenization, stopword removal, lemmatization.
- Performs sentiment analysis using lexicon-based methods (TextBlob included).
- Includes placeholders and structure for adding Machine Learning models (Naive Bayes, SVM) - **Requires a labeled dataset for training**.
- Visualizes results:
  - Overall sentiment distribution (pie chart).
  - Sentiment distribution by category (e.g., Top Genres, Top Artists - stacked bar charts).
  - Sentiment trends over time (line chart).
- Saves the processed data with sentiment scores.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd lyrics_analysis
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK data (if not done automatically by the script):**
    Run Python and enter:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    exit()
    ```
5.  **Place your dataset:**
    - Put your lyrics CSV file (e.g., `song_lyrics.csv`) inside the `data/` directory.
    - Make sure the file name matches the `RAW_DATA_FILE` variable in `main.py` or update the variable. Ensure the CSV contains columns for lyrics, language, genre (tag), artist, and year (adjust column names in `main.py` if different).

## How to Run

Execute the main pipeline script from the project's root directory:

```bash
python main.py
```
