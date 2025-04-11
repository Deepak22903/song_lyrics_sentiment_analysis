# Lyrics Sentiment Analysis Project

This project analyzes song lyrics to determine their sentiment (Positive, Negative, Neutral) using NLP techniques. It aims to understand emotional tones in music, identify trends, and potentially explore relationships between sentiment and features like genre, artist, or year.
The project provides an interactive web application built with Streamlit to perform sentiment analysis on song lyrics. It processes a dataset of lyrics, calculates sentiment polarity and subjectivity using TextBlob, generates various insightful visualizations (including distributions, trends, word clouds), and allows users to analyze custom text snippets on-the-fly.

**Team:**

- DEEPAK SHITOLE (642303019)
- SOMESHWAR GIRAM (642303009)
- OM BHUTKAR (642303005)

## Features

- Loads and preprocesses song lyrics data from a CSV file.
- Filters lyrics by language (default: English).
- Performs sentiment analysis using TextBlob to determine polarity (positive/negative/neutral) and subjectivity.
- Displays overall sentiment distribution across the dataset.
- Shows average sentiment polarity for the top N artists and genres.
- Visualizes sentiment polarity trends over the years present in the data.
- Generates an overall word cloud from the most frequent terms in the processed lyrics.
- Lists the top 10 most positive and most negative songs based on polarity score.
- Analyzes and displays the distribution of subjectivity scores using an interactive histogram.
- Provides an easy-to-use web dashboard powered by Streamlit.
- Allows users to input custom text (e.g., lyrics) for immediate sentiment analysis.
- Option to download the processed data (including sentiment scores) as a CSV file.

## Screenshot

_(It's highly recommended to add a screenshot of your running Streamlit app here!)_

```

[Insert Screenshot of the Streamlit App Here]

```

## Tech Stack

- **Backend:** Python 3.x
- **Data Handling:** Pandas
- **NLP & Sentiment:** NLTK, TextBlob
- **Web Framework:** Streamlit
- **Visualization:** Matplotlib, Seaborn, Plotly, WordCloud
- **(Potential):** Scikit-learn (if N-grams or other ML features are added)

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Deepak22903/song_lyrics_sentiment_analysis.git
    cd song_lyrics_sentiment_analysis
    ```

2.  **Create and activate a virtual environment (Recommended):**

    ```bash
    # Create venv
    python -m venv venv

    # Activate venv
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download necessary NLTK data:**
    Run the Python interpreter (`python`) and execute the following commands:

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    quit()
    ```

    _(Note: The application might attempt to download these automatically via `src/preprocessing.py`, but manual download ensures they are present.)_

5.  **Prepare Data:**
    - Dataset link -> [click here ](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information)
    - Place your lyrics dataset (e.g., `song_lyrics.csv` or the trimmed version like `song_lyrics_trimmed.csv`) inside the `data/` directory.
    - Ensure the filename matches the `RAW_DATA_FILE` constant defined in your configuration (likely at the top of `main.py`). Adjust the constant if your filename differs.
    - Other configurations like column names (`ARTIST_COLUMN`, `LYRICS_COLUMN`, etc.) are also typically set in `main.py`.

## Usage

1.  Make sure your virtual environment is activated.
2.  Navigate to the project's root directory in your terminal.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  The application interface will automatically open in your default web browser.
5.  Use the **"Run Full Analysis Pipeline"** button in the sidebar to process the dataset and view the main results.
6.  Use the **"Analyze Your Own Lyrics"** section to get sentiment scores for custom text input.

## Project Structure

````

.
├── data/
│ ├── song_lyrics.csv # Example raw dataset (or trimmed version)
│ └── processed_lyrics.csv # Output data with sentiment scores (after run)
├── plots/ # Output plots are saved here (after run)
├── src/
│ ├── **init**.py
│ ├── data_loader.py # Handles data loading
│ ├── preprocessing.py # Text preprocessing functions
│ ├── sentiment_analyzer.py# Sentiment analysis functions
│ ├── visualization.py # Plotting functions (static or Plotly figures)
│ └── utils.py # Utility functions (e.g., saving data)
├── venv/ # Virtual environment directory (if created)
├── app.py # Main Streamlit application script
├── main.py # Contains backend pipeline logic (run_full_analysis) & config
├── requirements.txt # Project dependencies
└── README.md # This file

```

## Future Enhancements (Ideas)

* Integrate VADER sentiment analysis for comparison or specific use cases.
* Implement Topic Modeling (e.g., LDA) to discover themes in lyrics.
* Add N-gram analysis to find common phrases.
* Allow user selection/filtering by specific artists, genres, or year ranges before running analysis.
* Add more interactive Plotly charts.
* Further optimize performance using Streamlit's caching (`@st.cache_data`, `@st.cache_resource`).
* Deploy the application online (e.g., Streamlit Community Cloud, Heroku, etc.).

````
