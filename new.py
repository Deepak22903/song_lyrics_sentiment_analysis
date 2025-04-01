import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Step 1: Load the dataset
# For this example, we're creating a simulated dataset
# In a real project, you would load your actual dataset

def create_sample_dataset(n=1000):
    """Create a sample dataset for demonstration purposes"""
    # Creating some sample genres
    genres = ['Pop', 'Rock', 'Hip Hop', 'R&B', 'Country', 'EDM']
    # Creating some sample artists
    artists = ['Artist A', 'Artist B', 'Artist C', 'Artist D', 'Artist E']
    # Creating some sample years
    years = list(range(2000, 2025))
    
    # Sample lyrics snippets (simulated)
    positive_snippets = [
        "love happiness joy beautiful wonderful amazing great",
        "happy excited thrilled perfect delighted fantastic awesome",
        "dream bright sunshine smile dancing celebration success"
    ]
    
    negative_snippets = [
        "sad lonely heartbreak pain tears suffering sorrow",
        "broken angry hurt dark depressed miserable hate",
        "lost empty abandoned struggle failure fear regret"
    ]
    
    neutral_snippets = [
        "walking talking thinking looking waiting remember",
        "time day night morning evening world people",
        "see hear know understand find change try"
    ]
    
    # Generate random data
    data = {
        'title': [f"Song {i}" for i in range(1, n+1)],
        'artist': np.random.choice(artists, n),
        'genre': np.random.choice(genres, n),
        'year': np.random.choice(years, n),
        'lyrics': []
    }
    
    for i in range(n):
        sentiment_choice = np.random.choice(['positive', 'negative', 'neutral'])
        if sentiment_choice == 'positive':
            lyrics = np.random.choice(positive_snippets) + " " + np.random.choice(neutral_snippets)
        elif sentiment_choice == 'negative':
            lyrics = np.random.choice(negative_snippets) + " " + np.random.choice(neutral_snippets)
        else:
            lyrics = np.random.choice(neutral_snippets) + " " + np.random.choice(neutral_snippets)
        
        # Add some randomness to the lyrics length
        repeat = np.random.randint(1, 5)
        data['lyrics'].append(lyrics * repeat)
    
    return pd.DataFrame(data)

# Create or load dataset
df = create_sample_dataset(1000)
print(f"Dataset shape: {df.shape}")
print(df.head())

# Step 2: Preprocess the lyrics
def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, numbers, punctuations
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into text
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Apply preprocessing to lyrics
df['cleaned_lyrics'] = df['lyrics'].apply(preprocess_text)
print("Sample of cleaned lyrics:")
print(df['cleaned_lyrics'].head())

# Step 3: Sentiment Analysis

# Method 1: Using VADER for lexicon-based sentiment analysis
sid = SentimentIntensityAnalyzer()

# Function to get sentiment scores
def get_vader_sentiment(text):
    """Get sentiment scores using VADER"""
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

# Apply VADER sentiment analysis
df['vader_scores'] = df['cleaned_lyrics'].apply(get_vader_sentiment)

# Extract compound score
df['vader_compound'] = df['vader_scores'].apply(lambda x: x['compound'])

# Classify sentiment based on compound score
def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['vader_sentiment'] = df['vader_compound'].apply(classify_sentiment)

# Method 2: Using TextBlob
def get_textblob_sentiment(text):
    """Get sentiment polarity using TextBlob"""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['textblob_polarity'] = df['cleaned_lyrics'].apply(get_textblob_sentiment)

# Classify TextBlob sentiment
df['textblob_sentiment'] = df['textblob_polarity'].apply(classify_sentiment)

# Step 4: Machine Learning Approach

# Create sentiment labels for machine learning (using VADER as ground truth for this example)
df['sentiment_label'] = df['vader_sentiment']

# Encode sentiment labels
sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['sentiment_code'] = df['sentiment_label'].map(sentiment_mapping)

# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['cleaned_lyrics'])
y = df['sentiment_code']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Step 5: Visualizations and Analysis

# Sentiment distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
vader_counts = df['vader_sentiment'].value_counts()
plt.pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('VADER Sentiment Distribution')

plt.subplot(1, 2, 2)
textblob_counts = df['textblob_sentiment'].value_counts()
plt.pie(textblob_counts, labels=textblob_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('TextBlob Sentiment Distribution')

plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.close()

# Sentiment across genres
plt.figure(figsize=(12, 6))
genre_sentiment = df.groupby('genre')['vader_sentiment'].value_counts(normalize=True).unstack() * 100
genre_sentiment.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Sentiment Distribution Across Genres')
plt.xlabel('Genre')
plt.ylabel('Percentage')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig('genre_sentiment.png')
plt.close()

# Sentiment trends over years
yearly_sentiment = df.groupby('year')['vader_sentiment'].value_counts(normalize=True).unstack() * 100
plt.figure(figsize=(14, 7))
yearly_sentiment.plot(kind='line', marker='o')
plt.title('Sentiment Trends Over Years')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Sentiment')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('yearly_trends.png')
plt.close()

# Word clouds for different sentiments
def generate_wordcloud(text, title, filename):
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, contour_width=3).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Create word clouds for each sentiment
positive_text = ' '.join(df[df['vader_sentiment'] == 'Positive']['cleaned_lyrics'])
negative_text = ' '.join(df[df['vader_sentiment'] == 'Negative']['cleaned_lyrics'])
neutral_text = ' '.join(df[df['vader_sentiment'] == 'Neutral']['cleaned_lyrics'])

generate_wordcloud(positive_text, 'Positive Sentiment Words', 'positive_wordcloud.png')
generate_wordcloud(negative_text, 'Negative Sentiment Words', 'negative_wordcloud.png')
generate_wordcloud(neutral_text, 'Neutral Sentiment Words', 'neutral_wordcloud.png')

# Artist sentiment analysis
artist_sentiment = df.groupby('artist')['vader_sentiment'].value_counts(normalize=True).unstack() * 100
plt.figure(figsize=(12, 6))
artist_sentiment.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Sentiment Distribution Across Artists')
plt.xlabel('Artist')
plt.ylabel('Percentage')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig('artist_sentiment.png')
plt.close()

# Function to predict sentiment for new lyrics
def predict_sentiment(lyrics, method='all'):
    """
    Predict sentiment for new lyrics
    
    Parameters:
    -----------
    lyrics : str
        The lyrics to analyze
    method : str
        The method to use ('vader', 'textblob', 'ml', or 'all')
        
    Returns:
    --------
    dict
        Dictionary with sentiment predictions
    """
    # Preprocess the lyrics
    cleaned_lyrics = preprocess_text(lyrics)
    
    results = {}
    
    if method in ['vader', 'all']:
        # VADER sentiment
        vader_scores = sid.polarity_scores(cleaned_lyrics)
        vader_compound = vader_scores['compound']
        vader_sentiment = classify_sentiment(vader_compound)
        
        results['vader'] = {
            'scores': vader_scores,
            'compound': vader_compound,
            'sentiment': vader_sentiment
        }
    
    if method in ['textblob', 'all']:
        # TextBlob sentiment
        textblob_polarity = get_textblob_sentiment(cleaned_lyrics)
        textblob_sentiment = classify_sentiment(textblob_polarity)
        
        results['textblob'] = {
            'polarity': textblob_polarity,
            'sentiment': textblob_sentiment
        }
    
    if method in ['ml', 'all']:
        # Machine Learning prediction
        tfidf_vector = tfidf_vectorizer.transform([cleaned_lyrics])
        ml_prediction = nb_classifier.predict(tfidf_vector)[0]
        ml_sentiment = list(sentiment_mapping.keys())[list(sentiment_mapping.values()).index(ml_prediction)]
        
        results['ml'] = {
            'prediction': ml_prediction,
            'sentiment': ml_sentiment
        }
    
    return results

# Example usage
sample_lyrics = "Today is a beautiful day and I feel amazing about everything in life"
results = predict_sentiment(sample_lyrics)
print("\nSample Lyrics Sentiment Analysis:")
print(f"Lyrics: {sample_lyrics}")
print(f"VADER: {results['vader']['sentiment']} (Compound: {results['vader']['compound']:.3f})")
print(f"TextBlob: {results['textblob']['sentiment']} (Polarity: {results['textblob']['polarity']:.3f})")
print(f"ML Model: {results['ml']['sentiment']}")

# Save the trained model and preprocessor for future use
import joblib
joblib.dump(nb_classifier, 'sentiment_classifier.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

print("\nAnalysis complete! Visualizations saved as PNG files.")
