# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# --- Add the calculation line inside this function ---
def plot_sentiment_distribution(df, sentiment_column, save_dir):
    """
    Generates and saves a bar plot showing the distribution of sentiment categories.

    Args:
        df (pd.DataFrame): DataFrame containing the sentiment data.
        sentiment_column (str): Name of the column with categorical sentiment values.
        save_dir (str): Directory path to save the plot.
    """
    if sentiment_column not in df.columns:
        print(f"Error: Sentiment column '{sentiment_column}' not found in DataFrame. Skipping distribution plot.")
        return
    if df[sentiment_column].isnull().all():
         print(f"Warning: Sentiment column '{sentiment_column}' contains only null values. Skipping distribution plot.")
         return

    try:
        # --- FIX: Calculate sentiment_counts BEFORE using it ---
        sentiment_counts = df[sentiment_column].value_counts()
        # -------------------------------------------------------

        if sentiment_counts.empty:
            print(f"No data found in '{sentiment_column}' for distribution plot. Skipping.")
            return

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{sentiment_column}_distribution.png')

        plt.figure(figsize=(8, 5))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, hue=sentiment_counts.index, palette='viridis', legend=False)
        plt.title(f'Distribution of Sentiment ({sentiment_column})')
        plt.xlabel('Sentiment Category')
        plt.ylabel('Number of Songs')
        plt.xticks(rotation=0) # Adjust rotation if needed for category names
        plt.tight_layout()

        plt.savefig(save_path)
        print(f"Saved sentiment distribution plot to: {save_path}")
        plt.close() # Close the plot figure to free memory

    except Exception as e:
        print(f"Error generating sentiment distribution plot: {e}")
        # Optionally close plot if it was created before error
        plt.close()


# --- Other visualization functions (plot_sentiment_by_category, etc.) ---

def plot_sentiment_by_category(df, category_column, sentiment_column, top_n=10, save_dir=None):
    """
    Generates and saves a bar plot showing the average sentiment polarity
    for the top N categories.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        category_column (str): Name of the column with categories (e.g., genre, artist).
        sentiment_column (str): Name of the column with NUMERIC sentiment scores
                                 (e.g., 'sentiment_tb_polarity').
        top_n (int): Number of top categories to display.
        save_dir (str): Directory path to save the plot.
    """
    # --- Check required columns ---
    required_columns = [category_column, sentiment_column]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing required columns ({required_columns}) for sentiment by category plot. Skipping.")
        return
    if not pd.api.types.is_numeric_dtype(df[sentiment_column]):
        print(f"Error: Sentiment column '{sentiment_column}' must be numeric for averaging. Skipping plot for '{category_column}'.")
        # Attempt to find the associated numeric polarity column if the categorical one was passed
        polarity_col = f"{sentiment_column}_polarity"
        if polarity_col in df.columns and pd.api.types.is_numeric_dtype(df[polarity_col]):
             print(f"Using '{polarity_col}' for numeric sentiment scores instead.")
             sentiment_column = polarity_col # Use the numeric column
        else:
             # Try common patterns if TextBlob was used
             tb_polarity_col = "sentiment_tb_polarity"
             if sentiment_column == "sentiment_tb" and tb_polarity_col in df.columns and pd.api.types.is_numeric_dtype(df[tb_polarity_col]):
                 print(f"Using '{tb_polarity_col}' for numeric sentiment scores instead.")
                 sentiment_column = tb_polarity_col # Use the numeric column
             else:
                 return # Cannot proceed without a numeric sentiment column

    df_filtered = df.dropna(subset=[category_column, sentiment_column])
    if df_filtered.empty:
        print(f"No valid data after dropping NaNs for '{category_column}' and '{sentiment_column}'. Skipping plot.")
        return

    try:
        # Calculate average sentiment per category
        # Group by category and calculate mean sentiment and count
        sentiment_by_cat = df_filtered.groupby(category_column)[sentiment_column].agg(['mean', 'count'])

        # Filter out categories with very few entries (optional, adjust threshold)
        min_count_threshold = 5
        sentiment_by_cat = sentiment_by_cat[sentiment_by_cat['count'] >= min_count_threshold]

        if sentiment_by_cat.empty:
             print(f"No categories found with at least {min_count_threshold} entries. Skipping plot for '{category_column}'.")
             return

        # Get top N categories by count (or mean sentiment, depending on goal)
        # Here, we sort by count to show most frequent, then plot their mean sentiment
        top_categories = sentiment_by_cat.nlargest(top_n, 'count')
        # Alternatively, sort by mean sentiment:
        # top_categories = sentiment_by_cat.nlargest(top_n, 'mean') # Top N most positive
        # top_categories = sentiment_by_cat.nsmallest(top_n, 'mean') # Top N most negative (adjust plot title)

        if top_categories.empty:
            print(f"No top {top_n} categories found for '{category_column}'. Skipping plot.")
            return

        # Plotting
        plt.figure(figsize=(12, 7))
        # Use the index (category names) for x-axis, and 'mean' column for y-axis
        sns.barplot(x=top_categories.index, y=top_categories['mean'], hue=top_categories.index, palette='coolwarm', legend=False)
        plt.title(f'Average Sentiment Polarity for Top {top_n} {category_column.capitalize()}s (by count)')
        plt.xlabel(category_column.capitalize())
        plt.ylabel('Average Sentiment Polarity')
        plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
        plt.tight_layout()

        # Ensure save directory exists
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'sentiment_by_{category_column}_top_{top_n}.png')
            plt.savefig(save_path)
            print(f"Saved sentiment by {category_column} plot to: {save_path}")

        plt.close() # Close the plot figure

    except Exception as e:
        print(f"Error generating sentiment by {category_column} plot: {e}")
        plt.close()


def plot_sentiment_trends_over_time(df, year_column, sentiment_column, save_dir=None):
    """
    Generates and saves a line plot showing average sentiment polarity over time (years).

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        year_column (str): Name of the column containing the year.
        sentiment_column (str): Name of the column with NUMERIC sentiment scores
                                 (e.g., 'sentiment_tb_polarity').
        save_dir (str): Directory path to save the plot.
    """
     # --- Check required columns ---
    required_columns = [year_column, sentiment_column]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing required columns ({required_columns}) for sentiment trends plot. Skipping.")
        return

    # --- Ensure year is numeric and sentiment is numeric ---
    try:
        # Attempt conversion if year isn't numeric, coercing errors to NaN
        if not pd.api.types.is_numeric_dtype(df[year_column]):
             print(f"Warning: Year column '{year_column}' is not numeric. Attempting conversion...")
             df[year_column] = pd.to_numeric(df[year_column], errors='coerce')

        # Check sentiment column type after potential fix in plot_sentiment_by_category logic
        if not pd.api.types.is_numeric_dtype(df[sentiment_column]):
             print(f"Error: Sentiment column '{sentiment_column}' must be numeric for averaging trends. Skipping plot.")
             # Attempt to find the associated numeric polarity column
             polarity_col = f"{sentiment_column}_polarity"
             if polarity_col in df.columns and pd.api.types.is_numeric_dtype(df[polarity_col]):
                 print(f"Using '{polarity_col}' for numeric sentiment scores instead.")
                 sentiment_column = polarity_col # Use the numeric column
             else:
                 # Try common patterns if TextBlob was used
                 tb_polarity_col = "sentiment_tb_polarity"
                 if sentiment_column == "sentiment_tb" and tb_polarity_col in df.columns and pd.api.types.is_numeric_dtype(df[tb_polarity_col]):
                     print(f"Using '{tb_polarity_col}' for numeric sentiment scores instead.")
                     sentiment_column = tb_polarity_col # Use the numeric column
                 else:
                     return # Cannot proceed

    except Exception as e:
         print(f"Error processing columns for trend plot: {e}. Skipping.")
         return


    # Drop rows where year or sentiment is missing AFTER potential conversion
    df_filtered = df.dropna(subset=[year_column, sentiment_column])

    # Optional: Filter out unrealistic years if necessary
    current_year = pd.Timestamp.now().year
    df_filtered = df_filtered[df_filtered[year_column].between(1900, current_year)] # Adjust range as needed

    if df_filtered.empty:
        print(f"No valid data after filtering/cleaning for sentiment trends plot. Skipping.")
        return

    try:
        # Calculate average sentiment per year
        sentiment_over_time = df_filtered.groupby(year_column)[sentiment_column].mean()

        if sentiment_over_time.empty:
            print(f"No yearly sentiment averages could be calculated. Skipping trends plot.")
            return

        # Plotting
        plt.figure(figsize=(12, 6))
        sentiment_over_time.plot(kind='line', marker='.', linestyle='-')
        plt.title(f'Average Sentiment Polarity ({sentiment_column}) Over Time')
        plt.xlabel('Year')
        plt.ylabel('Average Sentiment Polarity')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Ensure save directory exists
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'sentiment_trends_over_time.png')
            plt.savefig(save_path)
            print(f"Saved sentiment trends plot to: {save_path}")

        plt.close() # Close the plot figure

    except Exception as e:
        print(f"Error generating sentiment trends plot: {e}")
        plt.close()

# Add any other visualization functions you might have here...
