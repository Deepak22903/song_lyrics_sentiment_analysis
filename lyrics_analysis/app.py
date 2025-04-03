import streamlit as st
import pandas as pd
import os
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image # Needed if WordCloud generates an image object directly

# --- Backend Imports ---
# Adjust the import source ('main' or 'pipeline') based on where you defined the function
try:
    from main import (
        run_full_analysis,           # The refactored analysis function
        PLOT_DIR,                    # Directory where plots are saved by the backend
        PROCESSED_DATA_FILE,         # Path to the saved processed CSV
        TEXTBLOB_POLARITY_COLUMN,    # Name of the polarity column
        TEXTBLOB_SUBJECTIVITY_COLUMN,# Name of the subjectivity column
        PROCESSED_LYRICS_COLUMN,     # Name of the processed lyrics column
        ARTIST_COLUMN,               # Name of the artist column
        YEAR_COLUMN,                 # Name of the year column
        # Add other constants if needed, e.g., GENRE_COLUMN, TITLE_COLUMN
    )
except ImportError:
    st.error("Could not import backend functions from 'main.py'. Make sure it's in the correct path.")
    st.stop() # Stop the app if backend cannot be imported

try:
    # Import the single text analyzer function
    from src.sentiment_analyzer import analyze_single_text_sentiment
    # Optional: Import preprocessing if needed directly (e.g., for NLTK download)
    # from src.preprocessing import download_nltk_data
except ImportError:
     st.error("Could not import functions from 'src' directory. Make sure it's structured correctly.")
     st.stop()


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Lyrics Analysis Dashboard",
    page_icon="🎶",
    layout="wide" # Use wide layout for more space
)

# --- App Title ---
st.title("🎶 Lyrics Sentiment Analysis Dashboard 🎤")
st.markdown("Run the analysis pipeline on the song lyrics dataset and view insights.")
st.markdown(f"*(Code last updated around April 3, 2025 - Pune, India)*") # Acknowledge context date

# --- Session State Initialization ---
# Used to store results across reruns triggered by widgets
if 'analysis_run_complete' not in st.session_state:
    st.session_state.analysis_run_complete = False
    st.session_state.final_df = None
    st.session_state.plot_paths = None # Stores paths to saved plots


# --- Custom Input Section ---
st.markdown("---")
st.header("✨ Analyze Your Own Lyrics")
st.markdown("Paste any text below to get a quick sentiment analysis using TextBlob.")
custom_text = st.text_area("Paste lyrics or text here:", height=150, key="custom_text_input")
analyze_custom_button = st.button("Analyze Custom Text", key="analyze_custom_button")

if analyze_custom_button and custom_text:
    with st.spinner("Analyzing..."):
        # Optional: Ensure NLTK data needed by preprocessing is available
        # try:
        #     download_nltk_data()
        # except Exception as e:
        #     st.warning(f"Could not verify NLTK data: {e}")

        sentiment_result = analyze_single_text_sentiment(custom_text)

    st.subheader("Custom Text Analysis Result:")
    col1_custom, col2_custom, col3_custom = st.columns(3)
    col1_custom.metric("Sentiment Category", sentiment_result['sentiment'].capitalize())
    col2_custom.metric("Polarity Score", f"{sentiment_result['polarity']:.3f}")
    col3_custom.metric("Subjectivity Score", f"{sentiment_result['subjectivity']:.3f}")

elif analyze_custom_button and not custom_text:
    st.warning("Please paste some text into the text area first.")


# --- Separator ---
st.markdown("---")


# --- Sidebar Controls ---
st.sidebar.header("Controls")
st.sidebar.markdown("Run the main analysis on the configured dataset.")
run_button = st.sidebar.button("🚀 Run Full Analysis Pipeline", key="run_pipeline_button")
st.sidebar.markdown("---") # Separator


# --- Main Pipeline Trigger Logic ---
if run_button:
    st.info("⏳ Analysis started... This might take a while depending on the data size.")
    with st.spinner('Running full pipeline: Loading, preprocessing, analyzing, plotting...'):
        # Call the backend function
        final_df, plot_paths = run_full_analysis()
        # Store results in session state
        st.session_state.final_df = final_df
        st.session_state.plot_paths = plot_paths
        st.session_state.analysis_run_complete = True # Mark analysis as run
        st.rerun() # Rerun the script to update the display based on session state


# --- Main Results Display Area ---
# This entire block only shows if the analysis has been run successfully
if st.session_state.analysis_run_complete:
    if st.session_state.final_df is not None and st.session_state.plot_paths is not None:
        final_df = st.session_state.final_df
        plot_paths = st.session_state.plot_paths

        st.success("✅ Analysis Completed Successfully!")
        st.markdown("---")

        # --- Deeper Insights Section ---
        st.header("💡 Deeper Insights")

        # Use columns for better layout of insights
        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:
            # 1. Word Cloud
            st.subheader("☁️ Overall Word Cloud")
            if PROCESSED_LYRICS_COLUMN in final_df.columns:
                try:
                    full_processed_text = ' '.join(final_df[PROCESSED_LYRICS_COLUMN].dropna())
                    if full_processed_text:
                        # Generate WordCloud object
                        wordcloud = WordCloud(width=600, height=300, # Adjusted size for column
                                              background_color='white',
                                              colormap='viridis', # Example colormap
                                              max_words=150,
                                              contour_width=1,
                                              contour_color='steelblue').generate(full_processed_text)

                        # Display using matplotlib
                        fig_wc, ax_wc = plt.subplots(figsize=(10,5)) # Adjust figure size if needed
                        ax_wc.imshow(wordcloud, interpolation='bilinear')
                        ax_wc.axis('off')
                        st.pyplot(fig_wc)
                    else:
                        st.warning("Not enough text data to generate a word cloud.")
                except Exception as e:
                    st.error(f"Could not generate word cloud: {e}")
            else:
                st.warning(f"Column '{PROCESSED_LYRICS_COLUMN}' not found for Word Cloud.")


            # 3. Subjectivity Analysis
            st.subheader("🧐 Subjectivity Analysis")
            if TEXTBLOB_SUBJECTIVITY_COLUMN in final_df.columns:
                 try:
                     avg_subjectivity = final_df[TEXTBLOB_SUBJECTIVITY_COLUMN].mean()
                     st.metric(label="Average Subjectivity", value=f"{avg_subjectivity:.3f}")

                     # Plotly histogram for subjectivity
                     fig_subj = px.histogram(final_df, x=TEXTBLOB_SUBJECTIVITY_COLUMN,
                                             title="Distribution of Subjectivity Scores",
                                             nbins=30, opacity=0.7,
                                             labels={TEXTBLOB_SUBJECTIVITY_COLUMN:"Subjectivity Score"})
                     fig_subj.update_layout(bargap=0.1)
                     st.plotly_chart(fig_subj, use_container_width=True)
                 except Exception as e:
                     st.error(f"Could not display Subjectivity Analysis: {e}")
            else:
                 st.warning(f"Subjectivity column '{TEXTBLOB_SUBJECTIVITY_COLUMN}' not found.")


        with insight_col2:
            # 2. Most Positive / Negative Songs
            st.subheader(" extremes Sentiment Extremes")
            # --- Define columns needed ---
            # Try to find a 'title' column, otherwise use fallbacks
            if 'title' in final_df.columns:
                title_col = 'title'
            elif len(final_df.columns) > 1:
                title_col = final_df.columns[1] # Guess second column might be title
            else:
                title_col = final_df.columns[0] # Fallback to first column

            required_cols_extreme = [ARTIST_COLUMN, YEAR_COLUMN, TEXTBLOB_POLARITY_COLUMN, title_col]

            if all(col in final_df.columns for col in required_cols_extreme):
                df_sorted = final_df.sort_values(by=TEXTBLOB_POLARITY_COLUMN, ascending=False)
                cols_to_show = [ARTIST_COLUMN, title_col, YEAR_COLUMN, TEXTBLOB_POLARITY_COLUMN]

                st.write("🌟 Most Positive Songs (Top 10):")
                st.dataframe(df_sorted[cols_to_show].head(10).reset_index(drop=True), height=385, use_container_width=True) # Adjust height

                st.write("⛈️ Most Negative Songs (Top 10):")
                # Need to re-sort the tail ascending to show most negative first
                st.dataframe(df_sorted[cols_to_show].tail(10).sort_values(by=TEXTBLOB_POLARITY_COLUMN, ascending=True).reset_index(drop=True), height=385, use_container_width=True) # Adjust height
            else:
                st.warning(f"Cannot display sentiment extremes. Missing required columns. Needed: {required_cols_extreme}, Found: {list(final_df.columns)}")


        # --- Visualization Results (Original Plots) ---
        st.markdown("---")
        st.header("📊 Visualization Results")
        st.markdown("Plots generated by the analysis pipeline.")

        plot_col1, plot_col2 = st.columns(2)
        plot_map = {
            'distribution': plot_col1,
            'genre': plot_col1,
            'artist': plot_col2,
            'trends': plot_col2
        }
        plot_titles = {
            'distribution': "Overall Sentiment Counts",
            'genre': "Avg. Sentiment Polarity by Top Genres",
            'artist': "Avg. Sentiment Polarity by Top Artists",
            'trends': "Avg. Sentiment Polarity Trend by Year"
        }

        displayed_plots = 0
        for plot_name, plot_path in plot_paths.items():
            if os.path.exists(plot_path):
                target_column = plot_map.get(plot_name, st) # Default to main area if name not in map
                with target_column:
                     st.image(plot_path, caption=plot_titles.get(plot_name, plot_name.capitalize()))
                     displayed_plots += 1
            else:
                st.warning(f"Plot file not found: {plot_path}")

        if displayed_plots == 0:
            st.warning("No plot files were found or generated.")


        # --- Processed Data Sample Section ---
        st.markdown("---")
        st.header("📄 Processed Data Sample")

        st.subheader("Overall Sentiment Metrics")
        if TEXTBLOB_POLARITY_COLUMN in final_df.columns:
             try:
                 avg_polarity = final_df[TEXTBLOB_POLARITY_COLUMN].mean()
                 st.metric(label="Average Polarity", value=f"{avg_polarity:.3f}")
             except Exception as e:
                 st.warning(f"Could not calculate average polarity: {e}")
        else:
             st.warning(f"Polarity column '{TEXTBLOB_POLARITY_COLUMN}' not found for metrics.")

        st.subheader(f"Top 5 Rows (Total Rows: {len(final_df)})")
        st.dataframe(final_df.head(), use_container_width=True)

        # Provide download link for the processed data
        st.subheader("💾 Download Processed Data")
        if os.path.exists(PROCESSED_DATA_FILE):
             try:
                 with open(PROCESSED_DATA_FILE, "rb") as fp:
                     st.download_button(
                         label="Download processed_lyrics.csv",
                         data=fp,
                         file_name="processed_lyrics.csv", # Name for the downloaded file
                         mime="text/csv"
                     )
             except Exception as e:
                  st.error(f"Error reading processed data file for download: {e}")
        else:
             st.warning(f"Could not find processed data file at {PROCESSED_DATA_FILE} for download.")


    # Handle case where analysis ran but failed to produce results
    elif st.session_state.final_df is None:
        st.error("❌ Analysis ran but failed to produce valid results. Check the console/terminal where Streamlit is running for specific error messages from the backend.")

# Handle initial state before the button is ever pressed
elif not st.session_state.analysis_run_complete:
    st.info("Click the 'Run Full Analysis Pipeline' button in the sidebar to load data and see the results.")


# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("Dashboard created using Streamlit.")
# Add version number or other info if desired
# st.sidebar.markdown("Version: 1.1")
