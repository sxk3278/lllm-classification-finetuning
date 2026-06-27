
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- 1. Resource Architecture: Load and train models at startup with caching ---
@st.cache_resource
def load_and_train_models():
    try:
        # Load 'mini_train.csv'
        train_df = pd.read_csv('mini_train.csv')

        # Handle missing values for text columns
        text_columns = ["prompt", "response_a", "response_b"]
        for col in text_columns:
            train_df[col] = train_df[col].fillna("")

        # Combine prompt + responses into one text feature
        train_df["text"] = (
            train_df["prompt"]
            + " [A] "
            + train_df["response_a"]
            + " [B] "
            + train_df["response_b"]
        )

        # Convert target columns into one integer label
        target_columns = ["winner_model_a", "winner_model_b", "winner_tie"]
        train_df["target"] = train_df[target_columns].values.argmax(axis=1)

        # Fit a TfidfVectorizer
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            stop_words="english"
        )
        X_tfidf = vectorizer.fit_transform(train_df["text"])

        # Fit a LogisticRegression classifier
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        )
        model.fit(X_tfidf, train_df["target"])

        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: 'mini_train.csv' not found. Please ensure 'mini_train.csv' is in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during model setup: {e}")
        st.stop()

model, vectorizer = load_and_train_models()

# --- 2. Product Interface Design ---
st.set_page_config(layout="wide", page_title="AI Product PM Sandbox")

st.title("🤖 AI Product PM Sandbox: Human Alignment Predictor")
st.markdown("This system utilizes an optimized **Logistic Regression** pipeline trained on a small sample of human evaluator records to benchmark model alignment and predict user satisfaction.")
st.markdown("--- ")

# --- 3. Interactive Input Panel ---
st.subheader("Enter Chatbot Interaction Details")

prompt = st.text_area("System Prompt", height=150, placeholder="e.g., 'Write a short story about a space explorer.'")

col1, col2 = st.columns(2)

with col1:
    response_a = st.text_area("Chatbot Response Option A", height=200, placeholder="e.g., 'The explorer landed on a purple planet.'")

with col2:
    response_b = st.text_area("Chatbot Response Option B", height=200, placeholder="e.g., 'On a distant moon, a lone astronaut searched for life.'")

st.markdown("--- ")

# --- 4. Core Logic Execution ---
if st.button("⚡ Predict Human Preference Alignment"):
    if not prompt or not response_a or not response_b:
        st.warning("Please fill in all text input fields before predicting.")
    else:
        # Programmatically join the text inputs using the exact training string pattern
        input_text = (
            prompt
            + " [A] "
            + response_a
            + " [B] "
            + response_b
        )

        # Transform this string using the loaded vectorizer
        input_tfidf = vectorizer.transform([input_text])

        # Call predict_proba() on the LogisticRegression model
        probabilities = model.predict_proba(input_tfidf)[0]

        # Extract the 3 output probabilities
        prob_a = probabilities[0]
        prob_b = probabilities[1]
        prob_tie = probabilities[2]

        st.subheader("Prediction Results:")

        # --- 5. High-Impact Visualizations ---
        col_a, col_b, col_tie = st.columns(3)

        with col_a:
            st.metric("Probability Human Chooses Model A", f"{prob_a:.2%}")
            st.progress(prob_a)

        with col_b:
            st.metric("Probability Human Chooses Model B", f"{prob_b:.2%}")
            st.progress(prob_b)

        with col_tie:
            st.metric("Probability of a Tie", f"{prob_tie:.2%}")
            st.progress(prob_tie)

        st.markdown("\n\n---")
        st.info("Note: The sum of probabilities for A, B, and Tie should be approximately 100%.")
