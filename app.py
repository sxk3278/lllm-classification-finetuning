
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Resource Architecture: Load models at startup with caching ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('preference_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model or vectorizer files not found. Please ensure 'preference_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as this app.")
        st.stop()

model, vectorizer = load_models()

# --- 2. Product Interface Design ---
st.set_page_config(layout="wide", page_title="AI Product PM Sandbox")

st.title("🤖 AI Product PM Sandbox: Human Alignment Predictor")
st.markdown("This system utilizes an optimized gradient-boosted decision tree pipeline (LightGBM) trained on 57,000 real human evaluator records to benchmark model alignment and predict user satisfaction.")
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

        # Call predict_proba() on the LightGBM model
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
