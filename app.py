import streamlit as st
import pickle
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    with open('movie_model.pkl','rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("🎬 Movie Prediction App")

st.write("Enter movie features to get a prediction")

# ---- Example Inputs (adjust to your model) ----
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)

# Combine features
features = np.array([[feature_1, feature_2]])

# ---- Prediction ----
if st.button("Predict"):
    try:
        prediction = model.predict(features)
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
