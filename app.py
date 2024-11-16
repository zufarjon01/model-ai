import streamlit as st
import joblib

# Load the model
try:
    model = joblib.load('lung_cancer_prediction_model.pkl')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit UI for input
st.title("Lung Cancer Prediction")
age = st.number_input("Age", min_value=0, max_value=100, value=50)
smoking = st.radio("Do you smoke?", ("Yes", "No"))

if st.button("Predict"):
    prediction = model.predict([[age, smoking]])  # Adjust based on your features
    st.write(f"Prediction: {prediction}")
