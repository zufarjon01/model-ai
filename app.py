import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
try:
    model = joblib.load('lung_cancer_prediction_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit title
st.title("Lung Cancer Prediction")

# Input form for patient data
with st.form(key='input_form'):
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    smoking = st.selectbox("Smoking Status", options=["Yes", "No"])
    yellow_fingers = st.selectbox("Yellow Fingers", options=["Yes", "No"])
    anxiety = st.selectbox("Anxiety", options=["Yes", "No"])
    peer_pressure = st.selectbox("Peer Pressure", options=["Yes", "No"])
    chronic_disease = st.selectbox("Chronic Disease", options=["Yes", "No"])
    fatigue = st.selectbox("Fatigue", options=["Yes", "No"])
    allergy = st.selectbox("Allergy", options=["Yes", "No"])
    wheezing = st.selectbox("Wheezing", options=["Yes", "No"])
    alcohol_consuming = st.selectbox("Alcohol Consuming", options=["Yes", "No"])
    cough = st.selectbox("Cough", options=["Yes", "No"])
    shortness_of_breath = st.selectbox("Shortness of Breath", options=["Yes", "No"])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty", options=["Yes", "No"])
    chest_pain = st.selectbox("Chest Pain", options=["Yes", "No"])

    submit_button = st.form_submit_button(label="Submit")
    
    if submit_button:
        # Prepare input data for prediction
        input_data = {
            "Age": age,
            "Gender": 1 if gender == "Male" else 0,  # Male=1, Female=0
            "Smoking": 1 if smoking == "Yes" else 0,
            "Yellow Fingers": 1 if yellow_fingers == "Yes" else 0,
            "Anxiety": 1 if anxiety == "Yes" else 0,
            "Peer Pressure": 1 if peer_pressure == "Yes" else 0,
            "Chronic Disease": 1 if chronic_disease == "Yes" else 0,
            "Fatigue": 1 if fatigue == "Yes" else 0,
            "Allergy": 1 if allergy == "Yes" else 0,
            "Wheezing": 1 if wheezing == "Yes" else 0,
            "Alcohol Consuming": 1 if alcohol_consuming == "Yes" else 0,
            "Cough": 1 if cough == "Yes" else 0,
            "Shortness of Breath": 1 if shortness_of_breath == "Yes" else 0,
            "Swallowing Difficulty": 1 if swallowing_difficulty == "Yes" else 0,
            "Chest Pain": 1 if chest_pain == "Yes" else 0
        }

        # Convert input to DataFrame for prediction
        input_df = pd.DataFrame([input_data])

        # Make prediction using the model
        try:
            prediction = model.predict(input_df)
            prediction_prob = model.predict_proba(input_df)[:, 1]  # Probability of class 1

            # Display the results
            if prediction[0] == 1:
                st.subheader("Prediction: Positive for Lung Cancer")
                st.write(f"Probability: {prediction_prob[0]:.2f}")
            else:
                st.subheader("Prediction: Negative for Lung Cancer")
                st.write(f"Probability: {1 - prediction_prob[0]:.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
