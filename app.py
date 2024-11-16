# import streamlit as st
# import joblib
# import numpy as np

# # Load the trained model
# try:
#     model = joblib.load("lung_cancer_prediction_model.pkl")
#     st.success("Model loaded successfully!")
# except Exception as e:
#     st.error(f"Error loading the model: {e}")
#     st.stop()

# # Streamlit interface
# st.title("Lung Cancer Prediction App")

# st.markdown("""
# This app predicts the likelihood of lung cancer based on user inputs. Fill in the details below and click **Predict**.
# """)

# # Input fields for user data
# gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
# age = st.number_input("Age", min_value=1, max_value=120, step=1)
# smoking = st.selectbox("Smoking", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# yellow_fingers = st.selectbox("Yellow Fingers", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# anxiety = st.selectbox("Anxiety", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# peer_pressure = st.selectbox("Peer Pressure", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# chronic_disease = st.selectbox("Chronic Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# fatigue = st.selectbox("Fatigue", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# allergy = st.selectbox("Allergy", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# wheezing = st.selectbox("Wheezing", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# alcohol_consuming = st.selectbox("Alcohol Consuming", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# coughing = st.selectbox("Coughing", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# shortness_of_breath = st.selectbox("Shortness of Breath", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# swallowing_difficulty = st.selectbox("Swallowing Difficulty", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# chest_pain = st.selectbox("Chest Pain", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# # Predict button
# if st.button("Predict"):
#     try:
#         # Preparing input features
#         features = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
#                                chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
#                                coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])
        
#         # Model prediction
#         prediction = model.predict(features)
#         result = "Lung Cancer Detected" if prediction[0] == 1 else "No Lung Cancer Detected"
#         st.success(f"Prediction: {result}")
#     except Exception as e:
#         st.error(f"Error during prediction: {e}")
import streamlit as st
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load('lung_cancer_prediction_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit interface for user input
st.title("Lung Cancer Prediction")

# Example input fields for patient data (adjust according to your dataset)
age = st.number_input("Age", min_value=0, max_value=100, value=50)
smoking = st.selectbox("Smoker?", options=["Yes", "No"])
chronic_cough = st.selectbox("Chronic Cough?", options=["Yes", "No"])
shortness_of_breath = st.selectbox("Shortness of Breath?", options=["Yes", "No"])

# Convert inputs to a format suitable for the model
input_data = np.array([age, 1 if smoking == "Yes" else 0, 1 if chronic_cough == "Yes" else 0, 1 if shortness_of_breath == "Yes" else 0]).reshape(1, -1)

# Make predictions
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        st.write(f"Prediction: {'Cancer Likely' if prediction[0] == 1 else 'No Cancer Likely'}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
