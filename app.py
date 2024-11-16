import streamlit as st
import joblib
import numpy as np

# Function to load the model with error handling
def load_model(model_path):
    try:
        # Try loading the model with joblib
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the pre-trained model (ensure the correct file path)
model = load_model('lung_cancer_prediction_model.pkl')

if model:
    st.success("Model loaded successfully!")
else:
    st.warning("Model failed to load. Please check the file.")

# Streamlit interface for user input
st.title("Lung Cancer Prediction")

# Collect input features from the user
age = st.number_input("Age", min_value=0, max_value=100, value=50)
smoking = st.radio("Do you smoke?", ("Yes", "No"))
smoking = 1 if smoking == "Yes" else 0  # Encoding 'Yes' as 1 and 'No' as 0

# Additional input fields, assuming these are relevant to your model
cough = st.radio("Do you have a persistent cough?", ("Yes", "No"))
cough = 1 if cough == "Yes" else 0

chest_pain = st.radio("Do you experience chest pain?", ("Yes", "No"))
chest_pain = 1 if chest_pain == "Yes" else 0

# Define any other necessary inputs here (e.g., breathlessness, weight loss, etc.)

# Prediction button
if st.button("Predict"):
    if model:
        # Example input array, replace with actual model input features
        features = np.array([[age, smoking, cough, chest_pain]])  # Add other features as needed
        prediction = model.predict(features)

        # Display the result
        if prediction[0] == 1:
            st.write("Prediction: High likelihood of lung cancer.")
        else:
            st.write("Prediction: Low likelihood of lung cancer.")
    else:
        st.error("Model not available for prediction.")
