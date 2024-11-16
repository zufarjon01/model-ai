import streamlit as st
import numpy as np
from joblib import load

# 1. Modelni yuklash
model_path = 'lung_cancer_prediction_model.joblib'  # Modelning to'liq manzili
model = load(model_path)

# 2. Kiruvchi ma'lumotlar uchun formani yaratish
st.title('Lung Cancer Prediction')

# Kiruvchi ma'lumotlarni to'plash
age = st.number_input('Age', min_value=0, max_value=100, value=50)
smoking = st.selectbox('Smoking', ['Yes', 'No'])
yellow_fingers = st.selectbox('Yellow Fingers', ['Yes', 'No'])
anxiety = st.selectbox('Anxiety', ['Yes', 'No'])
peer_pressure = st.selectbox('Peer Pressure', ['Yes', 'No'])
chronic_disease = st.selectbox('Chronic Disease', ['Yes', 'No'])
fatigue = st.selectbox('Fatigue', ['Yes', 'No'])
allergy = st.selectbox('Allergy', ['Yes', 'No'])
wheezing = st.selectbox('Wheezing', ['Yes', 'No'])
alcohol_consumption = st.selectbox('Alcohol Consumption', ['Yes', 'No'])
coughing = st.selectbox('Coughing', ['Yes', 'No'])
shortness_of_breath = st.selectbox('Shortness of Breath', ['Yes', 'No'])
swallowing_difficulty = st.selectbox('Swallowing Difficulty', ['Yes', 'No'])
chest_pain = st.selectbox('Chest Pain', ['Yes', 'No'])

# 3. Ma'lumotlarni modelga kiritish uchun raqamli formatga o'tkazish
input_data = np.array([
    age,
    1 if smoking == 'Yes' else 0,
    1 if yellow_fingers == 'Yes' else 0,
    1 if anxiety == 'Yes' else 0,
    1 if peer_pressure == 'Yes' else 0,
    1 if chronic_disease == 'Yes' else 0,
    1 if fatigue == 'Yes' else 0,
    1 if allergy == 'Yes' else 0,
    1 if wheezing == 'Yes' else 0,
    1 if alcohol_consumption == 'Yes' else 0,
    1 if coughing == 'Yes' else 0,
    1 if shortness_of_breath == 'Yes' else 0,
    1 if swallowing_difficulty == 'Yes' else 0,
    1 if chest_pain == 'Yes' else 0
]).reshape(1, -1)

# Ma'lumotlarni ko'rsatish (formatini va o'lchamini tekshirish)
st.write('Input data shape:', input_data.shape)

# 4. Bashorat qilish
if st.button('Predict'):
    st.write('Input data:', input_data)  # Modelga kirayotgan ma'lumotni ko'rsatish
    try:
        # Modelni ishlatish
        prediction = model.predict(input_data)

        # Bashoratni foydalanuvchiga ko'rsatish
        st.write('Prediction:', prediction)
        
        if prediction[0] == 1:
            st.warning('High risk of lung cancer.')
        else:
            st.success('Low risk of lung cancer.')
    except Exception as e:
        st.error(f'Error occurred during prediction: {e}')
