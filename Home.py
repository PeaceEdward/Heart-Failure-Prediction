import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

filepath = os.path.abspath(os.path.join(os.getcwd(), "resources", "data", "heart_disease_data.csv"))

df=pd.read_csv(filepath)

model_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "models", "lmodel.pkl"))
scaler_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "models", "stscaler.pkl"))


# load the pre-trained model and scaler from files
with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# define a function to predict the likelihood of heart failure
def predict_heart_failure(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # scale the input data using the pre-trained scaler
    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    scaled_data = scaler.transform(input_data)

    # make a prediction using the pre-trained logistic regression model
    pred = model.predict_proba(scaled_data)[:, 1]
    return pred[0]

# define the Streamlit app
def app():
    # set the page title
    st.set_page_config(page_title='Heart Failure Prediction')

    # set the app header
    st.title('Heart Failure Prediction')

    # add input widgets for heart health parameters
    age = st.slider('Age', 18, 100, 50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    chol = st.slider('Cholesterol (mg/dL)', 100, 600, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', [0, 1])
    restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
    thalach = st.slider('Maximum Heart Rate Achieved (bpm)', 50, 250, 150)
    exang = st.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0, 1.0, 0.1)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Flourosopy', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', [0, 1, 2, 3])

    # convert the sex input to numerical form (0 for male, 1 for female)
    sex = 0 if sex == 'Male' else 1

    # add a button to make the prediction
    if st.button('Predict'):
        # predict the likelihood of heart failure
        pred = predict_heart_failure(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

        # display the prediction result
        st.write(f'The likelihood of heart failure is {pred:.2f}.')
