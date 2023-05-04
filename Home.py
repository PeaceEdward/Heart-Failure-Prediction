import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

filepath = os.path.abspath(os.path.join(os.getcwd(), "resources", "data", "heart_disease_data.csv"))

df=pd.read_csv(filepath)

model_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "models", "linmodel.pkl"))
scaler_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "models", "stascaler.pkl"))


# load the pre-trained model and scaler from files
with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
    
st.set_page_config(page_title='Heart Failure Prediction')

    
# define the Streamlit app
def main():
    # set the app header
    st.title('Heart Failure Prediction')
    
if __name__ == "__main__":
  main()

    # add input widgets for heart health parameters

cp_dict = {0: 'Typical angina', 1: 'Atypical angina', 2: 'Non-anginal pain', 3: 'Asymptomatic'}

cp = st.selectbox('Chest Pain Type', options=list(cp_dict.keys()), format_func=lambda x: cp_dict[x])
#cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
thalach = st.slider('Maximum Heart Rate Achieved (bpm)', 50, 250, 150)
exang_dict={0: 'No',1:'Yes'}
exang = st.selectbox('Exercise Induced Angina', options=list(exang_dict.keys()), format_func=lambda x:exang_dict[x])
oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0, 1.0, 0.1)

slope_dict={1:'Upsloping',2:'Flat',3:'Downsloping'}
slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=list(slope_dict.keys()), format_func=lambda x:slope_dict[x])

ca = st.selectbox('Number of Major Vessels Colored by Flouroscopy', [0, 1, 2, 3])

thal_dict = {1: 'normal', 2: 'fixed defect', 3: 'reversible defect'}

# Use the dictionary to display the string representation of thal in the app
thal = st.selectbox('Thalessemia', options=list(thal_dict.keys()), format_func=lambda x: thal_dict[x])



    # define a function to predict the likelihood of heart failure
def predict_heart_failure( cp, thalach, exang, oldpeak, slope, ca, thal):
# scale the input data using the pre-trained scaler
   input_data = np.array([ cp, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
   scaled_data = scaler.transform(input_data)

  # make a prediction using the pre-trained logistic regression model
   pred = model.predict_proba(scaled_data)[:, 1]
   return pred[0]

    # add a button to make the prediction
if st.button('Predict'):
# predict the likelihood of heart failure
    pred = predict_heart_failure(cp,thalach, exang, oldpeak, slope, ca, thal)

    # display the prediction result
    st.write(f'The likelihood of heart failure is {pred:.2f}.')
 

    
