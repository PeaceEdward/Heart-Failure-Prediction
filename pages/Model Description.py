import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

filepath = os.path.abspath(os.path.join(os.getcwd(), "resources", "data", "heart_disease_data.csv"))

df=pd.read_csv(filepath)

model_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "models", "linmodel.pkl"))
scaler_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "models", "stascaler.pkl"))


# load the pre-trained model and scaler from files
with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
    
st.title('Model Description')

X=df[['cp','thalach','exang','oldpeak','slope','ca','thal']]
y=df['target']
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, stratify=y,random_state=2)
# Train your machine learning model and make predictions on test data
# Calculate accuracy score and ROC curve

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Display the accuracy score
st.write(f"Accuracy score: {accuracy:.2f}")

# Display the ROC curve
st.line_chart({"false_positive_rate": fpr, "true_positive_rate": tpr}, width=0, height=0, use_container_width=True)


# Calculate precision and recall values
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

# Plot the precision-recall curve
fig, ax = plt.subplots(figsize=(8, 6))
plot_precision_recall_curve(model, X_test_scaled, y_test, ax=ax)
ax.set_title('Precision-Recall Curve')
st.pyplot(fig)

