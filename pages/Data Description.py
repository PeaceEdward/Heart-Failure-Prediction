import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# Read the heart disease data into a Pandas dataframe
filepath = os.path.abspath(os.path.join(os.getcwd(), "resources", "data", "heart_disease_data.csv"))
df = pd.read_csv(filepath)

# Display the number of samples and features in the dataset
num_samples, num_features = df.shape
st.write(f"Number of samples: {num_samples}")
st.write(f"Number of features: {num_features}")

# Display basic statistics about the dataset
st.write("Basic statistics:")
st.write(df.describe())

# Display a histogram of the target variable
st.write("Histogram of target variable:")
st.bar_chart(df["target"].value_counts())

# create separate dataframes for patients with and without heart disease
df_disease = df[df['target'] == 1]
df_no_disease = df[df['target'] == 0]

# plot the age distribution of patients with and without heart disease
fig, ax = plt.subplots()
ax.hist(df_disease['age'], alpha=0.5, label='Heart Disease')
ax.hist(df_no_disease['age'], alpha=0.5, label='No Heart Disease')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.legend()

# display the plot on Streamlit
st.pyplot(fig)


df_disease = df[df['target'] == 1]
df_no_disease = df[df['target'] == 0]

# create a scatter plot of age vs. maximum heart rate, colored by disease status
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(df_disease['age'], df_disease['thalach'], c='red', alpha=0.5, label='Heart Disease')
ax.scatter(df_no_disease['age'], df_no_disease['thalach'], c='blue', alpha=0.5, label='No Heart Disease')
ax.set_xlabel('Age')
ax.set_ylabel('Max Heart Rate')
ax.legend()

# display the plot in Streamlit
st.pyplot(fig)