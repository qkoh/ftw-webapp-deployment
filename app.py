# Import packages
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import streamlit as st

# Title of your web app
st.title('Forecasting Sales')

# Describe your web app
st.write('We demostrate how we can forecast advertising sales based on ad expenditure')

# Read data
data = pd.read_csv('data/advertising_regression - advertising_regression.csv')

# Show data
data

# Create sidebar

# Sidebar Description
st.sidebar.subheader('Adversting Costs')

# TV Slider
TV = st.sidebar.slider('TV Adversiting Cost', 0, 300, 150)

# Radio Adversting Cost Slider
Radio = st.sidebar.slider('Radio Adversiting Cost', 0, 50, 25)

# Newspaper Adversting Cost Slider
Newspaper = st.sidebar.slider('Newspaper Adversiting Cost', 0, 250, 125)

# Histogram
hist_values = np.histogram(data.radio, bins=300, range=(0, 300))[0]

# Show Bar Chart
st.bar_chart(hist_values)

# Histogram
hist_values = np.histogram(data.TV, bins=300, range=(0, 300))[0]

# Show Bar Chart
st.bar_chart(hist_values)

# Histogram
hist_values = np.histogram(data.newspaper, bins=300, range=(0, 300))[0]

# Show Bar Chart
st.bar_chart(hist_values)

# Load saved machine learning model
saved_model = joblib.load('advertising_model.sav')

# Predict sales using variables/features
predicted_sales = saved_model.predict([[TV, Radio, Newspaper]])[0]

# Print prediction
st.write(f"Predicted sales is {int(predicted_sales*1000)} dollars.")
