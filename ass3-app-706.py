import pickle
import numpy as np
from sklearn.linear_model import Perceptron

# Install Streamlit
# !pip install streamlit
import streamlit as st

model = pickle.load(open('/content/per_model-706.sav', 'rb'))

pickle.dump(model, open('per_model-706.sav', 'wb'))

st.title("Iris Species Prediction using Perceptron")

x1 = st.slider('Select Input1', 0.0, 10.0, 3.0)
x2 = st.slider('Select Input2', 0.0, 10.0, 5.0)
x3 = st.slider('Select Input3', 0.0, 10.0, 4.0)
x4 = st.slider('Select Input4', 0.0, 10.0, 7.0)

X_new = np.array([[x1,x2,x3,x4]])
X_new

model.predict(X_new)
