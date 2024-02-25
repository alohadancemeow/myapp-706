# create PerceptronMe class ---

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class PerceptronMe(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        print('w=',self.w_)
        for ep in range(self.n_iter):
            print('--- Epoch: ', ep+1)
            errors = 0
            for xi, target in zip(X, y):
                print('x = ',xi, 'y_test = ', target, 'y_predict = ', self.predict(xi))
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                print('w = ',self.w_[1:], 'w0 = ', self.w_[0])
            self.errors_.append(errors)
            print('errors: ', errors)

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', 'x', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')



# PerceptronMe ----

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron

# Install Streamlit
# !pip install streamlit
import streamlit as st

model = pickle.load(open('per_model-706.sav', 'rb'))

st.title("Iris Species Prediction using Perceptron")

x1 = st.slider('Select Input1', 0.0, 10.0, 3.0)
x2 = st.slider('Select Input2', 0.0, 10.0, 5.0)
x3 = st.slider('Select Input3', 0.0, 10.0, 4.0)
x4 = st.slider('Select Input4', 0.0, 10.0, 7.0)

X_new = np.array([[x1,x2,x3,x4]])

model.predict(X_new)

st.write("## Prediction Result")
st.write("Species: ", predict[0])
