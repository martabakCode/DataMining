import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

@st.cache()
def load_data():
    iris = pd.read_csv('iris.csv')
    
    X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' ]]
    y = iris['Species']
    
    return iris, X,y

@st.cache()
def train_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # membuat model Decision Tree
    tree_model = DecisionTreeClassifier()

    # melakukan pelatihan model terhadap data
    tree_model = tree_model.fit(X_train, y_train)
    
    y_pred = tree_model.predict(X_test)

    acc_secore = round(accuracy_score(y_pred, y_test), 3)
    
    
    return tree_model,acc_secore
    

@st.cache()
def predict(x,y,features):
    tree_model,acc_score = train_model(x,y)
    
    predict = tree_model.predict(np.array(features).reshape(1,-1))
    
    return predict,acc_score

    
    
