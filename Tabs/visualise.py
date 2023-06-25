import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import plot_confusion_matrix
from sklearn import tree
import streamlit as st

from web_functions import train_model

def app(df, x, y):
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Visualisasi Prediksi Tanaman Iris")

    if st.checkbox("Plot Confusion Matrix"):
        model,score = train_model(x,y)
        plt.figure(figsize=(10,6))
        plot_confusion_matrix(model, x, y, values_format='d')
        st.pyplot()

    if st.checkbox("Plot Decision Tree"):
        model,score = train_model(x,y)
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=3, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['Iris-setosa','Iris-versicolor','Iris-virginica']
        )
        st.export_graphviz(dot_data)
