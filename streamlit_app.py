#import library yang dibutuhkan

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

from web_functions import load_data
from web_functions import predict
from web_functions import train_model

#membuat sidebar
st.sidebar.title("Air Quality Index")

#membuat radio option


pm10 = st.sidebar.text_input('Input Kadar PM10 (Partikulat udara) : ')
so2 = st.sidebar.text_input('Input Kadar SO2 (Sulfur dioksida) : ')
co = st.sidebar.text_input('Input Kadar CO (Karbon monoksida) : ')
o3 = st.sidebar.text_input('Input Kadar O3 (Trioksigen/Ozon): ')
no2 = st.sidebar.text_input('Input Kadar NO2 (Oksida nitrogen) : ')

features = [pm10,so2,co,o3,no2]

df,x,y = load_data()

fig = plt.figure(figsize=(10, 4))
sns.countplot(x="categori", data=df)

st.title("Air Quality Index Jakarta")

st.header("Tabel dataset (10 Data Teratas)")
st.table(df.head(10))
st.subheader("Informasi dataset")
st.info(df)
st.subheader("Data Setiap Kategori")
st.pyplot(fig)
st.subheader("Decision Tree)")
model,score = train_model(x,y)
dot_data = tree.export_graphviz(
    decision_tree=model, max_depth=3, out_file=None, filled=True, rounded=True,
    feature_names=x.columns, class_names=['BAIK','SEDANG','TIDAK SEHAT']
)
st.graphviz_chart(dot_data)
#Tombol Prediksi
if st.sidebar.button("Prediksi"):
    prediction, score = predict(x,y,features)
    score = score
    if(prediction == 'BAIK'):
        st.sidebar.success("Index udara baik")
    elif(prediction == 'SEDANG'):
        st.sidebar.success("Index udara sedang")
    elif(prediction == 'TIDAK SEHAT'):
        st.sidebar.success("Index udara tidak baik")

