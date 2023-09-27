#import library yang dibutuhkan

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import os.path
import pandas as pd
from sklearn import tree
from csv import writer
from web_functions import load_data
from web_functions import predict
from web_functions import train_model

# memanggil dataset
df,x,y = load_data()

# Judul dari tab kanan
st.title("Air Quality Index Jakarta")

# Menampilkan 10 Data Teratas dan Terbawah
first_row = df.head(10)
last_row = df.tail(10)
result = pd.concat([first_row,last_row])
st.header("Tabel dataset (10 Data Teratas&Terbawah)")
st.table(result)

# Menampilkan Jumlahrow dataset
st.subheader("Jumlah row dataset")
st.info(df[df.columns[0]].count())

# Menampilkan Data Setiap Kategori
st.subheader("Data Setiap Kategori")
fig = plt.figure(figsize=(10, 4))
sns.countplot(x="categori", data=df)
st.pyplot(fig)

# Menampilkan Pengaturan Testing
st.header("Pengaturan Testing")
number = st.slider("Presentase data",1,100)
if number > 0:
    number = number/100
    model,score,data,data1 = train_model(x,y,number)
    st.subheader("Tabel Test")
    st.table(pd.concat([data,data1],axis=1))

# Decision Tree Show    
st.subheader("Decision Tree")
dot_data = tree.export_graphviz(
    decision_tree=model, max_depth=3, out_file=None, filled=True, rounded=True,
    feature_names=x.columns, class_names=['BAIK','SEDANG','TIDAK SEHAT']
)
st.graphviz_chart(dot_data)

# File Upload
st.header("File Upload")
uploaded_file = st.file_uploader("Choose a CSV file name : indeks-standar-pencemar-udara-di-spku-dataset.csv")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = uploaded_file.getvalue().decode('utf-8').splitlines()         
    st.session_state["preview"] = ''
    for i in range(0, min(5, len(data))):
        st.session_state["preview"] += data[i]
preview = st.text_area("CSV Preview", "", height=150, key="preview")
upload_state = st.text_area("Upload State", "", key="upload_state")

def upload():
    if uploaded_file is None:
        st.session_state["upload_state"] = "Upload a file first!"
    else:
        data = uploaded_file.getvalue().decode('utf-8')
        parent_path = pathlib.Path(__file__).parent.parent.resolve()           
        save_path = os.path.join(parent_path, "datamining")
        complete_name = os.path.join(save_path, uploaded_file.name)
        destination_file = open(complete_name, "w")
        destination_file.write(data)
        destination_file.close()
        st.session_state["upload_state"] = "Saved " + complete_name + " successfully!"
st.button("Upload file to Sandbox", on_click=upload)

#membuat sidebar title
st.sidebar.title("Air Quality Index")

# Inputan untuk sidebar
pm10 = st.sidebar.number_input('Input Kadar PM10 (Partikulat udara) : ',min_value=1)
so2 = st.sidebar.number_input('Input Kadar SO2 (Sulfur dioksida) : ',min_value=1)
co = st.sidebar.number_input('Input Kadar CO (Karbon monoksida) : ',min_value=1)
o3 = st.sidebar.number_input('Input Kadar O3 (Trioksigen/Ozon): ',min_value=1)
no2 = st.sidebar.number_input('Input Kadar NO2 (Oksida nitrogen) : ',min_value=1)

#memasukan inputan ke array features
features = [pm10,so2,co,o3,no2]

#Tombol Prediksi
button = st.sidebar.button("Prediksi")
if button:
    prediction, score = predict(x,y,number,features)
    score = score
    # Menampilkan hasil prediksi
    if(prediction == 'BAIK'):
        st.sidebar.success("Index udara baik")
    elif(prediction == 'SEDANG'):
        st.sidebar.success("Index udara sedang")
    elif(prediction == 'TIDAK SEHAT'):
        st.sidebar.success("Index udara tidak baik")
    
    maks = features.index(max(list(map(int, features))))
    #append to csv
    
    features.append(max(list(map(int, features))))
    #52,23,29,24,12,52
    if maks == 0:
        features.append('PM10')
    elif maks == 1:
        features.append('SO2')
    elif maks == 2:
        features.append('CO')
    elif maks == 3:
        features.append('O3')
    elif maks == 4:
        features.append('NO2')
    #52,23,29,24,12,52,PM10
    features.insert(0, 'DKI5 (Kebon Jeruk) Jakarta Barat')
    #DKI5 (Kebon Jeruk) Jakarta Barat,52,23,29,24,12,52,PM10
    features.insert(0, '2020-12-31')
    #2020-01-09,DKI5 (Kebon Jeruk) Jakarta Barat,52,23,29,24,12,52,PM10
    
    
    features.append(prediction[0])
    #2020-01-09,DKI1 (Bunderan HI),52,23,29,24,12,52,PM10,SEDANG
    with open('indeks-standar-pencemar-udara-di-spku-dataset.csv', 'a') as f_object:
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
    
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(features)
    
        # Close the file object
        f_object.close()