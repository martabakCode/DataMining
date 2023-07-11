#import library yang dibutuhkan

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import os.path
from sklearn import tree
from csv import writer
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
st.table(df.tail(10))
st.subheader("Jumlah row dataset")
st.info(df[df.columns[0]].count())
st.subheader("Data Setiap Kategori")
st.pyplot(fig)
st.subheader("Decision Tree")
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
    
    #append to csv
    features.append(str(max(list(map(int, features)))))
    if features.index(str(max(list(map(int, features))))) == 0:
        features.append('PM10')
    elif features.index(str(max(list(map(int, features))))) == 1:
        features.append('SO2')
    elif features.index(str(max(list(map(int, features))))) == 2:
        features.append('CO')
    elif features.index(str(max(list(map(int, features))))) == 3:
        features.append('O3')
    elif features.index(str(max(list(map(int, features))))) == 4:
        features.append('NO2')
    features.insert(0, 'DKI5 (Kebon Jeruk) Jakarta Barat')
    features.insert(0, '2020-12-31')
    features.append(prediction[0])
    with open('indeks-standar-pencemar-udara-di-spku-dataset.csv', 'a') as f_object:
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
    
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(features)
    
        # Close the file object
        f_object.close()

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