import streamlit as st
import pickle 
import pandas as pd
import json 
import requests
from streamlit_lottie import st_lottie 
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder

st.set_page_config(
page_title="Prediction App")

html_temp="""
<div style = "background-color: darkbLue; padding: 16px">
<h2 style="color: white; text-align:center;"> Welcome to Car Price Prediction App </h2>
</div>"""

st.markdown(html_temp, unsafe_allow_html=True)

url = requests.get( 

    "https://lottie.host/897555a9-df94-4ed5-bda9-7c782589dfac/3hoe6phFCL.json") 
# Creating a blank dictionary to store JSON file, 
# as their structure is similar to Python Dictionary 

url_json = dict() 

  

if url.status_code == 200: 

    url_json = url.json() 

else: 

    print("Error in the URL") 

st_lottie(url_json) 

st.write("\n\n"*2)

rf_model = 'rf_model_final'
#rf_trans = 'transformer_final'

transformer = OrdinalEncoder()


with st.sidebar:
    st.subheader('Car Specs to Predict Price')

make_model = st.sidebar.selectbox("Model Selection", ("Audi A3","Audi A2", "Audi A1", "Opel Insignia", "Opel Astra", "Opel Corsa", "Renault Clio", "Renault Espace", "Renault Duster"))
hp_kW = st.sidebar.number_input("Horse Power:",min_value=40, max_value=294, value=120, step=5)
age = st.sidebar.number_input("age:",min_value=0, max_value=3, value=0, step=1)
km = st.sidebar.number_input("km:",min_value=0, max_value=317000, value=10000, step=5000)
weight_kg = st.sidebar.number_input("Weight kg:",min_value=840, max_value=2471, value=1000, step=10)
Gears = st.sidebar.number_input("Gears:",min_value=5, max_value=8, value=5, step=1)
Gearing_Type = st.sidebar.radio("Gearing Type", ("Manual", "Automatic", "Semi-automatic"))


model = pickle.load(open(rf_model, 'rb'))


my_dict = { "age":age, "hp_kW":hp_kW, "km":km, "Gearing_Type":Gearing_Type, "make_model":make_model, "Weight_kg": weight_kg, "Gears":Gears}
df = pd.DataFrame.from_dict([my_dict])

transformer.fit(df[["Gearing_Type", "make_model"]])


st.write("Selected Specs: \n")
st.table(df)

df2 = df.copy()

df[["Gearing_Type","make_model"]]= transformer.transform(df[["Gearing_Type","make_model"]])


if st.button("Predict"):
    pred = model.predict(df)
    st.success("The estimated price of your car is $ {}. ".format(int(pred[0])))
    
st.write("\n\n")

with st.sidebar:
    st.subheader('📩 **Contact:** ')
st.sidebar.write("**LinkedIn:** [link](https://www.linkedin.com/in/sara-alnajim-24ab24177)")
st.sidebar.write("**GitHub:** [link](https://github.com/saraalnajim)")

st.write("\n\n")
