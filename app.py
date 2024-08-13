import numpy as np
import streamlit as st
import pickle

# Import model
with open('pipe.pkl', 'rb') as file:
    pipe = pickle.load(file)

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI', 'Microsoft', 'Toshiba', 'Huawei',
                                 'Xiaomi', 'Vero', 'Razer', 'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG'])

# Type of laptop
lap_type = st.selectbox('Type', ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible',
                                 'Workstation', 'Netbook'])

# RAM
ram_type = st.selectbox('RAM(GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight', min_value=0.0)

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen Size
screen_size = st.number_input('Screen Size', min_value=0.0)

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x1800', '2880x1800',
                                                '2560x1600', '2560x1440', '2304x1440'])

# CPU
cpu = st.selectbox('CPU', ['Intel Core i5', 'Intel Core i7', 'AMD Processor', 'Intel Core i3', 'other Intel Processor'])

# HDD
hdd = st.selectbox('HDD(GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD(GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', ['Intel', 'AMD', 'Nvidia'])

# OS
os = st.selectbox('OS', ['Mac', 'Others', 'Windows'])

if st.button('Predict Price'):
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    X_res, Y_res = map(int, resolution.split('x'))
    if screen_size != 0:
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    else:
        st.error("Screen size cannot be zero.")
        ppi = 0

    query = np.array([company, lap_type, ram_type, weight, touchscreen, ips,
                      ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    try:
        prediction = pipe.predict(query)
        st.title("The Price of laptop is " + str(int(np.exp(prediction[0]))))
    except Exception as e:
        st.error(f"An error occurred: {e}")