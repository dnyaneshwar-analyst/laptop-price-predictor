"""
Streamlit app for Laptop Price Prediction.

Usage:
    streamlit run app.py

Requires model.pkl (produced by train.py) in the same directory.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILE = "model.pkl"

st.set_page_config(page_title="Laptop Price Predictor", page_icon="laptop", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)


model = load_model()

st.title("💻 Laptop Price Predictor")
st.write("Enter laptop specifications to estimate its price.")

if model is None:
    st.error(
        f"Model file '{MODEL_FILE}' not found. "
        f"Run `python train.py` first to train and save the model."
    )
    st.stop()

COMPANIES = ['Acer', 'Apple', 'Asus', 'Chuwi', 'Dell', 'Fujitsu', 'Google', 'HP',
              'Huawei', 'LG', 'Lenovo', 'MSI', 'Mediacom', 'Microsoft', 'Razer',
              'Samsung', 'Toshiba', 'Vero', 'Xiaomi']

TYPES = ['2 in 1 Convertible', 'Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation']

OS_OPTIONS = ['Windows 10', 'Windows 10 S', 'Windows 7', 'macOS', 'Mac OS X',
               'Linux', 'Chrome OS', 'Android', 'No OS']

RAM_OPTIONS = [2, 4, 6, 8, 12, 16, 24, 32, 64]

CPU_OPTIONS = [
    "Intel Core i3 1.8GHz", "Intel Core i5 2.5GHz", "Intel Core i5 1.6GHz",
    "Intel Core i7 2.8GHz", "Intel Core i7 3.1GHz",
    "Intel Celeron Dual Core N3060 1.6GHz",
    "Intel Pentium Quad Core N3710 1.6GHz",
    "AMD A9-Series 9420 3GHz", "AMD Ryzen 1600 3.2GHz",
]

GPU_OPTIONS = [
    "Intel HD Graphics 620", "Intel Iris Plus Graphics 640", "Intel UHD Graphics 620",
    "Nvidia GeForce GTX 1050", "Nvidia GeForce GTX 1060", "Nvidia GeForce MX150",
    "AMD Radeon 530", "AMD Radeon RX 580",
]

PRIMARY_STORAGE_SIZES = ["128GB", "256GB", "512GB", "1TB", "2TB"]
PRIMARY_STORAGE_TYPES = ["SSD", "HDD", "Flash Storage", "Hybrid"]

SECONDARY_STORAGE_SIZES = ["None", "128GB", "256GB", "512GB", "1TB", "2TB"]
SECONDARY_STORAGE_TYPES = ["SSD", "HDD", "Flash Storage", "Hybrid"]

RESOLUTIONS = ["1366x768", "1600x900", "1920x1080", "2560x1440", "3840x2160", "2560x1600"]

with st.form("laptop_form"):
    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox("Brand", COMPANIES)
        type_name = st.selectbox("Type", TYPES)
        ram = st.selectbox("RAM (GB)", RAM_OPTIONS, index=3)
        weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        os_choice = st.selectbox("Operating System", OS_OPTIONS)
        inches = st.number_input("Screen Size (inches)", min_value=10.0, max_value=18.0, value=15.6, step=0.1)

    with col2:
        cpu = st.selectbox("CPU", CPU_OPTIONS)
        gpu = st.selectbox("GPU", GPU_OPTIONS)
        resolution = st.selectbox("Screen Resolution", RESOLUTIONS, index=2)
        touchscreen = st.checkbox("Touchscreen")
        ips = st.checkbox("IPS Display")

    st.subheader("Storage")
    sc1, sc2 = st.columns(2)
    with sc1:
        primary_size = st.selectbox("Primary Storage Size", PRIMARY_STORAGE_SIZES, index=1)
        primary_type = st.selectbox("Primary Storage Type", PRIMARY_STORAGE_TYPES, index=0)
    with sc2:
        secondary_size = st.selectbox("Secondary Storage Size", SECONDARY_STORAGE_SIZES, index=0)
        secondary_type = st.selectbox("Secondary Storage Type", SECONDARY_STORAGE_TYPES, index=0)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    res_prefix = ""
    if touchscreen:
        res_prefix += "Touchscreen "
    if ips:
        res_prefix += "IPS Panel "
    screen_resolution = f"{res_prefix}{resolution}".strip()

    memory_str = f"{primary_size} {primary_type}"
    if secondary_size != "None":
        memory_str += f" + {secondary_size} {secondary_type}"

    input_df = pd.DataFrame([{
        "Company": company,
        "TypeName": type_name,
        "Inches": inches,
        "ScreenResolution": screen_resolution,
        "Cpu": cpu,
        "Ram": f"{ram}GB",
        "Memory": memory_str,
        "Gpu": gpu,
        "OpSys": os_choice,
        "Weight": f"{weight}kg",
    }])

    try:
        pred_log = model.predict(input_df)[0]
        price = np.expm1(pred_log)
        st.success(f"### Estimated Price: ₹{price:,.0f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

st.caption("Model: Linear Regression | Trained on 1303 laptops dataset")
