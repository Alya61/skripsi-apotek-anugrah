import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# 1. Konfigurasi Tampilan
st.set_page_config(page_title="FSM APOTEK ANUGRAH BEKASI", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌿 Kategori Obat</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Apotek Anugerah Bekasi - Kategori Fast moving, Medium moving, & Slow Moving</p>", unsafe_allow_html=True)
st.divider()

# 2. Fungsi Load Model & Scaler
@st.cache_resource
def load_model():
    with open('model_kmeans.pkl', 'rb') as f:
        return pickle.load(f)

model_kmeans = load_model()
scaler = MinMaxScaler()

# 3. Sidebar untuk Upload
st.sidebar.header("Pusat Kontrol")
uploaded_file = st.sidebar.file_uploader("Drag & Drop File Excel Penjualan", type=["xlsx"])

if uploaded_file:
    # Membaca file yang baru diupload (Dinamis)
    df_raw = pd.read_excel(uploaded_file)
    
   try:
        X = df_raw[['Frekuensi', 'Volume', 'Nilai_Transaksi']]
        
        X_scaled = scaler.fit_transform(X)
        
        df_raw['Cluster'] = model_kmeans.predict(X_scaled)
        
        # Mapping Kategori (Dinamis berdasarkan hasil cluster)
        centers = df_raw.groupby('Cluster')['Frekuensi'].mean().sort_values(ascending=False).index
        mapping = {centers[0]: 'Fast Moving', centers[1]: 'Medium Moving', centers[2]: 'Slow Moving'}
        df_raw['Kategori'] = df_raw['Cluster'].map(mapping)
