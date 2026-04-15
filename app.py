import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# 1. Konfigurasi Tampilan
st.set_page_config(page_title="FMS Apotek Anugerah", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌿 Kategori Obat</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Apotek Anugerah Bekasi - Kategori Fast Movin, Medium Moving, & Slow Moving</p>", unsafe_allow_html=True)
st.divider()

# 2. Fungsi Load Model
@st.cache_resource
def load_model():
    with open('model_kmeans.pkl', 'rb') as f:
        return pickle.load(f)

model_kmeans = load_model()
scaler = MinMaxScaler()

# 3. Sidebar
st.sidebar.header("Pusat Kontrol")
uploaded_file = st.sidebar.file_uploader("Drag & Drop File Excel Penjualan", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file)
    try:
        # Menyesuaikan kolom dengan file mentah kamu
        X = df_raw[['Frekuensi', 'Volume', 'Nilai_Transaksi']]
        X_scaled = scaler.fit_transform(X)
        
        # Prediksi
        df_raw['Cluster'] = model_kmeans.predict(X_scaled)
        
        # Urutkan kategori (Frekuensi tertinggi = Fast)
        centers = df_raw.groupby('Cluster')['Frekuensi'].mean().sort_values(ascending=False).index
        mapping = {centers[0]: 'Fast Moving', centers[1]: 'Medium Moving', centers[2]: 'Slow Moving'}
        df_raw['Kategori'] = df_raw['Cluster'].map(mapping)
        
        # Dashboard Metrik
        counts = df_raw['Kategori'].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("🔥 Fast Moving", f"{counts.get('Fast Moving', 0)} Obat")
        c2.metric("⚡ Medium Moving", f"{counts.get('Medium Moving', 0)} Obat")
        c3.metric("🧊 Slow Moving", f"{counts.get('Slow Moving', 0)} Obat")
        
        st.divider()
        
        tab1, tab2 = st.tabs(["📈 Visualisasi Sebaran", "📄 Data Hasil Klasifikasi"])
        with tab1:
            fig = px.scatter_3d(df_raw, x='Frekuensi', y='Volume', z='Nilai_Transaksi',
                                color='Kategori', hover_name='Nama Obat',
                                color_discrete_map={'Fast Moving':'red', 'Medium Moving':'orange', 'Slow Moving':'blue'})
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.write("### Tabel Detail Klasifikasi")
            st.dataframe(df_raw, use_container_width=True)
            
        # Tombol Download
        csv = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button("Download Hasil (.csv)", data=csv, file_name='Hasil_Kategori_Apotek.csv')
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}. Pastikan kolom di Excel adalah: Nama Obat, Frekuensi, Volume, Nilai_Transaksi")
else:
    st.info("👋 Silakan masukkan file Excel untuk memulai analisis otomatis.")
