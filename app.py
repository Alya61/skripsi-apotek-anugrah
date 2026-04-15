import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from sklearn.cluster import KMeans

# 1. Konfigurasi Tampilan
st.set_page_config(page_title="FMS Apotek Anugerah", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌿 Kategori Obat</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Apotek Anugerah Bekasi - Klasifikasi Otomatis Berdasarkan Data Penjualan</p>", unsafe_allow_html=True)
st.divider()

# 2. Sidebar
st.sidebar.header("Pusat Kontrol")
uploaded_file = st.sidebar.file_uploader("Drag & Drop File Excel Penjualan", type=["xlsx", "xls"])

if uploaded_file:
    # Membaca Data Mentah
    df = pd.read_excel(uploaded_file)
    
    try:
        # --- DATA PREPARATION (AGREGASI) ---
        # Menyesuaikan dengan logika agregasi di Colab kamu
        data_agregasi = df.groupby('Nama Obat', sort=False).agg({
            'Tanggal Transaksi': 'count',
            'Jumlah Terjual': 'sum',
            'Total Harga': 'sum'
        }).reset_index()

        data_agregasi.columns = ['Nama Obat', 'Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']

        # --- DATA CLEANING ---
        # Menghapus missing value & duplikat secara otomatis
        data_agregasi.dropna(inplace=True)
        data_agregasi.drop_duplicates(inplace=True)

        # --- TRANSFORMASI & NORMALISASI ---
        fitur_numerik = ['Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']
        scaler = MinMaxScaler()
        data_normalisasi = scaler.fit_transform(data_agregasi[fitur_numerik])
        
        X_modeling = pd.DataFrame(data_normalisasi, columns=fitur_numerik)

        # --- MODELING (K-MEANS) ---
        # Menggunakan K=3 sesuai riset kamu
        model_kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        data_agregasi['Cluster'] = model_kmeans.fit_predict(X_modeling)

        # --- PROFILING & PELABELAN ---
        centroid = data_agregasi.groupby('Cluster')[fitur_numerik].mean()
        urutan = centroid['Frekuensi Transaksi'].sort_values(ascending=False).index
        mapping_kategori = {urutan[0]: 'Fast Moving', urutan[1]: 'Medium Moving', urutan[2]: 'Slow Moving'}
        data_agregasi['Kategori'] = data_agregasi['Cluster'].map(mapping_kategori)

        # --- TAMPILAN DASHBOARD ---
        counts = data_agregasi['Kategori'].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("🔥 Fast Moving", f"{counts.get('Fast Moving', 0)} Obat")
        c2.metric("⚡ Medium Moving", f"{counts.get('Medium Moving', 0)} Obat")
        c3.metric("🧊 Slow Moving", f"{counts.get('Slow Moving', 0)} Obat")

        st.divider()

        tab1, tab2 = st.tabs(["📊 Visualisasi 3D", "📋 Tabel Hasil Klasifikasi"])

        with tab1:
            st.subheader("Visualisasi Sebaran Obat")
            fig = px.scatter_3d(data_agregasi, 
                                x='Frekuensi Transaksi', y='Volume Penjualan', z='Nilai Transaksi',
                                color='Kategori', hover_name='Nama Obat',
                                color_discrete_map={'Fast Moving': 'red', 'Medium Moving': 'orange', 'Slow Moving': 'blue'},
                                opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Detail Data Obat")
            st.dataframe(data_agregasi[['Nama Obat', 'Kategori', 'Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']], use_container_width=True)

        # Tombol Download
        csv = data_agregasi.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil Analisis (.csv)", data=csv, file_name='Hasil_Klasifikasi_Apotek.csv')

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}. Pastikan file Excel memiliki kolom: 'Nama Obat', 'Tanggal Transaksi', 'Jumlah Terjual', dan 'Total Harga'.")
else:
    st.info("👋 Halo! Silakan masukkan file Excel penjualan Apotek Anugerah Bekasi untuk dianalisis.")
