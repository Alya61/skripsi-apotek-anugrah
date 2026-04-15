import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px

# 1. Konfigurasi Tampilan
st.set_page_config(page_title="Sistem Kategori Obat Apotek Anugerah", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌿 Sistem Kategori Obat</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Apotek Anugerah Bekasi - Penentuan Kategori Fast, Medium, & Slow Moving</p>", unsafe_allow_html=True)
st.divider()

# 2. Sidebar
st.sidebar.header("Pusat Kontrol")
uploaded_file = st.sidebar.file_uploader("Upload File Excel Penjualan", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    try:
        # --- TAHAP AGREGASI (Sesuai Logika Colab) ---
        data_agregasi = df.groupby('Nama Obat', sort=False).agg({
            'Tanggal Transaksi': 'count',
            'Jumlah Terjual': 'sum',
            'Total Harga': 'sum'
        }).reset_index()

        data_agregasi.columns = ['Nama Obat', 'Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']
        data_agregasi.dropna(inplace=True)
        data_agregasi.drop_duplicates(inplace=True)

        # --- NORMALISASI ---
        fitur_numerik = ['Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data_agregasi[fitur_numerik])
        
        # --- MODELING K-MEANS (K=3) ---
        model_kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        data_agregasi['Cluster'] = model_kmeans.fit_predict(X_scaled)

        # --- KUNCI LOGIKA DINAMIS (PENTING!) ---
        # Menentukan label berdasarkan rata-rata Frekuensi agar tidak tertukar
        stats = data_agregasi.groupby('Cluster')['Frekuensi Transaksi'].mean().sort_values(ascending=False)
        mapping = {stats.index[0]: 'Fast Moving', stats.index[1]: 'Medium Moving', stats.index[2]: 'Slow Moving'}
        data_agregasi['Kategori'] = data_agregasi['Cluster'].map(mapping)

        # --- TAMPILAN DASHBOARD ---
        counts = data_agregasi['Kategori'].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Fast Moving", f"{counts.get('Fast Moving', 0)} Obat")
        c2.metric("🟠 Medium Moving", f"{counts.get('Medium Moving', 0)} Obat")
        c3.metric("🔵 Slow Moving", f"{counts.get('Slow Moving', 0)} Obat")

        st.divider()
        
        # Visualisasi 3D
        st.subheader("📈 Sebaran Data Berdasarkan Kategori")
        fig = px.scatter_3d(data_agregasi, x='Frekuensi Transaksi', y='Volume Penjualan', z='Nilai Transaksi',
                            color='Kategori', hover_name='Nama Obat',
                            color_discrete_map={'Fast Moving': 'red', 'Medium Moving': 'orange', 'Slow Moving': 'blue'})
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --- TABEL PER KATEGORI ---
        st.subheader("📋 Daftar Obat Per Kategori")
        t1, t2, t3 = st.tabs(["🔴 Fast", "🟠 Medium", "🔵 Slow"])

        with t1:
            st.dataframe(data_agregasi[data_agregasi['Kategori'] == 'Fast Moving'][['Nama Obat', 'Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']], use_container_width=True)
        with t2:
            st.dataframe(data_agregasi[data_agregasi['Kategori'] == 'Medium Moving'][['Nama Obat', 'Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']], use_container_width=True)
        with t3:
            st.dataframe(data_agregasi[data_agregasi['Kategori'] == 'Slow Moving'][['Nama Obat', 'Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']], use_container_width=True)

        # Tombol Download
        csv = data_agregasi.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("📥 Download Hasil (.csv)", data=csv, file_name='Hasil_Kategori_Apotek.csv')

    except Exception as e:
        st.error(f"Format file tidak sesuai: {e}")
else:
    st.info("Silakan unggah file Excel untuk melihat pembagian kategori otomatis.")
