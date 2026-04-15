import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from sklearn.cluster import KMeans

# 1. Konfigurasi Tampilan
st.set_page_config(page_title="Kategori Obat Apotek Anugerah", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌿 Sistem Kategori Obat</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Apotek Anugerah Bekasi - Penentuan Kategori Fast, Medium, & Slow Moving</p>", unsafe_allow_html=True)
st.divider()

# 2. Sidebar
st.sidebar.header("Pusat Kontrol")
uploaded_file = st.sidebar.file_uploader("Drag & Drop File Excel Penjualan", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    try:
        # --- DATA PREPARATION ---
        data_agregasi = df.groupby('Nama Obat', sort=False).agg({
            'Tanggal Transaksi': 'count',
            'Jumlah Terjual': 'sum',
            'Total Harga': 'sum'
        }).reset_index()

        data_agregasi.columns = ['Nama Obat', 'Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']

        # --- DATA CLEANING ---
        data_agregasi.dropna(inplace=True)
        data_agregasi.drop_duplicates(inplace=True)

        # --- TRANSFORMASI & NORMALISASI ---
        fitur_numerik = ['Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']
        scaler = MinMaxScaler()
        data_normalisasi = scaler.fit_transform(data_agregasi[fitur_numerik])
        X_modeling = pd.DataFrame(data_normalisasi, columns=fitur_numerik)

        # --- MODELING (K-MEANS) ---
        model_kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        data_agregasi['Cluster'] = model_kmeans.fit_predict(X_modeling)

        # --- PROFILING & PELABELAN ---
        centroid = data_agregasi.groupby('Cluster')['Frekuensi Transaksi'].mean().sort_values(ascending=False).index
        mapping_kategori = {centroid[0]: 'Fast Moving', centroid[1]: 'Medium Moving', centroid[2]: 'Slow Moving'}
        data_agregasi['Kategori'] = data_agregasi['Cluster'].map(mapping_kategori)

        # --- TAMPILAN DASHBOARD ---
        counts = data_agregasi['Kategori'].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("🔥 Total Fast Moving", f"{counts.get('Fast Moving', 0)} Obat")
        c2.metric("⚡ Total Medium Moving", f"{counts.get('Medium Moving', 0)} Obat")
        c3.metric("🧊 Total Slow Moving", f"{counts.get('Slow Moving', 0)} Obat")

        st.divider()

        # --- BAGIAN VISUALISASI ---
        st.subheader("📈 Visualisasi Sebaran Kategori")
        fig = px.scatter_3d(data_agregasi, 
                            x='Frekuensi Transaksi', y='Volume Penjualan', z='Nilai_Transaksi' if 'Nilai_Transaksi' in data_agregasi else 'Nilai Transaksi',
                            color='Kategori', hover_name='Nama Obat',
                            color_discrete_map={'Fast Moving': 'red', 'Medium Moving': 'orange', 'Slow Moving': 'blue'},
                            opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --- BAGIAN 3 TABEL KATEGORI (REVISI) ---
        st.subheader("📋 Daftar Obat Per Kategori")
        
        col_fast, col_med, col_slow = st.tabs(["🔵 Fast Moving", "🟠 Medium Moving", "🔴 Slow Moving"])

        with col_fast:
            st.success("Daftar Obat Prioritas Tinggi (Penjualan Paling Cepat)")
            df_fast = data_agregasi[data_agregasi['Kategori'] == 'Fast Moving']
            st.dataframe(df_fast[['Nama Obat', 'Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']], use_container_width=True)

        with col_med:
            st.warning("Daftar Obat Stok Stabil")
            df_med = data_agregasi[data_agregasi['Kategori'] == 'Medium Moving']
            st.dataframe(df_med[['Nama Obat', 'Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']], use_container_width=True)

        with col_slow:
            st.info("Daftar Obat Penjualan Lambat (Evaluasi Stok)")
            df_slow = data_agregasi[data_agregasi['Kategori'] == 'Slow Moving']
            st.dataframe(df_slow[['Nama Obat', 'Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']], use_container_width=True)

        # Tombol Download
        csv = data_agregasi.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil Kategori (.csv)", data=csv, file_name='Hasil_Kategori_Apotek.csv')

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}. Pastikan file Excel sesuai dengan format mentah.")
else:
    st.info("👋 Silakan masukkan file Excel untuk melihat pembagian kategori obat.")
