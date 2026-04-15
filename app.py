import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# 1. Konfigurasi Tampilan
st.set_page_config(page_title="FSM APOTEK ANUGRAH BEKASI", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌿 Kategori Obat</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Apotek Anugerah Bekasi - Klasifikasi Fast moving, Medium moving, & Slow Moving</p>", unsafe_allow_html=True)
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
        X = df_raw[['Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']]
        
        # Normalisasi data baru secara otomatis
        X_scaled = scaler.fit_transform(X)
        
        # Prediksi menggunakan model yang sudah dilatih
        df_raw['Cluster'] = model_kmeans.predict(X_scaled)
        
        # Mapping Kategori (Dinamis berdasarkan hasil cluster)
        centers = df_raw.groupby('Cluster')['Frekuensi'].mean().sort_values(ascending=False).index
        mapping = {centers[0]: 'Fast Moving', centers[1]: 'Medium Moving', centers[2]: 'Slow Moving'}
        df_raw['Kategori'] = df_raw['Cluster'].map(mapping)
        
        # --- TAMPILAN DASHBOARD ---
        
        # Box Angka Ringkasan (Menyesuaikan jumlah data yang masuk)
        counts = df_raw['Kategori'].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("🔥 Fast Moving", f"{counts.get('Fast Moving', 0)} Obat")
        c2.metric("⚡ Medium Moving", f"{counts.get('Medium Moving', 0)} Obat")
        c3.metric("🧊 Slow Moving", f"{counts.get('Slow Moving', 0)} Obat")
        
        st.divider()
        
        # Tab Visualisasi & Tabel
        tab1, tab2 = st.tabs(["📈 Visualisasi Sebaran", "📄 Data Hasil Klasifikasi"])
        
        with tab1:
            fig = px.scatter_3d(df_raw, x='Frekuensi Transaksi', y='Volume Penjualan', z='Nilai Transaksi',
                                color='Kategori', hover_name='Nama Obat',
                                color_discrete_map={'Fast Moving':'blue', 'Medium Moving':'orange', 'Slow Moving':'red'})
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.write("### Daftar Obat Sudah Di Kategorikan")
            pilihan = st.multiselect("Filter Kategori:", ['Fast Moving', 'Medium Moving', 'Slow Moving'], default=['Fast Moving'])
            st.dataframe(df_raw[df_raw['Kategori'].isin(pilihan)], use_container_width=True)
            
        # Tombol Download Hasil Dinamis
        st.download_button("Download Hasil (.xlsx)", 
                           data=df_raw.to_csv(index=False).encode('utf-8'),
                           file_name='Hasil_Kategori_Terbaru.csv')

    except Exception as e:
        st.error(f"Terjadi kesalahan format: {e}. Pastikan kolom file sesuai (Nama Obat, Frekuensi Transaksi, Volume Penjualan, Nilai Transaksi)")
else:
    st.info("👋 Halo! Silakan masukkan file Excel penjualan Apotek untuk memulai analisis otomatis.")