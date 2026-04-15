import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# 1. Konfigurasi Tampilan
st.set_page_config(page_title="FMS Apotek Anugerah", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌿 Kategori Obat</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Apotek Anugerah Bekasi - Kategori Fast Moving, Medium Moving, & Slow Moving</p>", unsafe_allow_html=True)
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
       # data understanding
import pandas as pd
import io
from google.colab import files


print("Silakan unggah file Excel atau CSV data penjualan Anda:")
uploaded = files.upload()


file_name = list(uploaded.keys())[0]


if file_name.endswith('.csv'):
    df = pd.read_csv(io.BytesIO(uploaded[file_name]))
elif file_name.endswith(('.xls', '.xlsx')):
    df = pd.read_excel(io.BytesIO(uploaded[file_name]))
else:
    print("Format file tidak didukung. Harap unggah .csv atau .xlsx")

print(f"\nBerhasil memuat file: {file_name}")
print(df.head())

# data preparation (agregasi)
data_agregasi = df.groupby('Nama Obat', sort=False).agg({
    'Tanggal Transaksi': 'count',    # Menghitung Frekuensi Transaksi secara TOTAL
    'Jumlah Terjual': 'sum',         # Menghitung Volume Penjualan secara TOTAL
    'Total Harga': 'sum'             # Menghitung Nilai Transaksi secara TOTAL
}).reset_index()


data_agregasi.columns = ['Nama Obat', 'Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']


from google.colab import data_table
data_table.enable_dataframe_formatter()

print(f"Total terdapat {len(data_agregasi)} jenis obat yang berhasil diagregasi.")
display(data_agregasi)


# data preparation (cleaning)
# 1. pengecekan missing value
missing_data = data_agregasi.isnull().sum()
total_data = len(data_agregasi)

print("--- Pengecekan Missing Value ---")
for col in ['Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']:
    n_missing = missing_data[col]
    if n_missing > 0:
        persentase = (n_missing / total_data) * 100
        print(f"Kolom {col}: ditemukan {n_missing} data kosong ({persentase:.2f}%)")

        # drop jika < 5%, imputasi mean jika >= 5%
        if persentase < 5:
            data_agregasi.dropna(subset=[col], inplace=True)
            print(f"Tindakan: Menghapus baris (Drop Rows) karena missing value < 5%")
        else:
            data_agregasi[col] = data_agregasi[col].fillna(data_agregasi[col].mean())
            print(f"Tindakan: Imputasi Nilai Rata-rata (Mean Imputation) karena missing value >= 5%")
    else:
        print(f"Kolom {col}: Bersih (0 missing value)")

# 2. pengecekan data duplikat
jumlah_duplikat = data_agregasi.duplicated().sum()
if jumlah_duplikat > 0:
    data_agregasi.drop_duplicates(inplace=True)
    print(f"\n--- Pengecekan Duplikat ---\nBerhasil menghapus {jumlah_duplikat} data duplikat.")
else:
    print("\n--- Pengecekan Duplikat ---\nTidak ditemukan data duplikat.")

# 3. menampilkan data yang telah dibersihkan
print(f"\nTotal data setelah pembersihan: {len(data_agregasi)} item obat.")
display(data_agregasi.head())

# transformasi dan normalisasi data

# 1. tranformasi data
fitur_numerik = ['Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']
data_transformasi = data_agregasi[fitur_numerik]

# 2. normalisasi data (min - max scaling)
#untuk menyamakan skala antar variabel (rentang 0 sampai 1)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_normalisasi = scaler.fit_transform(data_transformasi)

df_normalized = pd.DataFrame(data_normalisasi, columns=fitur_numerik)

df_normalized.insert(0, 'Nama Obat', data_agregasi['Nama Obat'].values)

print("--- Tahap Transformasi & Normalisasi Selesai ---")
print("Data telah diubah ke bentuk numerik dan diskalakan (0-1).")
display(df_normalized.head(10))

# modeling (K-Means)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

X_modeling = df_normalized[['Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']]

# 2. Penentuan K Optimal (Metode Elbow)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_modeling)
    wcss.append(kmeans.inertia_)

# Visualisasi Grafik Elbow
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('Metode Elbow - Penentuan Jumlah Cluster Optimal')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# 3. Proses Clustering (Euclidean Distance)
# Berdasarkan grafik Elbow, gunakan k=3
k_optimal = 3
model_kmeans = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42)
df_normalized['Cluster'] = model_kmeans.fit_predict(X_modeling)

# 4. Profiling & Pelabelan (Fast, Medium, Slow Moving)
# Menghitung rata-rata (centroid) untuk menentukan label
centroid = df_normalized.groupby('Cluster')[['Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']].mean()

# Mengurutkan klaster berdasarkan Frekuensi tertinggi ke terendah
urutan = centroid['Frekuensi Transaksi'].sort_values(ascending=False).index
mapping_kategori = {
    urutan[0]: 'Fast Moving',
    urutan[1]: 'Medium Moving',
    urutan[2]: 'Slow Moving'
}

# Menerapkan label ke dataset
df_normalized['Kategori'] = df_normalized['Cluster'].map(mapping_kategori)

# output hasil

print("\n" + "="*40)
print("HASIL ANALISIS K-MEANS (N=731 OBAT)")
print("="*40)

print("\n[1] Sebaran Kategori Obat:")
print(df_normalized['Kategori'].value_counts())

print("\n[2] Rata-rata Karakteristik per Kategori (Centroid):")
display(df_normalized.groupby('Kategori')[['Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']].mean())

# Menampilkan Sampel Data Final
print("\n[3] Sampel 10 Hasil Akhir Pengelompokan:")

display(df_normalized[['Nama Obat', 'Kategori', 'Frekuensi Transaksi', 'Volume Penjualan', 'Nilai Transaksi']].head(10))

# evaluasi (SILHOUETTE SCORE)
from sklearn.metrics import silhouette_score

# Menghitung nilai Silhouette Score untuk K=3
score = silhouette_score(X_modeling, df_normalized['Cluster'])

print(f"--- Hasil Evaluasi Model ---")
print(f"Silhouette Score untuk K=3: {score:.4f}")

# Interpretasi Skor
if score > 0.7:
    print("Interpretasi: Struktur Cluster Sangat Kuat")
elif score > 0.5:
    print("Interpretasi: Struktur Cluster Kuat (Baik)")
elif score > 0.25:
    print("Interpretasi: Struktur Cluster Cukup Kuat")
else:
    print("Interpretasi: Struktur Cluster Lemah")

# VISUALISASI 3D
import plotly.express as px

fig = px.scatter_3d(df_normalized,
                    x='Frekuensi Transaksi',
                    y='Volume Penjualan',
                    z='Nilai Transaksi',
                    color='Kategori',
                    hover_name='Nama Obat',
                    title='Visualisasi 3D Pengelompokan Obat Apotek Anugerah',

                    category_orders={"Kategori": ["Fast Moving", "Medium Moving", "Slow Moving"]},
                    color_discrete_map={
                        'Fast Moving': 'red',
                        'Medium Moving': 'orange',
                        'Slow Moving': 'blue'
                    },
                    opacity=0.7)

fig.update_layout(scene = dict(
                    xaxis_title='Frekuensi Transaksi',
                    yaxis_title='Volume Penjualan',
                    zaxis_title='Nilai Transaksi'))

fig.show()
