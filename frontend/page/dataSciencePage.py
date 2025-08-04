from packages import *
from utils.contants import BASE_URL
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    response = requests.get(f'{BASE_URL}/customers')
    return pd.DataFrame(response.json())

@st.cache_data
def load_reviews():
    response = requests.get(f'{BASE_URL}/reviews')
    return pd.DataFrame(response.json())

reviews = [] # load_reviews()
data = load_data()


def dataSciencePage():
    st.title("Data Sains")
    st.subheader("Analisa Sentimen, Segmentasi Pelanggan, dan Rekomendasi Produk menggunakan Machine Learning.")
    st.write("Data Statistik Pelanggan")
    st.dataframe(data)
    # **📌 Statistik Umum**
    col1, col2 = st.columns(2)
    col3, col4 = st.columns([3,1])
    col5, col6 = st.columns(2)
    col7, col8 = st.columns([3,1])
    col1.metric("Rata-rata Usia", f"{data['Age'].mean():.1f}")
    col3.metric("Pendapatan Rata-rata Total selama setahun", f"Rp. {data['Income'].mean():,.0f}", width=1000)
    col5.metric("Rata-rata Spending Score", f"{data['SpendingScore'].mean():.1f}")
    col6.metric("Rata-rata Loyalty Score", f"{data['LoyaltyScore'].mean():.2f}")
    col7.metric("Rata-rata CLV", f"Rp. {data['CLV'].mean():,.0f} ")

    # **📈 Visualisasi Cluster**
    st.subheader("📊 A. Segmentasi Pelanggan (Clustering)")
    fig = px.scatter(data, x="Income", y="SpendingScore", color=data["Cluster"].astype(str),
                    hover_data=["Name", "Age"], title="Grafik : Segmentasi Pelanggan menggunakan K-Means Algorithm")
    st.plotly_chart(fig)

    # **🔮 Prediksi Cluster untuk Pelanggan Baru**
    st.write("##### Contoh : Prediksi Segmentasi Pelanggan Baru")

    age = st.number_input("Masukkan Umur", min_value=19, max_value=60, step=1, key="1")
    income = st.number_input("Masukkan Total Pendapatan Tahunan", min_value=50000000, max_value=1000000000, step=1000, key="2")
    spending = st.number_input("Masukkan Spending Score", min_value=50, max_value=100, step=1, key="3")

    if st.button("Prediksi Cluster", key="btn_cluster"):
        response = requests.post(f'{BASE_URL}/predict', json={"Age": age, "Income": income, "SpendingScore": spending})
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Pelanggan termasuk dalam Cluster: {result['Cluster']}")
        else:
            st.error("Gagal mendapatkan prediksi.")


    # **📈 Visualisasi Loyalty Score & CLV**
    st.subheader("📈 B. Tingkat Loyalitas Customer")
    st.write("LoyaltyScore adalah Skor berdasarkan jumlah transaksi & nilai total pembelian")
    st.write("Customer Lifetime Value didapat dari loyality score dikali pendapatan dan dibagi perseribu")
    fig = px.scatter(data, x="LoyaltyScore", y="CLV", color=data["Churn"].astype(str),
                    hover_data=["Name", "Age"], title="Grafik : Tingkat Loyalitas Customer")
    st.plotly_chart(fig)

    # **📩 Download Laporan PDF**
    st.subheader("📜C. Laporan Analisis Pelanggan")
    if st.button("Download Laporan PDF"):
        pdf_url = f'{BASE_URL}/report'
        st.markdown(f"[Klik di sini untuk mendownload laporan]({pdf_url})")

    

    # **🔮 Prediksi Resiko untuk Pelanggan Baru**
    st.subheader("🔮D. Prediksi Churn (Tingkat Resiko Pelanggan)")

    age_churn = st.number_input("Masukkan Umur", min_value=19, max_value=60, step=1, key="4")
    income_churn = st.number_input("Masukkan Pendapatan Tahunan", min_value=50000000, max_value=1000000000, step=1000, key="5")
    spending_churn = st.number_input("Masukkan Spending Score", min_value=50, max_value=100, step=1, key="6")

    if st.button("Prediksi Churn", key="btn_churn"):
        response = requests.post(f'{BASE_URL}/predict_churn', json={"Age": age_churn, "Income": income_churn, "SpendingScore": spending_churn})
        
        if response.status_code == 200:
            result = response.json()
            print(result)
            churn_prob = result["Churn_Probability"]
            if churn_prob > 0.5:
                st.error(f"Pelanggan memiliki tingkat resiko tinggi: {churn_prob:.2%}")
            else:
                st.success(f"Pelanggan memiliki tingkat resiko rendah: {churn_prob:.2%}")
        else:
            st.error("Gagal mendapatkan prediksi.")

    # **📌 Tampilkan Ulasan Pelanggan**
    st.subheader("🔮E. Ulasan Pelanggan & Analisis Sentimen")
    st.dataframe(reviews)
    # **🔍 Analisis Sentimen untuk Review Baru**
    st.write("🔍 Cek Sentimen Ulasan")

    review_text = st.text_area("Masukkan Ulasan Pelanggan", key="7")

    if st.button("Analisis Sentimen", key="8"):
        response = requests.post(f'{BASE_URL}/analyze_review', json={"Review": review_text})
        
        if response.status_code == 200:
            result = response.json()
            sentiment = result["Sentiment"]
            
            if sentiment == "Positif":
                st.success(f"Sentimen: {sentiment} 😊")
            elif sentiment == "Negatif":
                st.error(f"Sentimen: {sentiment} 😞")
            else:
                st.warning(f"Sentimen: {sentiment} 😐")
        else:
            st.error("Gagal menganalisis ulasan.")

    # **📌 Rekomendasi Produk**
    st.subheader("🛍 F. Rekomendasi Produk")

    customer_name = st.text_input("Masukkan Nama Pelanggan", key="9")

    if st.button("Dapatkan Rekomendasi", key="10"):
        response = requests.post(f'{BASE_URL}/recommend', json={"Name": customer_name})
        
        if response.status_code == 200:
            result = response.json()
            print(result)
            recommendations = result["Recommendations"]
            
            if recommendations:
                st.success(f"Produk yang direkomendasikan untuk {customer_name}: {', '.join(recommendations)}")
            else:
                st.warning(f"Tidak ada rekomendasi untuk {customer_name}.")
        else:
            st.error("Gagal mendapatkan rekomendasi.")