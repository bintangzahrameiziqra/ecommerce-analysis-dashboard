import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide"
)

sns.set_theme(style="whitegrid")

# =========================
# Custom CSS
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #F8FAFC;
    }

    .title-text {
        font-size: 38px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 5px;
    }

    .subtitle-text {
        font-size: 16px;
        color: #64748B;
        margin-bottom: 25px;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        border: 1px solid #E2E8F0;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #1E293B;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .insight-box {
        background-color: #EFF6FF;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #2563EB;
        color: #1E293B;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "main_data.csv")

    df = pd.read_csv(data_path)

    datetime_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


all_df = load_data()

# =========================
# Helper Functions
# =========================
def create_daily_orders_df(df):
    daily_orders_df = (
        df.resample(rule="D", on="order_purchase_timestamp")
        .agg(
            order_count=("order_id", "nunique"),
            total_payment_value=("payment_value", "sum")
        )
        .reset_index()
    )
    return daily_orders_df


def create_rfm_df(df):
    reference_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm_df = (
        df.groupby("customer_unique_id", as_index=False)
        .agg(
            last_order_date=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("payment_value", "sum")
        )
    )

    rfm_df["recency"] = (reference_date - rfm_df["last_order_date"]).dt.days

    # Menggunakan rank agar qcut lebih stabil
    rfm_df["r_score"] = pd.qcut(
        rfm_df["recency"].rank(method="first"),
        5,
        labels=[5, 4, 3, 2, 1]
    ).astype(int)

    rfm_df["f_score"] = pd.qcut(
        rfm_df["frequency"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm_df["m_score"] = pd.qcut(
        rfm_df["monetary"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm_df["rfm_total_score"] = rfm_df[["r_score", "f_score", "m_score"]].sum(axis=1)

    def segment_customer(score):
        if score >= 13:
            return "High Value"
        elif score >= 10:
            return "Loyal Customer"
        elif score >= 7:
            return "Potential Loyalist"
        else:
            return "At Risk"

    rfm_df["segment"] = rfm_df["rfm_total_score"].apply(segment_customer)

    return rfm_df


def create_segment_summary(rfm_df):
    segment_summary = (
        rfm_df.groupby("segment", as_index=False)
        .agg(
            total_customers=("customer_unique_id", "nunique"),
            total_payment_value=("monetary", "sum"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean")
        )
        .sort_values("total_payment_value", ascending=False)
    )
    return segment_summary


def create_state_summary(df):
    state_summary = (
        df.groupby("customer_state", as_index=False)
        .agg(
            total_payment_value=("payment_value", "sum"),
            total_transactions=("order_id", "nunique"),
            total_customers=("customer_unique_id", "nunique")
        )
        .sort_values("total_payment_value", ascending=False)
    )

    payment_median = state_summary["total_payment_value"].median()
    transaction_median = state_summary["total_transactions"].median()

    state_summary["performance_group"] = np.select(
        [
            (state_summary["total_payment_value"] >= payment_median) &
            (state_summary["total_transactions"] >= transaction_median),

            (state_summary["total_payment_value"] >= payment_median) &
            (state_summary["total_transactions"] < transaction_median),

            (state_summary["total_payment_value"] < payment_median) &
            (state_summary["total_transactions"] >= transaction_median)
        ],
        [
            "High Value - High Frequency",
            "High Value - Low Frequency",
            "Low Value - High Frequency"
        ],
        default="Low Value - Low Frequency"
    )

    return state_summary


def create_city_summary(df):
    city_summary = (
        df.groupby(["customer_state", "customer_city"], as_index=False)
        .agg(
            total_payment_value=("payment_value", "sum"),
            total_transactions=("order_id", "nunique"),
            avg_lat=("geolocation_lat", "mean"),
            avg_lng=("geolocation_lng", "mean")
        )
        .sort_values("total_payment_value", ascending=False)
    )
    return city_summary


# =========================
# Sidebar Filter
# =========================
min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()

with st.sidebar:
    st.title("🛒 Dashboard Filter")
    st.write("Gunakan filter berikut untuk menyesuaikan periode analisis.")

    start_date, end_date = st.date_input(
        label="Rentang Waktu",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

    st.markdown("---")
    st.write("Dataset: **E-Commerce Public Dataset**")
    st.write("Analisis: **RFM & Regional Performance**")

main_df = all_df[
    (all_df["order_purchase_timestamp"].dt.date >= start_date) &
    (all_df["order_purchase_timestamp"].dt.date <= end_date)
]

# =========================
# Prepare Data
# =========================
daily_orders_df = create_daily_orders_df(main_df)
rfm_df = create_rfm_df(main_df)
segment_summary = create_segment_summary(rfm_df)
state_summary = create_state_summary(main_df)
city_summary = create_city_summary(main_df)

# =========================
# Header
# =========================
st.markdown('<div class="title-text">E-Commerce Public Dataset Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Dashboard analisis segmentasi pelanggan RFM dan performa transaksi berdasarkan wilayah.</div>',
    unsafe_allow_html=True
)

# =========================
# Metric Cards
# =========================
total_orders = main_df["order_id"].nunique()
total_customers = main_df["customer_unique_id"].nunique()
total_payment = main_df["payment_value"].sum()
avg_payment = main_df["payment_value"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Orders", f"{total_orders:,}")

with col2:
    st.metric("Total Customers", f"{total_customers:,}")

with col3:
    st.metric("Total Payment Value", format_currency(total_payment, "BRL", locale="pt_BR"))

with col4:
    st.metric("Average Payment", format_currency(avg_payment, "BRL", locale="pt_BR"))

st.markdown("---")

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Overview",
        "👥 RFM Segmentation",
        "📍 Regional Analysis",
        "✅ Insight & Recommendation"
    ]
)

# =========================
# Tab 1 - Overview
# =========================
with tab1:
    st.markdown('<div class="section-title">Daily Transaction Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            daily_orders_df["order_purchase_timestamp"],
            daily_orders_df["order_count"],
            linewidth=2,
            marker="o",
            markersize=3
        )
        ax.set_title("Jumlah Order Harian", fontsize=16, fontweight="bold")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Jumlah Order")
        plt.xticks(rotation=30)
        st.pyplot(fig)

    with col2:
        st.markdown(
            """
            <div class="insight-box">
            <b>Overview</b><br><br>
            Grafik ini menunjukkan perubahan jumlah order harian selama periode yang dipilih.
            Melalui grafik ini, pola aktivitas transaksi dapat dilihat secara lebih jelas.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(
            daily_orders_df.sort_values("order_count", ascending=False).head(10),
            use_container_width=True
        )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        daily_orders_df["order_purchase_timestamp"],
        daily_orders_df["total_payment_value"],
        linewidth=2
    )
    ax.set_title("Total Payment Value Harian", fontsize=16, fontweight="bold")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Total Payment Value")
    plt.xticks(rotation=30)
    st.pyplot(fig)

# =========================
# Tab 2 - RFM
# =========================
with tab2:
    st.markdown('<div class="section-title">RFM Customer Segmentation</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Average Recency", f"{rfm_df['recency'].mean():.1f} hari")

    with col2:
        st.metric("Average Frequency", f"{rfm_df['frequency'].mean():.2f}")

    with col3:
        st.metric("Average Monetary", format_currency(rfm_df["monetary"].mean(), "BRL", locale="pt_BR"))

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=segment_summary,
            x="segment",
            y="total_payment_value",
            ax=ax
        )
        ax.set_title("Total Payment Value per Segmen RFM", fontsize=14, fontweight="bold")
        ax.set_xlabel("Segmen Pelanggan")
        ax.set_ylabel("Total Payment Value")
        plt.xticks(rotation=15)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            segment_summary["total_customers"],
            labels=segment_summary["segment"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax.set_title("Distribusi Customer Berdasarkan Segmen RFM", fontsize=14, fontweight="bold")
        st.pyplot(fig)

    st.markdown(
        """
        <div class="insight-box">
        <b>Insight RFM:</b><br>
        Segmentasi RFM membantu melihat pelanggan berdasarkan seberapa baru mereka bertransaksi,
        seberapa sering mereka melakukan transaksi, dan seberapa besar nilai transaksi mereka.
        Segmen dengan kontribusi payment value tinggi dapat menjadi prioritas strategi retensi.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Ringkasan Segmen RFM")
    st.dataframe(segment_summary, use_container_width=True)

# =========================
# Tab 3 - Regional Analysis
# =========================
with tab3:
    st.markdown('<div class="section-title">Regional Performance Analysis</div>', unsafe_allow_html=True)

    top_states = state_summary.head(10)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=top_states,
            x="total_payment_value",
            y="customer_state",
            ax=ax
        )
        ax.set_title("Top 10 State Berdasarkan Payment Value", fontsize=14, fontweight="bold")
        ax.set_xlabel("Total Payment Value")
        ax.set_ylabel("State")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=top_states.sort_values("total_transactions", ascending=False),
            x="total_transactions",
            y="customer_state",
            ax=ax
        )
        ax.set_title("Top 10 State Berdasarkan Jumlah Transaksi", fontsize=14, fontweight="bold")
        ax.set_xlabel("Total Transaksi")
        ax.set_ylabel("State")
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=state_summary,
        x="total_transactions",
        y="total_payment_value",
        hue="performance_group",
        s=130,
        ax=ax
    )
    ax.set_title("Hubungan Total Transaksi dan Total Payment Value per State", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Transaksi")
    ax.set_ylabel("Total Payment Value")
    ax.legend(title="Kategori Performa", bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig)

    st.markdown(
        """
        <div class="insight-box">
        <b>Insight Wilayah:</b><br>
        State dengan jumlah transaksi tinggi cenderung memiliki total payment value yang tinggi.
        Analisis ini dapat membantu menentukan wilayah utama yang perlu dipertahankan dan wilayah lain
        yang masih memiliki peluang untuk dikembangkan.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Ringkasan Performa State")
    st.dataframe(state_summary, use_container_width=True)

    st.subheader("Sebaran Kota Berdasarkan Payment Value")

    geo_city_summary = city_summary.dropna(subset=["avg_lat", "avg_lng"]).head(200)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        geo_city_summary["avg_lng"],
        geo_city_summary["avg_lat"],
        s=geo_city_summary["total_payment_value"] / 300,
        c=geo_city_summary["total_payment_value"],
        alpha=0.6
    )
    plt.colorbar(scatter, label="Total Payment Value")
    ax.set_title("Sebaran Wilayah Berdasarkan Total Payment Value", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)

# =========================
# Tab 4 - Insight & Recommendation
# =========================
with tab4:
    st.markdown('<div class="section-title">Insight & Recommendation</div>', unsafe_allow_html=True)

    st.subheader("Conclusion Pertanyaan 1")
    st.write(
        """
        Berdasarkan hasil analisis RFM, pelanggan dalam dataset didominasi oleh segmen
        Potential Loyalist dan Loyal Customer. Segmen Potential Loyalist memiliki jumlah pelanggan
        paling banyak, sedangkan kontribusi terbesar terhadap total payment value berasal dari
        segmen Loyal Customer. Hal ini menunjukkan bahwa pelanggan yang lebih loyal cenderung
        memberikan kontribusi nilai transaksi yang lebih besar bagi bisnis.
        """
    )

    st.write(
        """
        Segmen High Value juga menjadi segmen yang perlu diperhatikan. Walaupun jumlah pelanggannya
        relatif lebih kecil, segmen ini memiliki rata-rata nilai transaksi yang tinggi. Sementara itu,
        segmen At Risk menunjukkan tanda-tanda mulai tidak aktif karena memiliki nilai recency yang
        lebih tinggi dan kontribusi payment value yang lebih rendah.
        """
    )

    st.subheader("Conclusion Pertanyaan 2")
    st.write(
        """
        Berdasarkan hasil analisis wilayah, distribusi transaksi dan payment value tidak tersebar
        secara merata antar state. State SP menjadi wilayah dengan kontribusi terbesar, baik dari sisi
        jumlah transaksi maupun total payment value, kemudian diikuti oleh RJ dan MG.
        """
    )

    st.write(
        """
        Hasil analisis juga menunjukkan bahwa wilayah dengan jumlah transaksi yang tinggi cenderung
        memiliki total payment value yang tinggi pula. Dengan kata lain, semakin tinggi aktivitas
        pembelian di suatu wilayah, maka semakin besar juga kontribusi payment value yang dihasilkan
        oleh wilayah tersebut.
        """
    )

    st.subheader("Recommendation / Action Item")
    st.markdown(
        """
        1. Bisnis dapat memprioritaskan strategi retensi pada segmen **Loyal Customer** karena segmen ini memberikan kontribusi terbesar terhadap total payment value.

        2. Segmen **Potential Loyalist** dapat didorong agar berkembang menjadi Loyal Customer melalui promo pembelian ulang, reminder pembelian, atau rekomendasi produk.

        3. Segmen **High Value** perlu dipertahankan karena memiliki rata-rata nilai transaksi yang tinggi walaupun jumlah pelanggannya tidak sebanyak segmen lain.

        4. Segmen **At Risk** perlu menjadi perhatian karena menunjukkan tanda-tanda mulai tidak aktif, sehingga dapat diberikan kampanye reaktivasi.

        5. State **SP, RJ, dan MG** dapat dipertahankan sebagai pasar utama karena memiliki kontribusi transaksi dan payment value tertinggi.

        6. Wilayah dengan kontribusi lebih rendah dapat dianalisis lebih lanjut untuk melihat peluang pertumbuhan pasar.
        """
    )

st.caption("Copyright © 2026 | E-Commerce Public Dataset Dashboard")