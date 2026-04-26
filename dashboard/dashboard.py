import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
import numpy as np

sns.set(style="darkgrid")

st.set_page_config(
    page_title="E-Commerce Analysis Dashboard",
    page_icon="🛒",
    layout="wide"
)

# =========================
# Helper Function
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("dashboard/main_data.csv")

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

    rfm_df["r_score"] = pd.qcut(
        rfm_df["recency"],
        5,
        labels=[5, 4, 3, 2, 1],
        duplicates="drop"
    ).astype(int)

    rfm_df["f_score"] = pd.qcut(
        rfm_df["frequency"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop"
    ).astype(int)

    rfm_df["m_score"] = pd.qcut(
        rfm_df["monetary"],
        5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop"
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
            (state_summary["total_transactions"] < transaction_median),

            (state_summary["total_payment_value"] >= payment_median) &
            (state_summary["total_transactions"] >= transaction_median),

            (state_summary["total_payment_value"] < payment_median) &
            (state_summary["total_transactions"] >= transaction_median)
        ],
        [
            "High Value - Low Frequency",
            "High Value - High Frequency",
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
# Load Data
# =========================

all_df = load_data()

min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()

# =========================
# Sidebar
# =========================

with st.sidebar:
    st.title("Filter Dashboard")
    st.write("Gunakan filter berikut untuk melihat performa transaksi berdasarkan periode tertentu.")

    start_date, end_date = st.date_input(
        label="Rentang Waktu",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = all_df[
    (all_df["order_purchase_timestamp"].dt.date >= start_date) &
    (all_df["order_purchase_timestamp"].dt.date <= end_date)
]

# =========================
# Prepare DataFrame
# =========================

daily_orders_df = create_daily_orders_df(main_df)
rfm_df = create_rfm_df(main_df)
segment_summary = create_segment_summary(rfm_df)
state_summary = create_state_summary(main_df)
city_summary = create_city_summary(main_df)

# =========================
# Dashboard Title
# =========================

st.title("🛒 E-Commerce Public Dataset Dashboard")
st.write(
    "Dashboard ini menampilkan hasil analisis transaksi e-commerce berdasarkan "
    "segmentasi pelanggan RFM dan distribusi payment value berdasarkan wilayah."
)

# =========================
# Metric
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_orders = main_df["order_id"].nunique()
    st.metric("Total Orders", value=total_orders)

with col2:
    total_customers = main_df["customer_unique_id"].nunique()
    st.metric("Total Customers", value=total_customers)

with col3:
    total_payment = format_currency(
        main_df["payment_value"].sum(),
        "BRL",
        locale="pt_BR"
    )
    st.metric("Total Payment Value", value=total_payment)

with col4:
    avg_payment = format_currency(
        main_df["payment_value"].mean(),
        "BRL",
        locale="pt_BR"
    )
    st.metric("Average Payment", value=avg_payment)

# =========================
# Daily Orders
# =========================

st.subheader("Daily Orders and Payment Value")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(
    daily_orders_df["order_purchase_timestamp"],
    daily_orders_df["order_count"],
    marker="o",
    linewidth=1
)
ax.set_title("Jumlah Order Harian")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Jumlah Order")
plt.xticks(rotation=30)
st.pyplot(fig)

# =========================
# RFM Analysis
# =========================

st.subheader("RFM Customer Segmentation")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=segment_summary,
        x="segment",
        y="total_payment_value",
        ax=ax
    )
    ax.set_title("Total Payment Value per Segmen RFM")
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
    ax.set_title("Distribusi Customer Berdasarkan Segmen RFM")
    st.pyplot(fig)

st.write("Ringkasan Segmen RFM")
st.dataframe(segment_summary)

# =========================
# Regional Analysis
# =========================

st.subheader("Regional Performance Analysis")

top_states = state_summary.head(10)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    data=top_states,
    x="customer_state",
    y="total_payment_value",
    ax=ax
)
ax.set_title("Top 10 State Berdasarkan Total Payment Value")
ax.set_xlabel("State")
ax.set_ylabel("Total Payment Value")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=state_summary,
    x="total_transactions",
    y="total_payment_value",
    hue="performance_group",
    s=120,
    ax=ax
)
ax.set_title("Hubungan Total Transaksi dan Total Payment Value per State")
ax.set_xlabel("Total Transaksi")
ax.set_ylabel("Total Payment Value")
ax.legend(title="Kategori Performa", bbox_to_anchor=(1.05, 1), loc="upper left")
st.pyplot(fig)

st.write("Ringkasan Performa Wilayah")
st.dataframe(state_summary)

# =========================
# Optional Geo Scatter
# =========================

st.subheader("Sebaran Wilayah Berdasarkan Payment Value")

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
ax.set_title("Sebaran Wilayah Berdasarkan Total Payment Value")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
st.pyplot(fig)

st.caption("Dashboard dibuat berdasarkan E-Commerce Public Dataset.")