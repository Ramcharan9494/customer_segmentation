"""
Customer Segmentation App — RFM Features + K-Means Clustering
Converted & hardened from the original analysis notebook.
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation — RFM + K-Means",
    page_icon="🛍️",
    layout="wide",
)

REQUIRED_COLS = {
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
}

SEGMENT_INFO = {
    "VIP / High-Value": {
        "color": "#2ecc71",
        "profile": "Purchases very frequently, bought recently, and drives the "
                    "highest revenue.",
        "actions": ["Offer VIP membership benefits", "Give early access to new products",
                    "Provide personalized, white-glove service"],
    },
    "Regular": {
        "color": "#3498db",
        "profile": "Buys fairly regularly and contributes a moderate, steady amount "
                    "of revenue.",
        "actions": ["Introduce a loyalty program", "Recommend related products",
                    "Encourage more frequent purchases with seasonal promos"],
    },
    "Inactive / Lost": {
        "color": "#e74c3c",
        "profile": "Hasn't purchased in a long time, buys infrequently, and spends "
                    "the least.",
        "actions": ["Send discount coupons", "Launch a win-back campaign",
                    "Send personalized re-engagement emails"],
    },
}
FALLBACK_COLORS = px.colors.qualitative.Set2


# ----------------------------------------------------------------------------
# Pipeline (cached so the app stays fast on reruns)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        try:
            return pd.read_csv(buf, encoding="ISO-8859-1")
        except UnicodeDecodeError:
            buf.seek(0)
            return pd.read_csv(buf)
    return pd.read_excel(buf)


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=["CustomerID", "Description"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[df["UnitPrice"] > 0]
    df = df[df["Quantity"] > 0]
    # case-insensitive cancelled-invoice filter (bug fix vs. original notebook)
    df = df[~df["InvoiceNo"].astype(str).str.upper().str.startswith("C")]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


@st.cache_data(show_spinner=False)
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    reference_date = df["InvoiceDate"].max() + pd.DateOffset(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    # a handful of customers can net out negative/zero due to returns; drop them
    rfm = rfm[rfm["Monetary"] > 0]
    return rfm


@st.cache_data(show_spinner=False)
def compute_wcss(rfm: pd.DataFrame, k_max: int = 10):
    X = rfm[["Recency", "Frequency", "Monetary"]]
    X_scaled = StandardScaler().fit_transform(X)
    wcss = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    return wcss


@st.cache_resource(show_spinner=False)
def fit_kmeans(rfm: pd.DataFrame, k: int):
    X = rfm[["Recency", "Frequency", "Monetary"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels) if k > 1 else float("nan")
    return scaler, kmeans, labels, sil


def label_segments(rfm_with_clusters: pd.DataFrame) -> dict:
    """Rank clusters by mean Monetary value (desc) and assign business names.
    Computed dynamically so labels stay correct regardless of which raw
    cluster index K-Means happens to assign to which group."""
    means = rfm_with_clusters.groupby("cluster")["Monetary"].mean().sort_values(ascending=False)
    ranked = list(means.index)
    names = ["VIP / High-Value", "Regular", "Inactive / Lost"]
    mapping = {}
    for i, c in enumerate(ranked):
        mapping[c] = names[i] if i < len(names) else f"Segment {i + 1}"
    return mapping


def segment_color(name: str, i: int) -> str:
    return SEGMENT_INFO.get(name, {}).get("color", FALLBACK_COLORS[i % len(FALLBACK_COLORS)])


# ----------------------------------------------------------------------------
# Sidebar — data input & settings
# ----------------------------------------------------------------------------
st.sidebar.title("🛍️ Segmentation Settings")
uploaded = st.sidebar.file_uploader(
    "Upload transaction data (.xlsx or .csv)",
    type=["xlsx", "csv"],
    help="Needs columns: InvoiceNo, StockCode, Description, Quantity, "
         "InvoiceDate, UnitPrice, CustomerID, Country (same schema as the "
         "UCI 'Online Retail' dataset).",
)

k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=8, value=3)
st.sidebar.caption("3 is the elbow-method optimum found in the original analysis — "
                    "try other values and compare the silhouette score.")

st.title("Customer Segmentation Dashboard")
st.caption("RFM feature engineering + K-Means clustering, built from the original notebook")

if uploaded is None:
    st.info(
        "👈 Upload a transactions file to get started. Expected columns: "
        + ", ".join(sorted(REQUIRED_COLS))
        + ".\n\nNo dataset handy? The original project used the public "
          "[UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail)."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Run pipeline
# ----------------------------------------------------------------------------
with st.spinner("Loading data..."):
    raw_df = load_data(uploaded.getvalue(), uploaded.name)

missing_cols = REQUIRED_COLS - set(raw_df.columns)
if missing_cols:
    st.error(f"This file is missing required column(s): {', '.join(sorted(missing_cols))}")
    st.stop()

with st.spinner("Cleaning data & engineering RFM features..."):
    clean_df = clean_data(raw_df)
    rfm = compute_rfm(clean_df)

if len(rfm) < k:
    st.error(f"Only {len(rfm)} customers remain after cleaning — reduce k or check your data.")
    st.stop()

with st.spinner("Fitting K-Means..."):
    scaler, kmeans, labels, sil_score = fit_kmeans(rfm, k)
    rfm = rfm.copy()
    rfm["cluster"] = labels
    segment_map = label_segments(rfm)
    rfm["Segment"] = rfm["cluster"].map(segment_map)

# ----------------------------------------------------------------------------
# Top-line metrics
# ----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers analyzed", f"{len(rfm):,}")
c2.metric("Total revenue", f"£{rfm['Monetary'].sum():,.0f}")
c3.metric("Clusters (k)", k)
c4.metric("Silhouette score", f"{sil_score:.3f}" if sil_score == sil_score else "—",
          help="Ranges from -1 to 1. Higher means better-separated clusters.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Cluster Explorer", "🧭 Segment Profiles", "📈 Elbow Method", "🔮 Predict a Customer"]
)

# ----------------------------------------------------------------------------
# Tab 1 — Cluster explorer
# ----------------------------------------------------------------------------
with tab1:
    seg_names = [segment_map[c] for c in sorted(segment_map)]
    color_map = {name: segment_color(name, i) for i, name in enumerate(seg_names)}

    colA, colB = st.columns(2)
    with colA:
        fig1 = px.scatter(
            rfm, x="Recency", y="Monetary", color="Segment",
            color_discrete_map=color_map, opacity=0.7,
            hover_data=["Frequency"], title="Recency vs Monetary",
        )
        st.plotly_chart(fig1, use_container_width=True)
    with colB:
        fig2 = px.scatter(
            rfm, x="Frequency", y="Monetary", color="Segment",
            color_discrete_map=color_map, opacity=0.7,
            hover_data=["Recency"], title="Frequency vs Monetary",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Customers per segment")
    counts = rfm["Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Customers"]
    fig3 = px.bar(counts, x="Segment", y="Customers", color="Segment",
                  color_discrete_map=color_map, text="Customers")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Segmented customer table")
    st.dataframe(
        rfm.reset_index()[["CustomerID", "Recency", "Frequency", "Monetary", "Segment"]]
        .sort_values("Monetary", ascending=False),
        use_container_width=True,
    )
    csv_bytes = rfm.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download segmented customers (CSV)", csv_bytes,
                        file_name="customer_segments.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# Tab 2 — Segment profiles & business recommendations
# ----------------------------------------------------------------------------
with tab2:
    summary = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(2)
    summary = summary.loc[[s for s in seg_names if s in summary.index]]

    for name in summary.index:
        info = SEGMENT_INFO.get(name)
        with st.container(border=True):
            st.markdown(f"### {name}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg. Recency (days)", f"{summary.loc[name, 'Recency']:.1f}")
            m2.metric("Avg. Frequency", f"{summary.loc[name, 'Frequency']:.1f}")
            m3.metric("Avg. Monetary", f"£{summary.loc[name, 'Monetary']:,.0f}")
            if info:
                st.write(info["profile"])
                st.markdown("**Recommended actions:**")
                for a in info["actions"]:
                    st.markdown(f"- {a}")

    fig4 = px.bar(
        summary.reset_index().melt(id_vars="Segment", var_name="Metric", value_name="Value"),
        x="Segment", y="Value", color="Metric", barmode="group",
        title="Average RFM values per segment",
    )
    st.plotly_chart(fig4, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 3 — Elbow method (for choosing k)
# ----------------------------------------------------------------------------
with tab3:
    st.write("Use this to sanity-check your choice of *k* in the sidebar.")
    wcss = compute_wcss(rfm, k_max=10)
    fig5 = px.line(x=list(range(1, 11)), y=wcss, markers=True,
                    labels={"x": "Number of clusters (k)", "y": "WCSS (inertia)"},
                    title="Elbow Method")
    fig5.add_vline(x=k, line_dash="dash", line_color="red",
                    annotation_text=f"current k={k}")
    st.plotly_chart(fig5, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 4 — Predict a single customer's segment
# ----------------------------------------------------------------------------
with tab4:
    st.write("Enter a customer's RFM values to see which segment they'd fall into "
             "under the current model.")
    p1, p2, p3 = st.columns(3)
    recency = p1.number_input("Recency (days since last purchase)", min_value=0, value=30)
    frequency = p2.number_input("Frequency (number of orders)", min_value=1, value=5)
    monetary = p3.number_input("Monetary (total spend, £)", min_value=0.0, value=500.0)

    if st.button("Predict segment", type="primary"):
        X_new = scaler.transform([[recency, frequency, monetary]])
        cluster_id = kmeans.predict(X_new)[0]
        seg_name = segment_map[cluster_id]
        color = segment_color(seg_name, list(segment_map).index(cluster_id))
        st.markdown(
            f"<div style='padding:1rem;border-radius:0.5rem;background-color:{color}22;"
            f"border:1px solid {color}'><h4 style='margin:0;color:{color}'>"
            f"Predicted segment: {seg_name}</h4></div>",
            unsafe_allow_html=True,
        )
        info = SEGMENT_INFO.get(seg_name)
        if info:
            st.write(info["profile"])
