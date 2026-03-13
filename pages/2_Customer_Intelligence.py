# =============================================================================
# pages/2_Customer_Intelligence.py
# =============================================================================
# PURPOSE:
#   Customer segmentation using RFM Analysis + K-Means Clustering.
#   Churn prediction using Random Forest Classifier.
#
# WHAT THE USER SEES:
#   1. RFM Scatter plot — visualize customers in R/F/M space
#   2. Segment distribution — pie chart showing Champions vs At-Risk etc.
#   3. Segment characteristics table — avg spend, recency per segment
#   4. Churn risk table — each customer's probability of churning
#   5. Actionable recommendations per segment
# =============================================================================

import streamlit as st
from utils.theme import apply_theme, smart_fmt, plotly_defaults
from utils.currency import fmt, get_currency_symbol
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.segmentation import (
    compute_rfm, apply_kmeans_segmentation,
    get_segment_summary, generate_segment_insights,
)
from utils.churn_model import predict_churn, get_churn_insights

st.set_page_config(page_title="Customer Intelligence", page_icon="👥", layout="wide")
apply_theme()

# Guard
if "df" not in st.session_state or st.session_state["df"] is None:
    st.warning("⚠️ No data loaded. Please go to Home and upload a CSV first.")
    st.stop()

df = st.session_state["df"]

# Ensure currency is always set (fallback to GBP for UK dataset, USD otherwise)
if "currency_code" not in st.session_state:
    st.session_state["currency_code"] = "GBP"

if "customer_id" not in df.columns:
    st.error("❌ This module requires a `customer_id` column. Please upload a dataset with customer information.")
    st.stop()

st.title("👥 Customer Intelligence")
st.caption("RFM Analysis + K-Means Clustering + Churn Prediction")
st.divider()

# =============================================================================
# SECTION 1: RFM COMPUTATION
# =============================================================================
st.subheader("📊 RFM Analysis")
st.markdown("""
**RFM** stands for:
- **R**ecency — How recently did the customer buy? (lower days = better)
- **F**requency — How many times did they buy?
- **M**onetary — How much total money did they spend?
""")

with st.spinner("Computing RFM scores..."):
    rfm = compute_rfm(df)

if rfm.empty:
    st.error("Could not compute RFM. Needs: date, customer_id, revenue, order_id columns.")
    st.stop()

# Show raw RFM stats
col1, col2, col3 = st.columns(3)
col1.metric("Avg Recency",   f"{rfm['recency'].mean():.0f} days")
col2.metric("Avg Frequency", f"{rfm['frequency'].mean():.1f} orders")
col3.metric("Avg Spend",     fmt(rfm['monetary'].mean()))

# =============================================================================
# SECTION 2: K-MEANS SEGMENTATION
# =============================================================================
st.divider()
st.subheader("🎯 Customer Segments (K-Means Clustering)")

# Allow user to choose number of segments
n_clusters = st.slider(
    "Number of segments",
    min_value=2, max_value=6, value=4,
    help="How many groups to divide customers into. 4 is the classic RFM segmentation."
)

with st.spinner("Running K-Means clustering..."):
    rfm_segmented = apply_kmeans_segmentation(rfm, n_clusters=n_clusters)

if "segment" not in rfm_segmented.columns:
    st.error("Segmentation failed — not enough customers for clustering.")
    st.stop()

segment_summary = get_segment_summary(rfm_segmented)

# --- Segment Distribution Pie Chart ---
col_pie, col_table = st.columns([1, 1.5], gap="large")

with col_pie:
    st.markdown("**Segment Distribution**")
    fig_pie = px.pie(
        segment_summary,
        values="customers",
        names="segment",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_table:
    st.markdown("**Segment Characteristics**")
    display_seg = segment_summary.copy()
    display_seg["avg_monetary"] = display_seg["avg_monetary"].map(fmt)
    display_seg["total_revenue"] = display_seg["total_revenue"].map(fmt)
    display_seg["avg_recency"] = display_seg["avg_recency"].map("{} days".format)
    display_seg = display_seg.rename(columns={
        "segment": "Segment",
        "customers": "Customers",
        "avg_recency": "Avg Recency",
        "avg_frequency": "Avg Orders",
        "avg_monetary": "Avg Spend",
        "total_revenue": "Total Revenue",
    })
    st.dataframe(display_seg, use_container_width=True, hide_index=True)

# --- RFM 3D Scatter Plot ---
st.markdown("**3D RFM Visualization** — Each dot = one customer")
st.caption("Hover over points to see customer details. Color = segment.")

fig_3d = px.scatter_3d(
    rfm_segmented,
    x="recency",
    y="frequency",
    z="monetary",
    color="segment",
    hover_data=["customer_id"],
    opacity=0.7,
    labels={
        "recency":   "Recency (days)",
        "frequency": "Frequency (orders)",
        "monetary":  f"Monetary ({get_currency_symbol()})",
    },
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_3d.update_layout(height=500)
st.plotly_chart(fig_3d, use_container_width=True)

# --- Segment Recommendations ---
st.subheader("💡 Actionable Recommendations by Segment")
insights = generate_segment_insights(segment_summary)
for insight in insights:
    st.markdown(f"- {insight}")

st.divider()

# =============================================================================
# SECTION 3: CHURN PREDICTION
# =============================================================================
st.subheader("🔮 Churn Prediction (Random Forest Model)")
st.markdown("""
The model predicts which customers are likely to **stop buying**.
It uses behavioral features (recency, frequency, spend, customer age) to assign a churn probability.
""")

with st.spinner("Training churn prediction model... (this may take a moment)"):
    churn_df = predict_churn(df)

if churn_df.empty:
    st.warning("Churn prediction needs more data (at least 10+ customers with purchase history).")
else:
    # --- Risk Distribution ---
    risk_counts = churn_df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]

    col_risk_chart, col_risk_stats = st.columns([1, 1], gap="large")

    with col_risk_chart:
        st.markdown("**Churn Risk Distribution**")
        color_map = {
            "🔴 High Risk":    "#e74c3c",
            "🟠 Medium Risk":  "#e67e22",
            "🟡 Low Risk":     "#f1c40f",
            "🟢 Safe":         "#27ae60",
        }
        fig_risk = px.bar(
            risk_counts,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            color_discrete_map=color_map,
        )
        fig_risk.update_layout(height=350, showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_risk_stats:
        st.markdown("**Top High-Risk Customers**")
        st.caption("These customers have the highest probability of churning — prioritize outreach")
        high_risk = churn_df[churn_df["risk_level"] == "🔴 High Risk"][
            ["customer_id", "churn_probability", "days_since_last_order", "total_orders", "avg_order_value"]
        ].head(10)

        if not high_risk.empty:
            high_risk["churn_probability"] = high_risk["churn_probability"].map("{:.1%}".format)
            high_risk["avg_order_value"] = high_risk["avg_order_value"].map(fmt)
            st.dataframe(high_risk, use_container_width=True, hide_index=True)
        else:
            st.success("🎉 No high-risk customers detected!")

    # --- Churn Probability Histogram ---
    st.markdown("**Churn Probability Distribution**")
    fig_hist = px.histogram(
        churn_df,
        x="churn_probability",
        nbins=20,
        color_discrete_sequence=["#667eea"],
        labels={"churn_probability": "Churn Probability", "count": "Customers"},
    )
    fig_hist.add_shape(
        type="line", x0=0.75, x1=0.75, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=1, dash="dash"),
    )
    fig_hist.add_annotation(
        x=0.75, y=1, xref="x", yref="paper",
        text="High Risk Threshold", showarrow=False,
        yanchor="bottom", font=dict(color="red", size=11),
    )
    fig_hist.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Churn Insights
    st.subheader("💡 Churn Insights")
    churn_insights = get_churn_insights(churn_df)
    for insight in churn_insights:
        st.markdown(f"- {insight}")

    # Full churn table (expandable)
    with st.expander("📋 View All Customer Churn Scores"):
        display_churn = churn_df[
            ["customer_id", "churn_probability", "risk_level",
             "days_since_last_order", "total_orders", "avg_order_value", "total_revenue"]
        ].copy()
        display_churn["churn_probability"] = display_churn["churn_probability"].map("{:.1%}".format)
        display_churn["avg_order_value"] = display_churn["avg_order_value"].map(fmt)
        display_churn["total_revenue"] = display_churn["total_revenue"].map(fmt)
        st.dataframe(display_churn, use_container_width=True, hide_index=True)
