# =============================================================================
# pages/7_LTV_Analysis.py
# =============================================================================
# PURPOSE:
#   Customer Lifetime Value (LTV) analysis and prediction.
#   Answers: "How much is each customer worth over their lifetime?
#             Which customers should we spend marketing budget to retain?"
#
# WHAT IS LTV?
#   LTV = total revenue a customer generates over their entire relationship
#         with your business — including future purchases we haven't seen yet.
#
# HOW WE CALCULATE IT:
#   Historical LTV (what they've spent so far):
#     historical_ltv = sum of all their orders
#
#   Predicted Future LTV (BG/NBD-inspired simplified model):
#     predicted_ltv = avg_order_value × predicted_future_orders
#     predicted_future_orders = frequency_rate × predicted_active_months
#     predicted_active_months = based on recency + churn probability
#
#   Total LTV = historical_ltv + predicted_future_ltv
#
# SEGMENTATION THRESHOLDS:
#   Platinum: top 10% LTV
#   Gold:     10–30%
#   Silver:   30–60%
#   Bronze:   bottom 40%
#
# WHY THIS MATTERS FOR COLLEGE PROJECT:
#   LTV is the most important metric in e-commerce — it determines:
#   - How much you can afford to spend acquiring a customer (CAC)
#   - Which customers to target with loyalty rewards
#   - Which segments are most profitable
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.currency import fmt
from utils.theme import apply_theme, page_header, section_header, smart_fmt, insight_card, plotly_defaults, PLOTLY_COLORS

st.set_page_config(page_title="LTV Analysis", page_icon="💎", layout="wide")
apply_theme()

# Guard
if "df" not in st.session_state or st.session_state["df"] is None:
    st.warning("⚠️ No data loaded. Please go to Home and upload a CSV first.")
    st.stop()

df = st.session_state["df"]

if "customer_id" not in df.columns or "revenue" not in df.columns:
    st.error("❌ LTV Analysis requires `customer_id` and `revenue` columns.")
    st.stop()

page_header("Customer Lifetime Value", "Predict how much each customer is worth — and which ones to prioritise for retention marketing", "💎")
st.divider()

# =============================================================================
# COMPUTE LTV PER CUSTOMER
# =============================================================================
@st.cache_data
def compute_ltv(df: pd.DataFrame, prediction_months: int = 12) -> pd.DataFrame:
    """
    Computes historical and predicted LTV for every customer.

    Args:
        df: Main DataFrame
        prediction_months: How many months ahead to predict

    Returns:
        DataFrame with one row per customer containing LTV metrics
    """
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    reference_date = df["date"].max() if "date" in df.columns else pd.Timestamp.now()

    agg = {"revenue": ["sum", "mean", "count"]}
    if "order_id" in df.columns:
        agg["order_id"] = "nunique"
    if "date" in df.columns:
        agg["date"] = ["min", "max"]

    ltv = df.groupby("customer_id").agg(agg)
    ltv.columns = ["_".join(c).strip() for c in ltv.columns]
    ltv = ltv.reset_index()

    # Rename for clarity
    ltv = ltv.rename(columns={
        "revenue_sum": "historical_ltv",
        "revenue_mean": "avg_order_value",
        "revenue_count": "total_transactions",
    })

    if "order_id_nunique" in ltv.columns:
        ltv = ltv.rename(columns={"order_id_nunique": "total_orders"})
    else:
        ltv["total_orders"] = ltv["total_transactions"]

    if "date_min" in ltv.columns and "date_max" in ltv.columns:
        ltv["first_purchase"] = pd.to_datetime(ltv["date_min"])
        ltv["last_purchase"]  = pd.to_datetime(ltv["date_max"])
        ltv["customer_age_days"]   = (reference_date - ltv["first_purchase"]).dt.days.clip(lower=1)
        ltv["recency_days"]        = (reference_date - ltv["last_purchase"]).dt.days
        ltv["customer_age_months"] = ltv["customer_age_days"] / 30.44
        # Orders per month
        ltv["order_rate"] = ltv["total_orders"] / ltv["customer_age_months"].clip(lower=1)
    else:
        ltv["customer_age_months"] = 6
        ltv["recency_days"] = 30
        ltv["order_rate"] = ltv["total_orders"] / 6

    # Churn probability proxy: customers not buying in 90+ days are considered at risk
    CHURN_THRESHOLD = 90
    ltv["churn_probability"] = (ltv["recency_days"] / CHURN_THRESHOLD).clip(0, 1)

    # Expected active months in prediction period
    ltv["expected_active_months"] = prediction_months * (1 - ltv["churn_probability"])

    # Predicted future LTV
    ltv["predicted_future_ltv"] = (
        ltv["avg_order_value"] * ltv["order_rate"] * ltv["expected_active_months"]
    ).clip(lower=0)

    # Total LTV = what they've spent + what they will spend
    ltv["total_ltv"] = ltv["historical_ltv"] + ltv["predicted_future_ltv"]

    # LTV Segment (percentile-based)
    p90 = ltv["total_ltv"].quantile(0.90)
    p70 = ltv["total_ltv"].quantile(0.70)
    p40 = ltv["total_ltv"].quantile(0.40)

    def segment(v):
        if v >= p90:   return "💎 Platinum"
        elif v >= p70: return "🥇 Gold"
        elif v >= p40: return "🥈 Silver"
        else:          return "🥉 Bronze"

    ltv["ltv_segment"] = ltv["total_ltv"].apply(segment)

    # ROI potential: predicted LTV vs already spent (value left to capture)
    ltv["untapped_value"] = ltv["predicted_future_ltv"]

    return ltv.sort_values("total_ltv", ascending=False).reset_index(drop=True)


# Settings
pred_months = st.slider("Prediction horizon (months)", min_value=3, max_value=24, value=12,
                        help="How many months ahead to project future purchases")

with st.spinner("Computing customer lifetime values..."):
    ltv_df = compute_ltv(df, prediction_months=pred_months)

if ltv_df.empty:
    st.error("Could not compute LTV. Check that your data has customer_id and revenue columns.")
    st.stop()

st.divider()

# =============================================================================
# KPI SUMMARY
# =============================================================================
section_header("📌 LTV Overview")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg Customer LTV",    smart_fmt(ltv_df["total_ltv"].mean()),         help="Average total value per customer")
k2.metric("Median LTV",          smart_fmt(ltv_df["total_ltv"].median()))
k3.metric("Top 10% LTV (Platinum)", smart_fmt(ltv_df["total_ltv"].quantile(0.90)))
k4.metric("Total Predicted Revenue", smart_fmt(ltv_df["predicted_future_ltv"].sum()), help=f"Projected over next {pred_months} months")
k5.metric("Avg Untapped Value",  smart_fmt(ltv_df["untapped_value"].mean()),     help="Predicted future value not yet captured")

st.divider()

# =============================================================================
# LTV DISTRIBUTION + SEGMENT BREAKDOWN — side by side
# =============================================================================
col1, col2 = st.columns([1.5, 1])

with col1:
    section_header("📊 LTV Distribution")
    fig_hist = px.histogram(
        ltv_df,
        x="total_ltv",
        nbins=50,
        color_discrete_sequence=["#6366f1"],
        labels={"total_ltv": "Total LTV (₹)"},
    )
    fig_hist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=320, font=dict(color="#94a3b8"),
        bargap=0.05,
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    section_header("🎯 Segment Breakdown")
    seg_counts = ltv_df["ltv_segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "customers"]
    seg_revenue = ltv_df.groupby("ltv_segment")["total_ltv"].sum().reset_index()
    seg_revenue.columns = ["segment", "total_ltv"]
    seg_summary = seg_counts.merge(seg_revenue, on="segment")
    seg_summary["% of Revenue"] = (seg_summary["total_ltv"] / seg_summary["total_ltv"].sum() * 100).round(1)

    fig_seg = px.pie(
        seg_summary,
        values="customers",
        names="segment",
        hole=0.45,
        color_discrete_sequence=["#d946ef", "#6366f1", "#3b82f6", "#1e3a5f"],
    )
    fig_seg.update_traces(textinfo="percent+label", textfont_size=11)
    fig_seg.update_layout(
        height=320, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig_seg, use_container_width=True)

st.divider()

# =============================================================================
# HISTORICAL vs PREDICTED LTV scatter
# =============================================================================
section_header("🔮 Historical vs Predicted LTV")
st.caption("Each dot = one customer. Dots above the diagonal have more predicted value than already captured.")

fig_scatter = px.scatter(
    ltv_df.head(2000),   # limit to 2000 for performance
    x="historical_ltv",
    y="predicted_future_ltv",
    color="ltv_segment",
    size="total_orders",
    size_max=15,
    opacity=0.7,
    color_discrete_map={
        "💎 Platinum": "#d946ef",
        "🥇 Gold":     "#f59e0b",
        "🥈 Silver":   "#94a3b8",
        "🥉 Bronze":   "#334155",
    },
    labels={
        "historical_ltv":       "Historical LTV (₹)",
        "predicted_future_ltv": f"Predicted LTV — Next {pred_months} Months (₹)",
        "ltv_segment":          "Segment",
    },
    hover_data=["customer_id", "total_orders", "recency_days"],
)
plotly_defaults(fig_scatter, height=420)
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# =============================================================================
# SEGMENT SUMMARY TABLE
# =============================================================================
section_header("📋 Segment Performance Summary")

seg_table = ltv_df.groupby("ltv_segment").agg(
    customers=("customer_id", "count"),
    avg_historical_ltv=("historical_ltv", "mean"),
    avg_predicted_ltv=("predicted_future_ltv", "mean"),
    avg_total_ltv=("total_ltv", "mean"),
    avg_orders=("total_orders", "mean"),
    avg_aov=("avg_order_value", "mean"),
    avg_recency=("recency_days", "mean"),
).reset_index()

seg_table["avg_historical_ltv"] = seg_table["avg_historical_ltv"].map(smart_fmt)
seg_table["avg_predicted_ltv"]  = seg_table["avg_predicted_ltv"].map(smart_fmt)
seg_table["avg_total_ltv"]      = seg_table["avg_total_ltv"].map(smart_fmt)
seg_table["avg_orders"]         = seg_table["avg_orders"].map("{:.1f}".format)
seg_table["avg_aov"]            = seg_table["avg_aov"].map(smart_fmt)
seg_table["avg_recency"]        = seg_table["avg_recency"].map("{:.0f} days".format)
seg_table.columns = ["Segment", "Customers", "Avg Historical LTV", "Avg Predicted LTV", "Avg Total LTV", "Avg Orders", "Avg Order Value", "Avg Recency"]

st.dataframe(seg_table, use_container_width=True, hide_index=True)

st.divider()

# =============================================================================
# TOP CUSTOMERS TABLE
# =============================================================================
section_header("🏆 Top 20 Customers by Total LTV")

top20 = ltv_df.head(20)[["customer_id", "ltv_segment", "historical_ltv", "predicted_future_ltv", "total_ltv", "total_orders", "avg_order_value", "recency_days"]].copy()
top20["historical_ltv"]       = top20["historical_ltv"].map(smart_fmt)
top20["predicted_future_ltv"] = top20["predicted_future_ltv"].map(smart_fmt)
top20["total_ltv"]            = top20["total_ltv"].map(smart_fmt)
top20["avg_order_value"]      = top20["avg_order_value"].map(smart_fmt)
top20["recency_days"]         = top20["recency_days"].map("{:.0f}d".format)
top20.columns = ["Customer ID", "Segment", "Historical LTV", "Predicted LTV", "Total LTV", "Orders", "Avg Order Value", "Last Purchase"]
st.dataframe(top20, use_container_width=True, hide_index=True)

st.divider()

# =============================================================================
# AUTO INSIGHTS
# =============================================================================
section_header("💡 LTV Insights")

platinum_pct = (ltv_df["ltv_segment"] == "💎 Platinum").mean() * 100
platinum_revenue_pct = ltv_df[ltv_df["ltv_segment"] == "💎 Platinum"]["total_ltv"].sum() / ltv_df["total_ltv"].sum() * 100
avg_ltv = ltv_df["total_ltv"].mean()
avg_aov = ltv_df["avg_order_value"].mean()
predicted_total = ltv_df["predicted_future_ltv"].sum()

insight_card(f"💎 Platinum customers are just <strong>{platinum_pct:.1f}%</strong> of your base but represent <strong>{platinum_revenue_pct:.1f}%</strong> of total LTV — focus retention efforts here first.", "success")
insight_card(f"💰 Average customer LTV is <strong>{smart_fmt(avg_ltv)}</strong> — this is the maximum you should spend to acquire a new customer (CAC ceiling).", "info")
insight_card(f"🔮 Predicted future revenue across all customers over the next {pred_months} months: <strong>{smart_fmt(predicted_total)}</strong>.", "info")

high_risk = ltv_df[(ltv_df["churn_probability"] > 0.7) & (ltv_df["ltv_segment"].isin(["💎 Platinum", "🥇 Gold"]))].shape[0]
if high_risk > 0:
    insight_card(f"⚠️ <strong>{high_risk}</strong> high-value customers (Platinum/Gold) have a churn probability >70%. Trigger win-back campaigns immediately.", "danger")

insight_card(f"📊 Average Order Value is <strong>{smart_fmt(avg_aov)}</strong>. Upsell opportunities in Platinum segment could significantly increase total LTV.", "info")
