# =============================================================================
# pages/5_Smart_Insights.py
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.theme import apply_theme, smart_fmt, plotly_defaults
from utils.currency import fmt, get_currency_symbol
from utils.metrics import (
    total_revenue, total_orders, total_customers,
    average_order_value, revenue_growth_rate, customer_growth_rate,
    monthly_revenue, top_products, revenue_by_category,
)

st.set_page_config(page_title="Smart Insights", page_icon="💡", layout="wide")
apply_theme()

# =============================================================================
# HELPER — compute_health_score
# =============================================================================
def compute_health_score(df: pd.DataFrame) -> tuple:
    breakdown = {
        "Revenue Growth": 0, "Customer Growth": 0,
        "Product Diversity": 0, "Data Completeness": 0,
    }
    try:
        rg = revenue_growth_rate(df)
        breakdown["Revenue Growth"] = 30 if rg > 10 else 22 if rg > 5 else 15 if rg > 0 else 8 if rg > -5 else 0
    except Exception:
        breakdown["Revenue Growth"] = 10

    try:
        cg = customer_growth_rate(df)
        breakdown["Customer Growth"] = 25 if cg > 10 else 18 if cg > 5 else 12 if cg > 0 else 6 if cg > -5 else 0
    except Exception:
        breakdown["Customer Growth"] = 8

    try:
        n = df["product"].nunique() if "product" in df.columns else 0
        breakdown["Product Diversity"] = 20 if n > 50 else 15 if n > 20 else 10 if n > 10 else 5 if n > 0 else 0
    except Exception:
        breakdown["Product Diversity"] = 5

    required = ["date", "revenue", "customer_id", "product", "category"]
    present = sum(1 for c in required if c in df.columns)
    breakdown["Data Completeness"] = round((present / len(required)) * 25)

    return sum(breakdown.values()), breakdown


# =============================================================================
# LOAD DATA
# =============================================================================
if "df" not in st.session_state or st.session_state["df"] is None:
    st.warning("⚠️ No data loaded. Please go to the **Home** page and upload your CSV file.")
    st.stop()

df = st.session_state["df"]

st.title("💡 Smart Insights")
st.caption("Auto-generated business intelligence report based on your data.")

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================
health_score, score_breakdown = compute_health_score(df)

if health_score >= 75:
    score_color = "#27ae60"; score_label = "Excellent 🟢"
elif health_score >= 55:
    score_color = "#f39c12"; score_label = "Good 🟡"
elif health_score >= 35:
    score_color = "#e67e22"; score_label = "Needs Attention 🟠"
else:
    score_color = "#e74c3c"; score_label = "Critical 🔴"

st.subheader("📋 Executive Summary")
col_score, col_kpis = st.columns([1, 3], gap="large")

with col_score:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=health_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Business Health Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": score_color},
            "steps": [
                {"range": [0, 35],   "color": "#fde8e8"},
                {"range": [35, 55],  "color": "#fef3cd"},
                {"range": [55, 75],  "color": "#d4edda"},
                {"range": [75, 100], "color": "#c3e6cb"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": health_score},
        },
    ))
    fig_gauge.update_layout(height=250, margin=dict(t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown(f"**Status: {score_label}**")

with col_kpis:
    st.markdown("**Score Breakdown**")
    max_scores = {"Revenue Growth": 30, "Customer Growth": 25, "Product Diversity": 20, "Data Completeness": 25}
    for component, score in score_breakdown.items():
        max_s = max_scores[component]
        pct = score / max_s
        st.markdown(f"**{component}**: {score}/{max_s}")
        st.progress(pct)
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue",   smart_fmt(total_revenue(df)),        help=f"Exact: {fmt(total_revenue(df))}")
    k2.metric("Orders",    f"{total_orders(df):,}")
    k3.metric("Growth",    f"{revenue_growth_rate(df):+.1f}%")
    k4.metric("Avg Order", smart_fmt(average_order_value(df)),  help=f"Exact: {fmt(average_order_value(df), 2)}")

st.divider()

# =============================================================================
# SALES HEALTH ANALYSIS
# =============================================================================
st.subheader("📈 Sales Health Analysis")
monthly = monthly_revenue(df)

if not monthly.empty and len(monthly) >= 3:
    monthly["mom_growth"] = monthly["revenue"].pct_change() * 100
    col_trend, col_insights = st.columns([1.5, 1], gap="large")

    with col_trend:
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Bar(
            x=monthly["month"].dt.strftime("%b %Y"),
            y=monthly["mom_growth"],
            marker_color=["#27ae60" if v >= 0 else "#e74c3c" for v in monthly["mom_growth"].fillna(0)],
            name="MoM Growth %",
        ))
        fig_growth.add_shape(type="line", x0=0, x1=1, y0=0, y1=0, xref="paper", yref="y",
                             line=dict(color="gray", width=1, dash="dash"))
        fig_growth.update_layout(title="Month-over-Month Revenue Growth", height=300,
                                 plot_bgcolor="rgba(0,0,0,0)", yaxis_ticksuffix="%")
        st.plotly_chart(fig_growth, use_container_width=True)

    with col_insights:
        st.markdown("**Sales Observations**")
        best_month  = monthly.loc[monthly["revenue"].idxmax()]
        worst_month = monthly.loc[monthly["revenue"].idxmin()]
        avg_growth  = monthly["mom_growth"].mean()
        observations = [
            f"📅 **Best month**: {best_month['month'].strftime('%B %Y')} ({fmt(best_month['revenue'])})",
            f"📉 **Weakest month**: {worst_month['month'].strftime('%B %Y')} ({fmt(worst_month['revenue'])})",
        ]
        if avg_growth > 5:
            observations.append(f"📈 **Trend**: Strong upward growth averaging {avg_growth:.1f}% MoM")
        elif avg_growth > 0:
            observations.append(f"➡️ **Trend**: Gradual growth at {avg_growth:.1f}% MoM")
        else:
            observations.append(f"⚠️ **Trend**: Revenue declining at {avg_growth:.1f}% MoM on average")
        if len(monthly) >= 6:
            cv = monthly["revenue"].std() / monthly["revenue"].mean()
            observations.append(
                f"🌊 **High seasonality** detected (CV={cv:.2f}). Plan inventory accordingly."
                if cv > 0.3 else
                f"✅ **Stable sales** — low seasonality (CV={cv:.2f})"
            )
        for obs in observations:
            st.markdown(f"- {obs}")

st.divider()

# =============================================================================
# CUSTOMER HEALTH ANALYSIS
# =============================================================================
if "customer_id" in df.columns and "revenue" in df.columns:
    st.subheader("👥 Customer Health Analysis")
    cust_rev    = df.groupby("customer_id")["revenue"].sum().sort_values(ascending=False)
    total_rev_v = cust_rev.sum()
    top10_pct   = cust_rev.head(10).sum()  / total_rev_v * 100
    top20_pct   = cust_rev.head(20).sum()  / total_rev_v * 100
    top100_pct  = cust_rev.head(100).sum() / total_rev_v * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Top 10 customers",  f"{top10_pct:.1f}% of revenue")
    c2.metric("Top 20 customers",  f"{top20_pct:.1f}% of revenue")
    c3.metric("Top 100 customers", f"{top100_pct:.1f}% of revenue")

    if top10_pct > 50:
        st.error("🔴 HIGH customer concentration — top 10 customers drive over half of revenue.")
    elif top10_pct > 30:
        st.warning("🟡 MODERATE customer concentration. Consider broadening your customer base.")
    else:
        st.success("🟢 HEALTHY revenue distribution across customer base.")
    st.divider()

# =============================================================================
# PRODUCT HEALTH ANALYSIS
# =============================================================================
if "product" in df.columns:
    st.subheader("📦 Product Health Analysis")
    top_prods = top_products(df, n=100)

    if not top_prods.empty:
        total_rev_p     = top_prods["revenue"].sum()
        concentration_3 = top_prods.head(3)["revenue"].sum()  / total_rev_p * 100
        concentration_10= top_prods.head(10)["revenue"].sum() / total_rev_p * 100

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Revenue Concentration Risk**")
            st.markdown(f"Top 3 products generate **{concentration_3:.1f}%** of revenue")
            st.markdown(f"Top 10 products generate **{concentration_10:.1f}%** of revenue")
            if concentration_3 > 60:
                st.error("🔴 HIGH concentration risk — heavily dependent on few products")
            elif concentration_3 > 40:
                st.warning("🟡 MODERATE concentration — consider expanding product range")
            else:
                st.success("🟢 HEALTHY product diversification")

        with col_b:
            tp_sorted = top_prods.sort_values("revenue", ascending=False).copy()
            tp_sorted["cumulative_pct"] = tp_sorted["revenue"].cumsum() / total_rev_p * 100
            tp_sorted["product_rank"]   = range(1, len(tp_sorted) + 1)
            fig_pareto = px.line(tp_sorted.head(30), x="product_rank", y="cumulative_pct",
                                 title="Revenue Concentration (Pareto Curve)",
                                 labels={"product_rank": "Product Rank", "cumulative_pct": "Cumulative Revenue %"})
            fig_pareto.add_shape(type="line", x0=0, x1=1, y0=80, y1=80, xref="paper", yref="y",
                                 line=dict(color="red", width=1, dash="dash"))
            fig_pareto.add_annotation(x=1, y=80, xref="paper", yref="y", text="80% threshold",
                                      showarrow=False, xanchor="right", font=dict(color="red", size=11))
            fig_pareto.update_layout(height=280, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pareto, use_container_width=True)

    st.divider()

# =============================================================================
# RISK ALERTS
# =============================================================================
st.subheader("🚨 Automated Risk Alerts")
alerts = []

growth = revenue_growth_rate(df)
if growth < -15:
    alerts.append(("🔴 CRITICAL", f"Revenue dropped {abs(growth):.1f}% month-over-month. Immediate investigation needed."))
elif growth < -5:
    alerts.append(("🟠 WARNING",  f"Revenue declining {abs(growth):.1f}% MoM. Monitor closely."))

if "customer_id" in df.columns and "revenue" in df.columns:
    cust_rev = df.groupby("customer_id")["revenue"].sum()
    top_cust_pct = cust_rev.nlargest(5).sum() / cust_rev.sum() * 100
    if top_cust_pct > 50:
        alerts.append(("🟠 WARNING", f"Top 5 customers account for {top_cust_pct:.1f}% of revenue — high dependency risk."))

missing_cols = [c for c in ["date", "revenue", "customer_id", "product"] if c not in df.columns]
if missing_cols:
    alerts.append(("🟡 INFO", f"Columns not detected: {', '.join(missing_cols)}. Some insights may be incomplete."))
if len(df) < 500:
    alerts.append(("🟡 INFO", "Small dataset (<500 rows). ML models and insights may be less accurate."))

if alerts:
    for level, message in alerts:
        if "CRITICAL" in level:
            st.error(f"**{level}**: {message}")
        elif "WARNING" in level:
            st.warning(f"**{level}**: {message}")
        else:
            st.info(f"**{level}**: {message}")
else:
    st.success("✅ No critical alerts detected. Business metrics look healthy!")

st.divider()

# =============================================================================
# STRATEGIC RECOMMENDATIONS
# =============================================================================
st.subheader("🎯 Strategic Recommendations")
recommendations = []
priority = 1

if growth < 0:
    recommendations.append((priority, "HIGH", "Launch a promotional campaign",
        f"Revenue declining {abs(growth):.1f}%. Consider a time-limited discount or bundle offer to stimulate demand."))
    priority += 1

if "customer_id" in df.columns:
    cust_g = customer_growth_rate(df)
    if cust_g < 0:
        recommendations.append((priority, "HIGH", "Improve customer acquisition",
            "Customer count declining. Review marketing channels and referral programs."))
        priority += 1

if "product" in df.columns:
    top_prods_rec = top_products(df, n=3)
    if not top_prods_rec.empty:
        top_name = top_prods_rec.iloc[0]["product"]
        recommendations.append((priority, "MEDIUM", f"Promote top seller: {top_name}",
            "Your best product drives significant revenue. Create bundles and feature it prominently in marketing."))
        priority += 1

if "category" in df.columns:
    cat_rev = revenue_by_category(df)
    if not cat_rev.empty and len(cat_rev) > 1:
        weakest = cat_rev.iloc[-1]
        recommendations.append((priority, "LOW", f"Review underperforming category: {weakest['category']}",
            f"This category represents only {weakest['percentage']:.1f}% of revenue. Consider discontinuing or refreshing the product line."))
        priority += 1

recommendations.append((priority, "MEDIUM", "Implement customer retention program",
    "Use the churn predictions (Customer Intelligence page) to identify at-risk customers and target them with win-back campaigns."))

for rank, level, title, detail in recommendations:
    color = "🔴" if level == "HIGH" else "🟡" if level == "MEDIUM" else "🔵"
    with st.expander(f"{color} Priority {rank}: {title}"):
        st.markdown(detail)
