# =============================================================================
# pages/1_Sales_Dashboard.py
# =============================================================================
# PURPOSE:
#   The main Sales Intelligence dashboard. Shows:
#     - KPI cards: Total Revenue, Orders, Customers, AOV
#     - Revenue trend chart (daily/weekly/monthly toggle)
#     - Top 10 products bar chart
#     - Revenue by category (pie chart)
#     - Revenue by region (bar chart)
#     - Auto-generated insights based on the data
#
# HOW STREAMLIT PAGES WORK:
#   This file is automatically detected by Streamlit because it's in pages/ folder.
#   The "1_" prefix sets its order in the sidebar.
#   st.session_state["df"] is the DataFrame loaded on app.py
# =============================================================================

import streamlit as st
from utils.theme import apply_theme, smart_fmt, plotly_defaults
from utils.currency import fmt, get_currency_symbol

def smart_fmt(value: float) -> str:
    """
    Shortens large currency values so they fit inside st.metric cards.
      >= 1,000,000  → $2.48M
      >= 1,000      → $24.8K
      < 1,000       → $248.50
    This prevents the value being cut off in narrow columns.
    """
    sym = get_currency_symbol()
    if value >= 1_000_000:
        return f"{sym}{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{sym}{value/1_000:.1f}K"
    else:
        return fmt(value, 2)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.metrics import (
    total_revenue, total_orders, total_customers, average_order_value,
    revenue_growth_rate, customer_growth_rate,
    daily_revenue, monthly_revenue, weekly_revenue,
    top_products, revenue_by_category, revenue_by_region,
    generate_kpi_insights,
)

# Page config
st.set_page_config(page_title="Sales Dashboard", page_icon="📊", layout="wide")
apply_theme()

# =============================================================================
# GUARD: Check if data is loaded
# =============================================================================
if "df" not in st.session_state or st.session_state["df"] is None:
    st.warning("⚠️ No data loaded. Please go to the **Home** page and upload a CSV first.")
    st.stop()  # stops execution of the rest of the page

df = st.session_state["df"]

# Ensure currency is always set (fallback to GBP for UK dataset, USD otherwise)
if "currency_code" not in st.session_state:
    st.session_state["currency_code"] = "GBP"

# =============================================================================
# HEADER
# =============================================================================
st.title("📊 Sales Dashboard")
st.caption(f"Data source: {st.session_state.get('data_source', 'Unknown')} | {len(df):,} transactions")
st.divider()

# =============================================================================
# KPI CARDS — Top row metrics
# =============================================================================
st.subheader("📌 Key Performance Indicators")

rev = total_revenue(df)
orders = total_orders(df)
customers = total_customers(df)
aov = average_order_value(df)
rev_growth = revenue_growth_rate(df)
cust_growth = customer_growth_rate(df)

# Row 1 — 5 columns so each card has enough space for the value
k1, k2, k3, k4, k5 = st.columns(5)

# smart_fmt shortens large numbers: $2,477,392 → $2.48M so it fits in the card
k1.metric("💰 Total Revenue",   smart_fmt(rev),
          help=f"Exact: {fmt(rev)}")
k2.metric("📦 Total Orders",    f"{orders:,}")
k3.metric("👥 Customers",       f"{customers:,}" if customers > 0 else "N/A")
k4.metric("🛒 Avg Order Value", smart_fmt(aov),
          help=f"Exact: {fmt(aov, 2)}")
k5.metric("📈 Revenue Growth",  f"{rev_growth:+.1f}%",
          delta=f"{rev_growth:+.1f}%",
          delta_color="normal")

# Row 2 — customer growth on its own row so it's never cramped
if customers > 0:
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("👤 Customer Growth",
              f"{cust_growth:+.1f}%" if cust_growth != 0 else "N/A",
              delta=f"{cust_growth:+.1f}%" if cust_growth != 0 else None,
              delta_color="normal")

st.divider()

# =============================================================================
# REVENUE TREND CHART
# =============================================================================
st.subheader("📈 Revenue Trend")

# Toggle button for time granularity
time_granularity = st.radio(
    "View by:",
    options=["Daily", "Weekly", "Monthly"],
    horizontal=True,
    index=2,  # default to Monthly
)

# Get appropriate time series based on selection
if time_granularity == "Daily":
    ts = daily_revenue(df)
    x_col, y_col = "date", "revenue"
elif time_granularity == "Weekly":
    ts = weekly_revenue(df)
    x_col, y_col = "week", "revenue"
else:
    ts = monthly_revenue(df)
    x_col, y_col = "month", "revenue"

if not ts.empty:
    # Area chart with gradient fill — more visually appealing than line
    fig_trend = px.area(
        ts,
        x=x_col,
        y=y_col,
        title=f"{time_granularity} Revenue",
        labels={y_col: f"Revenue ({get_currency_symbol()})", x_col: "Date"},
        color_discrete_sequence=["#667eea"],
    )
    fig_trend.update_traces(fillcolor="rgba(102, 126, 234, 0.2)")
    fig_trend.update_layout(
        height=400,
        hovermode="x unified",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Revenue trend not available — needs date and revenue columns.")

st.divider()

# =============================================================================
# PRODUCT & CATEGORY CHARTS — side by side
# =============================================================================
col_left, col_right = st.columns([1.4, 1], gap="large")

with col_left:
    st.subheader("🏆 Top 10 Products by Revenue")
    top_prods = top_products(df, n=10)

    if not top_prods.empty:
        # Horizontal bar chart — easier to read product names
        fig_products = px.bar(
            top_prods.sort_values("revenue"),
            x="revenue",
            y="product",
            orientation="h",
            labels={"revenue": f"Revenue ({get_currency_symbol()})", "product": "Product"},
            color="revenue",
            color_continuous_scale="Viridis",
        )
        fig_products.update_layout(
            height=400,
            showlegend=False,
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_products, use_container_width=True)
    else:
        st.info("Product analysis not available — needs product column.")

with col_right:
    st.subheader("🍕 Revenue by Category")
    cat_df = revenue_by_category(df)

    if not cat_df.empty:
        fig_cat = px.pie(
            cat_df.head(8),  # max 8 slices for readability
            values="revenue",
            names="category",
            hole=0.4,  # donut chart
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_cat.update_traces(textposition="inside", textinfo="percent+label")
        fig_cat.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("Category analysis not available — needs category column.")

st.divider()

# =============================================================================
# REGION ANALYSIS
# =============================================================================
reg_df = revenue_by_region(df)
if not reg_df.empty:
    st.subheader("🌍 Revenue by Region")
    fig_region = px.bar(
        reg_df.head(15),
        x="region",
        y="revenue",
        text="percentage",
        labels={"revenue": f"Revenue ({get_currency_symbol()})", "region": "Region"},
        color="revenue",
        color_continuous_scale="Blues",
    )
    fig_region.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_region.update_layout(
        height=380,
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_region, use_container_width=True)
    st.divider()

# =============================================================================
# MONTHLY BREAKDOWN TABLE
# =============================================================================
st.subheader("📅 Monthly Performance Breakdown")
monthly = monthly_revenue(df)

if not monthly.empty:
    # Calculate month-over-month change
    monthly["mom_growth"] = monthly["revenue"].pct_change() * 100

    # Format for display
    display_monthly = monthly.copy()
    display_monthly["month"] = display_monthly["month"].dt.strftime("%B %Y")
    display_monthly["revenue"] = display_monthly["revenue"].map(fmt)
    if "orders" in display_monthly.columns:
        display_monthly["orders"] = display_monthly["orders"].map("{:,}".format)
    if "customers" in display_monthly.columns:
        display_monthly["customers"] = display_monthly["customers"].map("{:,}".format)
    display_monthly["mom_growth"] = display_monthly["mom_growth"].map(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"
    )

    st.dataframe(display_monthly, use_container_width=True, hide_index=True)
    st.divider()

# =============================================================================
# AUTO-GENERATED INSIGHTS (Rule-Based — No AI API)
# =============================================================================
st.subheader("💡 Auto-Generated Insights")
st.caption("These insights are generated by analyzing patterns in your data — no AI API required!")

insights = generate_kpi_insights(df)
for insight in insights:
    st.markdown(f"- {insight}")
