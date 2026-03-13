# =============================================================================
# pages/3_Sales_Forecasting.py
# =============================================================================
# PURPOSE:
#   Sales forecasting using Polynomial Regression and Moving Average.
#   Shows historical revenue + predicted future revenue with confidence bands.
#
# WHAT THE USER SEES:
#   1. Combined chart: historical data + 30-day forecast + confidence interval
#   2. Forecast table: day-by-day predicted revenue
#   3. Category-level forecasts (if category column available)
#   4. Model quality metrics (R² score)
#   5. Plain-English forecast insights
# =============================================================================

import streamlit as st
from utils.theme import apply_theme, smart_fmt, plotly_defaults
from utils.currency import fmt, get_currency_symbol

def smart_fmt(value):
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

from utils.forecasting import forecast_revenue, forecast_by_category, moving_average_forecast

st.set_page_config(page_title="Sales Forecasting", page_icon="📈", layout="wide")
apply_theme()

# Guard
if "df" not in st.session_state or st.session_state["df"] is None:
    st.warning("⚠️ No data loaded. Please go to Home and upload a CSV first.")
    st.stop()

df = st.session_state["df"]

# Ensure currency is always set (fallback to GBP for UK dataset, USD otherwise)
if "currency_code" not in st.session_state:
    st.session_state["currency_code"] = "GBP"

if "date" not in df.columns or "revenue" not in df.columns:
    st.error("❌ Forecasting needs `date` and `revenue` columns.")
    st.stop()

st.title("📈 Sales Forecasting")
st.caption("Polynomial Regression + Moving Average — predicts future revenue based on historical patterns")
st.divider()

# =============================================================================
# FORECAST SETTINGS
# =============================================================================
col_settings1, col_settings2 = st.columns(2)
with col_settings1:
    forecast_days = st.slider(
        "Forecast horizon (days)",
        min_value=7, max_value=90, value=30,
        help="How many days into the future to forecast"
    )
with col_settings2:
    model_type = st.radio(
        "Forecasting model",
        ["Polynomial Regression", "Moving Average"],
        horizontal=True,
        help="Polynomial Regression captures trends better. Moving Average is simpler and more stable."
    )

st.divider()

# =============================================================================
# RUN FORECAST
# =============================================================================
with st.spinner(f"Generating {forecast_days}-day forecast..."):
    if model_type == "Polynomial Regression":
        result = forecast_revenue(df, forecast_days=forecast_days)
    else:
        hist = df.groupby(pd.Grouper(key="date", freq="D"))["revenue"].sum().reset_index()
        hist = hist[hist["revenue"] > 0].sort_values("date")
        ma_forecast = moving_average_forecast(df, forecast_days=forecast_days)
        result = {
            "historical": hist,
            "forecast": ma_forecast.rename(columns={"predicted_revenue": "predicted"}),
            "model_score": None,
            "insights": [],
        }

if "error" in result:
    st.error(f"❌ {result['error']}")
    st.stop()

# =============================================================================
# MAIN FORECAST CHART
# =============================================================================
st.subheader("📊 Revenue Forecast Chart")

historical = result["historical"]
forecast_df = result["forecast"]
model_score = result["model_score"]

# Build combined chart using Plotly Graph Objects for full control
fig = go.Figure()

# 1. Historical revenue (solid line)
fig.add_trace(go.Scatter(
    x=historical["date"],
    y=historical["revenue"],
    mode="lines",
    name="Historical Revenue",
    line=dict(color="#667eea", width=2),
))

# 2. Forecast (dashed line)
fig.add_trace(go.Scatter(
    x=forecast_df["date"],
    y=forecast_df["predicted"],
    mode="lines",
    name="Forecast",
    line=dict(color="#f97316", width=2, dash="dash"),
))

# 3. Confidence interval (shaded band) — only for polynomial regression
if "upper_bound" in forecast_df.columns and "lower_bound" in forecast_df.columns:
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"][::-1]]),
        fill="toself",
        fillcolor="rgba(249, 115, 22, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval",
        showlegend=True,
    ))

# Add vertical line at forecast start
# NOTE: add_vline() with annotation_text is broken in many Plotly versions
# with date axes. We use add_shape + add_annotation separately instead.
forecast_start = forecast_df["date"].min().strftime("%Y-%m-%d")
fig.add_shape(
    type="line",
    x0=forecast_start, x1=forecast_start,
    y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(color="gray", width=1, dash="dot"),
)
fig.add_annotation(
    x=forecast_start, y=1,
    xref="x", yref="paper",
    text="Forecast Start",
    showarrow=False,
    yanchor="bottom",
    font=dict(color="gray", size=11),
)

fig.update_layout(
    height=480,
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Date",
    yaxis_title=f"Revenue ({get_currency_symbol()})",
    legend=dict(orientation="h", y=-0.15),
)

st.plotly_chart(fig, use_container_width=True)

# Model quality
if model_score is not None:
    quality = "Excellent" if model_score > 0.8 else "Good" if model_score > 0.5 else "Moderate"
    st.caption(f"Model R² Score: **{model_score}** ({quality} fit) — "
               f"R² of 1.0 = perfect, 0.0 = no pattern detected")

st.divider()

# =============================================================================
# FORECAST SUMMARY STATS
# =============================================================================
st.subheader("📋 Forecast Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Projected Revenue", smart_fmt(forecast_df['predicted'].sum()),
            help=f"Exact: {fmt(forecast_df['predicted'].sum())}")
col2.metric("Daily Average",           smart_fmt(forecast_df['predicted'].mean()))
col3.metric("Peak Day Revenue",        smart_fmt(forecast_df['predicted'].max()))
col4.metric("Forecast Period",         f"{forecast_days} days")

# =============================================================================
# FORECAST TABLE
# =============================================================================
with st.expander("📅 Day-by-Day Forecast Table"):
    display_fc = forecast_df.copy()
    display_fc["date"] = display_fc["date"].dt.strftime("%d %b %Y")
    display_fc["predicted"] = display_fc["predicted"].map(fmt)
    if "upper_bound" in display_fc.columns:
        display_fc["upper_bound"] = display_fc["upper_bound"].map(fmt)
        display_fc["lower_bound"] = display_fc["lower_bound"].map(fmt)
        display_fc.columns = ["Date", "Predicted Revenue", "Upper Bound", "Lower Bound"]
    else:
        display_fc.columns = ["Date", "Predicted Revenue"]
    st.dataframe(display_fc, use_container_width=True, hide_index=True)

st.divider()

# =============================================================================
# CATEGORY FORECASTS
# =============================================================================
if "category" in df.columns:
    st.subheader("📦 Forecast by Category")

    with st.spinner("Generating category forecasts..."):
        cat_forecasts = forecast_by_category(df, forecast_days=forecast_days)

    if cat_forecasts:
        categories = list(cat_forecasts.keys())
        selected_cats = st.multiselect(
            "Select categories to compare:",
            options=categories,
            default=categories[:4],  # show first 4 by default
        )

        if selected_cats:
            fig_cat = go.Figure()
            colors = px.colors.qualitative.Set2

            for i, cat in enumerate(selected_cats):
                res = cat_forecasts[cat]
                if "error" not in res:
                    fig_cat.add_trace(go.Scatter(
                        x=res["forecast"]["date"],
                        y=res["forecast"]["predicted"],
                        mode="lines",
                        name=cat,
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))

            fig_cat.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title=f"Predicted Revenue ({get_currency_symbol()})",
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cat, use_container_width=True)
    st.divider()

# =============================================================================
# AUTO INSIGHTS
# =============================================================================
st.subheader("💡 Forecast Insights")
for insight in result.get("insights", []):
    st.markdown(f"- {insight}")

if not result.get("insights"):
    total = forecast_df["predicted"].sum()
    st.markdown(f"- 💰 Total projected revenue for the next **{forecast_days} days**: {fmt(total)}")
