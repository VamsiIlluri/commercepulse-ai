# =============================================================================
# pages/6_Cohort_Analysis.py
# =============================================================================
# PURPOSE:
#   Cohort Analysis — one of the most powerful techniques in e-commerce analytics.
#   Answers: "Of the customers who first bought in January, what % came back
#             in February? March? Six months later?"
#
# WHAT IS A COHORT?
#   A cohort = a group of customers who made their FIRST purchase in the same month.
#   Example: "January 2023 cohort" = all customers whose very first order was in Jan 2023.
#
# WHY IT MATTERS:
#   - If 40% of Jan customers come back in month 2, you have good retention
#   - If only 5% come back, you have a loyalty problem
#   - Compare cohorts to see if retention is improving over time
#
# HOW THE HEATMAP IS BUILT:
#   1. Find each customer's FIRST purchase month (cohort month)
#   2. For each subsequent order, compute "months since first purchase" (period)
#   3. Count how many customers from each cohort are still active at each period
#   4. Express as % of the cohort's starting size
#   5. Plot as heatmap: rows = cohorts, columns = periods (0, 1, 2, 3... months)
#
# COLOUR INTERPRETATION:
#   Dark purple = high retention (good!)
#   Light / white = low retention (customers dropped off)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.currency import fmt
from utils.theme import apply_theme, page_header, section_header, smart_fmt, insight_card, plotly_defaults, PLOTLY_COLORS

st.set_page_config(page_title="Cohort Analysis", page_icon="🔁", layout="wide")
apply_theme()

# Guard
if "df" not in st.session_state or st.session_state["df"] is None:
    st.warning("⚠️ No data loaded. Please go to Home and upload a CSV first.")
    st.stop()

df = st.session_state["df"]

if "customer_id" not in df.columns or "date" not in df.columns:
    st.error("❌ Cohort Analysis requires `customer_id` and `date` columns.")
    st.stop()

page_header("Cohort Analysis", "Visualise customer retention over time — which acquisition months produce the most loyal buyers?", "🔁")
st.divider()

# =============================================================================
# COMPUTE COHORT TABLE
# =============================================================================
@st.cache_data
def build_cohort_table(df: pd.DataFrame):
    """
    Builds the retention matrix for cohort analysis.

    Returns:
        cohort_pivot  — absolute customer counts (rows=cohort month, cols=period)
        retention_piv — percentage retained vs cohort size at period 0
        cohort_sizes  — number of customers in each cohort
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["order_month"] = df["date"].dt.to_period("M")

    # Step 1: Find each customer's first purchase month
    first_purchase = df.groupby("customer_id")["order_month"].min().reset_index()
    first_purchase.columns = ["customer_id", "cohort_month"]

    # Step 2: Merge back to get cohort_month for every transaction
    df = df.merge(first_purchase, on="customer_id")

    # Step 3: Calculate period = months since first purchase
    df["period"] = (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)

    # Step 4: Count unique customers per (cohort, period)
    cohort_data = (
        df.groupby(["cohort_month", "period"])["customer_id"]
        .nunique()
        .reset_index()
    )
    cohort_data.columns = ["cohort_month", "period", "customers"]

    # Step 5: Pivot to matrix
    cohort_pivot = cohort_data.pivot_table(
        index="cohort_month", columns="period", values="customers"
    )

    # Step 6: Calculate retention % (divide each row by period-0 value)
    cohort_sizes = cohort_pivot[0]
    retention_piv = cohort_pivot.divide(cohort_sizes, axis=0) * 100

    return cohort_pivot, retention_piv, cohort_sizes


with st.spinner("Building cohort matrix..."):
    cohort_abs, cohort_pct, cohort_sizes = build_cohort_table(df)

if cohort_abs.empty:
    st.error("Not enough data to build cohort analysis. Needs at least 3 months of data.")
    st.stop()

# Limit to max 18 periods and 18 cohorts for readability
max_periods = min(18, cohort_pct.shape[1])
max_cohorts = min(18, cohort_pct.shape[0])
cohort_display = cohort_pct.iloc[-max_cohorts:, :max_periods]

# Format index for display
cohort_display.index = cohort_display.index.astype(str)
cohort_display.columns = [f"Month {c}" if c > 0 else "Month 0\n(Acquisition)" for c in cohort_display.columns]

# =============================================================================
# RETENTION HEATMAP
# =============================================================================
section_header("📊 Retention Heatmap")
st.caption("Each cell = % of the cohort still purchasing at that month. Month 0 = 100% (acquisition month).")

# Build text annotations for heatmap
z_values = cohort_display.values
text_annotations = []
for row in z_values:
    text_row = []
    for val in row:
        if np.isnan(val):
            text_row.append("")
        else:
            text_row.append(f"{val:.0f}%")
    text_annotations.append(text_row)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=z_values,
    x=cohort_display.columns.tolist(),
    y=cohort_display.index.tolist(),
    text=text_annotations,
    texttemplate="%{text}",
    textfont=dict(size=11, color="white"),
    colorscale=[
        [0.0,  "#0f172a"],
        [0.05, "#1e3a5f"],
        [0.15, "#1e40af"],
        [0.30, "#3b82f6"],
        [0.50, "#6366f1"],
        [0.70, "#8b5cf6"],
        [0.85, "#a855f7"],
        [1.0,  "#d946ef"],
    ],
    zmin=0,
    zmax=100,
    colorbar=dict(
        title="Retention %",
        tickfont=dict(color="#94a3b8"),
        titlefont=dict(color="#94a3b8"),
    ),
    hoverongaps=False,
    hovertemplate="Cohort: %{y}<br>Period: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
))

fig_heatmap.update_layout(
    height=max(400, max_cohorts * 32),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8"),
    xaxis=dict(side="top", tickfont=dict(size=10, color="#64748b")),
    yaxis=dict(tickfont=dict(size=10, color="#64748b"), autorange="reversed"),
    margin=dict(l=10, r=10, t=60, b=10),
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()

# =============================================================================
# RETENTION CURVES — average retention across all cohorts
# =============================================================================
section_header("📈 Average Retention Curve")
st.caption("Shows the average % of customers still active at each period across all cohorts.")

avg_retention = cohort_display.mean(axis=0).reset_index()
avg_retention.columns = ["period", "retention_pct"]
avg_retention["period_num"] = range(len(avg_retention))

fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(
    x=avg_retention["period"],
    y=avg_retention["retention_pct"],
    mode="lines+markers",
    name="Avg Retention",
    line=dict(color="#6366f1", width=3),
    marker=dict(size=8, color="#6366f1"),
    fill="tozeroy",
    fillcolor="rgba(99,102,241,0.15)",
    hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
))

plotly_defaults(fig_curve, height=340)
fig_curve.update_layout(
    yaxis_title="% Customers Retained",
    xaxis_title="Months Since First Purchase",
    yaxis=dict(range=[0, 105], gridcolor="#1e293b"),
)
st.plotly_chart(fig_curve, use_container_width=True)

st.divider()

# =============================================================================
# COHORT SIZE BAR CHART
# =============================================================================
col1, col2 = st.columns([1.5, 1])

with col1:
    section_header("📦 Cohort Sizes (New Customers per Month)")

    sizes_df = cohort_sizes.reset_index()
    sizes_df.columns = ["cohort_month", "new_customers"]
    sizes_df["cohort_month"] = sizes_df["cohort_month"].astype(str)
    sizes_df = sizes_df.tail(18)

    fig_sizes = px.bar(
        sizes_df,
        x="cohort_month", y="new_customers",
        color="new_customers",
        color_continuous_scale=[[0, "#1e293b"], [0.5, "#6366f1"], [1, "#d946ef"]],
        labels={"cohort_month": "Cohort Month", "new_customers": "New Customers"},
    )
    fig_sizes.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=320, coloraxis_showscale=False,
        font=dict(color="#94a3b8"),
        xaxis=dict(tickangle=-45, gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_sizes, use_container_width=True)

with col2:
    section_header("📊 Key Retention Stats")

    # Month 1 retention (how many come back after first purchase)
    m1_col = "Month 1" if "Month 1" in cohort_display.columns else None
    m3_col = "Month 3" if "Month 3" in cohort_display.columns else None
    m6_col = "Month 6" if "Month 6" in cohort_display.columns else None

    m1_avg = cohort_display[m1_col].mean() if m1_col else None
    m3_avg = cohort_display[m3_col].mean() if m3_col else None
    m6_avg = cohort_display[m6_col].mean() if m6_col else None

    if m1_avg is not None:
        st.metric("Month 1 Retention", f"{m1_avg:.1f}%", help="% of new customers who buy again within 1 month")
    if m3_avg is not None:
        st.metric("Month 3 Retention", f"{m3_avg:.1f}%", help="% still active at 3 months")
    if m6_avg is not None:
        st.metric("Month 6 Retention", f"{m6_avg:.1f}%", help="% still active at 6 months")

    avg_cohort_size = int(cohort_sizes.mean())
    best_cohort = cohort_sizes.idxmax()
    st.metric("Avg Cohort Size", f"{avg_cohort_size:,} customers")
    st.metric("Best Acquisition Month", str(best_cohort))

st.divider()

# =============================================================================
# AUTO INSIGHTS
# =============================================================================
section_header("💡 Cohort Insights")

if m1_avg is not None:
    if m1_avg >= 30:
        insight_card(f"✅ Strong month-1 retention of {m1_avg:.1f}% — customers are coming back after their first purchase.", "success")
    elif m1_avg >= 15:
        insight_card(f"📊 Moderate month-1 retention of {m1_avg:.1f}%. Consider post-purchase email sequences to improve re-engagement.", "warning")
    else:
        insight_card(f"⚠️ Low month-1 retention of {m1_avg:.1f}%. Most customers are not returning. Focus on loyalty programs and follow-up offers.", "danger")

if m6_avg is not None:
    insight_card(f"📈 6-month retention averages {m6_avg:.1f}% across all cohorts — this represents your loyal customer base.", "info")

# Find best and worst retention cohorts
best_m1_idx = cohort_display[m1_col].idxmax() if m1_col else None
worst_m1_idx = cohort_display[m1_col].idxmin() if m1_col else None
if best_m1_idx and worst_m1_idx:
    best_val = cohort_display.loc[best_m1_idx, m1_col]
    worst_val = cohort_display.loc[worst_m1_idx, m1_col]
    insight_card(f"🏆 Best cohort: <strong>{best_m1_idx}</strong> with {best_val:.1f}% month-1 retention. Analyse what campaigns or products drove this.", "success")
    insight_card(f"📉 Weakest cohort: <strong>{worst_m1_idx}</strong> with {worst_val:.1f}% month-1 retention.", "warning")
