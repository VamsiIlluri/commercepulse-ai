# =============================================================================
# pages/4_Product_Intelligence.py
# =============================================================================
# PURPOSE:
#   Product performance analysis and Market Basket Analysis.
#   Answers: "Which products should we promote? What goes well together?"
#
# WHAT THE USER SEES:
#   1. Product performance scores (Stars, Declining, Underperformers)
#   2. Market Basket Analysis — frequently bought together
#   3. Association rules table (product_a → product_b with lift/confidence)
#   4. Product recommendation lookup: "What goes with Product X?"
#   5. Category performance comparison
# =============================================================================

import streamlit as st
from utils.theme import apply_theme, smart_fmt, plotly_defaults
from utils.currency import fmt, get_currency_symbol
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.recommendations import (
    product_performance_score,
    find_product_associations,
    get_recommendations_for_product,
)

st.set_page_config(page_title="Product Intelligence", page_icon="🛒", layout="wide")
apply_theme()

# Guard
if "df" not in st.session_state or st.session_state["df"] is None:
    st.warning("⚠️ No data loaded. Please go to Home and upload a CSV first.")
    st.stop()

df = st.session_state["df"]

# Ensure currency is always set (fallback to GBP for UK dataset, USD otherwise)
if "currency_code" not in st.session_state:
    st.session_state["currency_code"] = "GBP"

if "product" not in df.columns:
    st.error("❌ Product Intelligence needs a `product` column.")
    st.stop()

st.title("🛒 Product Intelligence")
st.caption("Product Performance Scoring + Market Basket Analysis (Association Rules)")
st.divider()

# =============================================================================
# SECTION 1: PRODUCT PERFORMANCE SCORING
# =============================================================================
st.subheader("⭐ Product Performance Scores")
st.markdown("""
Each product is scored on **Revenue (50%)**, **Order Frequency (30%)**, and **Growth Trend (20%)**.
Products are classified as Stars, Steady, Average, Declining, or Underperformers.
""")

with st.spinner("Scoring all products..."):
    perf_df = product_performance_score(df)

if perf_df.empty:
    st.warning("Could not compute product scores.")
else:
    # --- Category Filter ---
    categories = ["All"] + sorted(perf_df["category"].unique().tolist()) if "category" in perf_df.columns else ["All"]
    selected_cat = st.selectbox("Filter by category:", categories)

    display_perf = perf_df.copy()
    if selected_cat != "All" and "category" in display_perf.columns:
        display_perf = display_perf[display_perf["category"] == selected_cat]

    # --- Performance Distribution ---
    col_chart, col_stats = st.columns([1, 1.5], gap="large")

    with col_chart:
        cat_counts = display_perf["category"].value_counts().reset_index() if "category" in display_perf.columns else pd.DataFrame()
        perf_counts = display_perf["category"].value_counts().reset_index()

        # Show performance category distribution
        if "category" in display_perf.columns:
            cat_perf = display_perf.groupby("category")["total_revenue"].sum().reset_index()
            fig_cat = px.bar(
                cat_perf.sort_values("total_revenue", ascending=True).tail(10),
                x="total_revenue",
                y="category",
                orientation="h",
                color="total_revenue",
                color_continuous_scale="Viridis",
                labels={"total_revenue": f"Revenue ({get_currency_symbol()})", "category": "Category"},
                title="Revenue by Category"
            )
            fig_cat.update_layout(height=350, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cat, use_container_width=True)

    with col_stats:
        # Performance bubble chart: x=orders, y=revenue, size=composite_score
        fig_bubble = px.scatter(
            display_perf.head(50),  # limit to top 50 for clarity
            x="order_count",
            y="total_revenue",
            size="composite_score",
            color="category" if "category" in display_perf.columns else None,
            hover_name="product",
            hover_data={"composite_score": True, "trend_pct": True},
            labels={
                "order_count":   "Number of Orders",
                "total_revenue": f"Total Revenue ({get_currency_symbol()})",
                "composite_score": "Performance Score",
            },
            title="Products: Orders vs Revenue (size = performance score)",
        )
        fig_bubble.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bubble, use_container_width=True)

    # --- Performance Table ---
    st.markdown("**Top Products by Performance Score**")
    display_cols = ["product", "composite_score", "total_revenue", "order_count",
                    "avg_price", "trend_pct"]
    display_cols = [c for c in display_cols if c in display_perf.columns]

    show_perf = display_perf[display_cols].head(20).copy()
    if "total_revenue" in show_perf.columns:
        show_perf["total_revenue"] = show_perf["total_revenue"].map(fmt)
    if "avg_price" in show_perf.columns:
        show_perf["avg_price"] = show_perf["avg_price"].map(fmt)
    if "trend_pct" in show_perf.columns:
        show_perf["trend_pct"] = show_perf["trend_pct"].map("{:+.1f}%".format)

    st.dataframe(show_perf, use_container_width=True, hide_index=True)

st.divider()

# =============================================================================
# SECTION 2: MARKET BASKET ANALYSIS
# =============================================================================
st.subheader("🛍️ Market Basket Analysis")
st.markdown("""
**"Customers who bought X also bought Y"**

This uses **Association Rules** to find products that are frequently purchased together.
- **Support**: % of orders containing both products
- **Confidence**: If customer buys A, probability they also buy B
- **Lift**: How much more likely than by chance (>1 = positive association)
""")

if "order_id" not in df.columns:
    st.warning("⚠️ Market Basket Analysis requires an `order_id` column to identify baskets.")
else:
    min_support = st.slider(
        "Minimum support threshold",
        min_value=0.001, max_value=0.05, value=0.005, step=0.001,
        format="%.3f",
        help="Lower = more rules found but weaker associations. Higher = fewer but stronger rules."
    )

    with st.spinner("Mining association rules... (may take 30-60 seconds for large datasets)"):
        rules_df = find_product_associations(df, min_support=min_support)

    if rules_df.empty:
        st.info("No strong associations found at this support threshold. Try lowering it.")
    else:
        st.success(f"✅ Found **{len(rules_df)} product association rules**")

        # Scatter plot: Confidence vs Lift
        fig_rules = px.scatter(
            rules_df,
            x="confidence",
            y="lift",
            size="support",
            hover_data=["product_a", "product_b"],
            color="lift",
            color_continuous_scale="RdYlGn",
            labels={
                "confidence": "Confidence (if A → B probability)",
                "lift":       "Lift (strength of association)",
                "support":    "Support",
            },
            title="Association Rules: Confidence vs Lift",
        )
        fig_rules.add_shape(
            type="line", x0=0, x1=1, y0=1, y1=1,
            xref="paper", yref="y",
            line=dict(color="gray", width=1, dash="dash"),
        )
        fig_rules.add_annotation(
            x=1, y=1, xref="paper", yref="y",
            text="Lift=1 (no association)", showarrow=False,
            xanchor="right", font=dict(color="gray", size=11),
        )
        fig_rules.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rules, use_container_width=True)

        # Rules table
        st.markdown("**Top Association Rules (sorted by Lift)**")
        display_rules = rules_df.copy()
        display_rules["support"]    = display_rules["support"].map("{:.2%}".format)
        display_rules["confidence"] = display_rules["confidence"].map("{:.2%}".format)
        display_rules["lift"]       = display_rules["lift"].map("{:.2f}x".format)
        display_rules.columns = ["If customer buys", "They likely also buy", "Support", "Confidence", "Lift"]
        st.dataframe(display_rules, use_container_width=True, hide_index=True)

        # --- Product Recommendation Lookup ---
        st.divider()
        st.subheader("🔍 Product Recommendation Lookup")
        st.caption("Select a product to see what customers commonly buy alongside it")

        all_products = sorted(set(rules_df["product_a"].tolist() + rules_df["product_b"].tolist()))
        selected_product = st.selectbox("Choose a product:", all_products)

        if selected_product:
            recs = get_recommendations_for_product(rules_df, selected_product, n=5)
            if recs:
                st.markdown(f"**Customers who buy '{selected_product}' also tend to buy:**")
                for i, (rec_prod, lift, conf) in enumerate(recs, 1):
                    st.markdown(
                        f"{i}. **{rec_prod}** — "
                        f"Lift: `{lift:.2f}x` | Confidence: `{conf:.1%}`"
                    )
                st.info(
                    f"💡 **Tip**: Bundle '{selected_product}' with the top recommendation "
                    f"for a cross-sell campaign!"
                )
            else:
                st.info("No strong associations found for this product.")
