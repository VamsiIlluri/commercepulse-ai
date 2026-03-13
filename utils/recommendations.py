# =============================================================================
# utils/recommendations.py
# =============================================================================
# PURPOSE:
#   Implements Market Basket Analysis to find which products are frequently
#   bought together and generate product recommendations.
#
# WHAT IS MARKET BASKET ANALYSIS?
#   "Customers who bought X also bought Y"
#   Amazon uses this extensively. We use the Apriori algorithm.
#
# KEY CONCEPTS:
#   - Support:    How often items appear together (e.g., 5% of orders have A+B)
#   - Confidence: If A is bought, how likely is B? (e.g., 60% chance)
#   - Lift:       How much more likely is B given A vs by chance?
#                 Lift > 1 = positive association, higher = stronger
#
# HOW APRIORI WORKS:
#   1. Create a "basket matrix" — one row per order, one column per product
#      Each cell = 1 if product was in that order, 0 otherwise
#   2. Find all product combinations that appear together frequently
#   3. Calculate confidence and lift for each combination
#   4. Return high-lift rules as recommendations
# =============================================================================

import pandas as pd
import numpy as np


def build_basket_matrix(df: pd.DataFrame, max_products: int = 50) -> pd.DataFrame:
    """
    Creates the basket matrix needed for association rule mining.

    Example output:
      order_id  | Apple | Banana | Cherry
      order_001 |   1   |    1   |    0
      order_002 |   0   |    1   |    1
      order_003 |   1   |    0   |    1

    Args:
        df: Main DataFrame with order_id and product columns
        max_products: Limit to top N products to keep matrix manageable

    Returns:
        Boolean DataFrame (order × product matrix)
    """
    if "order_id" not in df.columns or "product" not in df.columns:
        return pd.DataFrame()

    # Limit to top products by frequency (otherwise matrix is too large)
    top_products = df["product"].value_counts().head(max_products).index
    df_filtered = df[df["product"].isin(top_products)]

    # Create basket: each row = one order, each column = one product
    # Values = 1 if product in order, 0 if not
    basket = (
        df_filtered
        .groupby(["order_id", "product"])["revenue"]
        .sum()
        .unstack(fill_value=0)
    )

    # Convert to binary (1 if any quantity, 0 otherwise)
    basket = (basket > 0).astype(int)
    return basket


def find_product_associations(df: pd.DataFrame, min_support: float = 0.02) -> pd.DataFrame:
    """
    Finds product pairs that are frequently bought together.

    MANUAL IMPLEMENTATION (without mlxtend to avoid dependency issues):
    We manually compute support and lift for all product pairs.

    Args:
        df: Main DataFrame
        min_support: Minimum fraction of orders that must contain the pair (default 2%)

    Returns:
        DataFrame with columns: [product_a, product_b, support, confidence, lift]
        Sorted by lift descending
    """
    basket = build_basket_matrix(df)
    if basket.empty or basket.shape[1] < 2:
        return pd.DataFrame()

    n_orders = len(basket)
    products = basket.columns.tolist()
    rules = []

    # Calculate support for each individual product
    product_support = basket.sum() / n_orders

    # Calculate pair support and confidence
    for i, prod_a in enumerate(products):
        for j, prod_b in enumerate(products):
            if i >= j:
                continue  # avoid duplicates and self-pairs

            # Support(A ∩ B) = orders containing both A and B / total orders
            both = ((basket[prod_a] == 1) & (basket[prod_b] == 1)).sum()
            support = both / n_orders

            if support < min_support:
                continue  # skip rare pairs

            # Confidence(A → B) = Support(A ∩ B) / Support(A)
            support_a = product_support[prod_a]
            confidence_ab = support / support_a if support_a > 0 else 0

            # Lift(A → B) = Confidence(A → B) / Support(B)
            # Lift > 1 means A and B are bought together more than by chance
            support_b = product_support[prod_b]
            lift = confidence_ab / support_b if support_b > 0 else 0

            if lift > 1.1:  # only meaningful associations
                rules.append({
                    "product_a":  prod_a,
                    "product_b":  prod_b,
                    "support":    round(support, 4),
                    "confidence": round(confidence_ab, 4),
                    "lift":       round(lift, 3),
                })

    if not rules:
        return pd.DataFrame()

    rules_df = pd.DataFrame(rules)
    return rules_df.sort_values("lift", ascending=False).head(30)


def get_recommendations_for_product(rules_df: pd.DataFrame, product: str, n: int = 5) -> list:
    """
    Returns top N product recommendations for a given product.

    Args:
        rules_df: Output from find_product_associations()
        product: Product to find recommendations for
        n: Number of recommendations

    Returns:
        List of (recommended_product, lift_score) tuples
    """
    if rules_df.empty:
        return []

    # Find rules where our product is on the left side
    recs = rules_df[rules_df["product_a"] == product][["product_b", "lift", "confidence"]]

    # Also check where it's on the right side
    recs_b = rules_df[rules_df["product_b"] == product][["product_a", "lift", "confidence"]]
    recs_b = recs_b.rename(columns={"product_a": "product_b"})

    combined = pd.concat([recs, recs_b], ignore_index=True)
    combined = combined.sort_values("lift", ascending=False).head(n)

    return list(zip(combined["product_b"], combined["lift"], combined["confidence"]))


def product_performance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores each product on multiple dimensions to identify:
      - Stars: High revenue + high growth
      - Steady performers: Consistent but not growing
      - Declining: Revenue dropping month over month
      - Underperformers: Low revenue + low frequency

    HOW SCORING WORKS:
      1. Calculate revenue, order count, avg price, recent trend for each product
      2. Normalize each metric to 0-100 scale
      3. Composite score = weighted average of all metrics
      4. Classify into categories

    Returns:
        DataFrame with product scores and categories
    """
    if "product" not in df.columns or "revenue" not in df.columns:
        return pd.DataFrame()

    agg = {"revenue": ["sum", "count", "mean"]}
    perf = df.groupby("product").agg(agg)
    perf.columns = ["total_revenue", "order_count", "avg_price"]
    perf = perf.reset_index()

    # Calculate recent trend (last 30 days vs previous 30 days)
    if "date" in df.columns:
        cutoff = df["date"].max() - pd.Timedelta(days=30)
        cutoff_prev = cutoff - pd.Timedelta(days=30)

        recent = df[df["date"] >= cutoff].groupby("product")["revenue"].sum()
        previous = df[(df["date"] >= cutoff_prev) & (df["date"] < cutoff)].groupby("product")["revenue"].sum()

        trend = ((recent - previous) / previous.replace(0, np.nan) * 100).fillna(0)
        perf["trend_pct"] = perf["product"].map(trend).fillna(0).round(1)
    else:
        perf["trend_pct"] = 0

    # Normalize metrics to 0-100
    def normalize(series):
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series([50] * len(series), index=series.index)
        return ((series - min_val) / (max_val - min_val) * 100).round(1)

    perf["revenue_score"]   = normalize(perf["total_revenue"])
    perf["frequency_score"] = normalize(perf["order_count"])
    perf["trend_score"]     = normalize(perf["trend_pct"])

    # Composite score: revenue matters most (50%), frequency (30%), trend (20%)
    perf["composite_score"] = (
        perf["revenue_score"]   * 0.50 +
        perf["frequency_score"] * 0.30 +
        perf["trend_score"]     * 0.20
    ).round(1)

    # Classify products
    def classify(row):
        if row["composite_score"] >= 70 and row["trend_pct"] > 0:
            return "⭐ Star"
        elif row["composite_score"] >= 70:
            return "💪 Steady Performer"
        elif row["composite_score"] >= 40:
            return "📊 Average"
        elif row["trend_pct"] < -20:
            return "📉 Declining"
        else:
            return "⚠️ Underperformer"

    perf["category"] = perf.apply(classify, axis=1)
    return perf.sort_values("composite_score", ascending=False)
