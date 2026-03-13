# =============================================================================
# utils/segmentation.py
# =============================================================================
# PURPOSE:
#   Implements RFM (Recency, Frequency, Monetary) Analysis + K-Means Clustering
#   to segment customers into meaningful groups.
#
# WHAT IS RFM?
#   R = Recency   → How recently did the customer buy? (days since last purchase)
#   F = Frequency → How often do they buy? (number of orders)
#   M = Monetary  → How much do they spend? (total revenue)
#
# WHAT IS K-MEANS CLUSTERING?
#   An unsupervised ML algorithm that groups customers into K clusters based on
#   their RFM scores. Customers with similar buying behavior are grouped together.
#
# SEGMENTS PRODUCED:
#   - Champions      : High R, High F, High M → Best customers
#   - Loyal          : Medium-High on all → Regular buyers
#   - At Risk        : Good past behavior but haven't bought recently
#   - Lost/Inactive  : Low recency, low frequency, low spend
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes RFM scores for each customer.

    HOW IT WORKS:
      1. Group orders by customer_id
      2. For each customer compute:
         - Recency: days since their last order (lower = more recent = better)
         - Frequency: how many unique orders they placed
         - Monetary: total money they spent
      3. Return a DataFrame with one row per customer

    Args:
        df: Main cleaned DataFrame (must have date, customer_id, revenue, order_id columns)

    Returns:
        DataFrame with columns: [customer_id, recency, frequency, monetary]
    """
    if not all(col in df.columns for col in ["date", "customer_id", "revenue"]):
        return pd.DataFrame()

    # "Today" = day after the last order in dataset (standard RFM convention)
    reference_date = df["date"].max() + pd.Timedelta(days=1)

    agg = {
        "date":    lambda x: (reference_date - x.max()).days,  # Recency
        "revenue": "sum",                                        # Monetary
    }
    if "order_id" in df.columns:
        agg["order_id"] = "nunique"  # Frequency (unique orders)
    else:
        agg["revenue_count"] = "count"  # fallback: count rows

    rfm = df.groupby("customer_id").agg(agg).reset_index()
    rfm.columns = ["customer_id", "recency", "monetary",
                   "frequency" if "order_id" in df.columns else "frequency"]

    # Ensure correct column order
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
    return rfm


def apply_kmeans_segmentation(rfm: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Applies K-Means clustering on RFM scores to create customer segments.

    HOW K-MEANS WORKS:
      1. Standardize RFM values (so monetary doesn't dominate due to large numbers)
      2. Initialize K cluster centers randomly
      3. Assign each customer to nearest cluster center
      4. Recalculate cluster centers
      5. Repeat until clusters stabilize
      6. Label clusters based on their average RFM profiles

    Args:
        rfm: DataFrame from compute_rfm()
        n_clusters: Number of segments to create (default=4)

    Returns:
        rfm DataFrame with added columns: [cluster, segment]
    """
    if rfm.empty or len(rfm) < n_clusters:
        return rfm

    # Step 1: Scale the features so they're on the same scale
    # Without scaling: monetary (₹50,000) would dominate over recency (30 days)
    scaler = StandardScaler()
    features = ["recency", "frequency", "monetary"]
    rfm_scaled = scaler.fit_transform(rfm[features])

    # Step 2: Run K-Means
    # random_state=42 ensures reproducible results every run
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Step 3: Label clusters based on their characteristics
    # We look at the average recency and monetary value of each cluster
    cluster_summary = rfm.groupby("cluster")[features].mean()

    # Sort clusters: low recency (recent buyers) + high monetary = Champions
    # We rank clusters and assign labels based on their relative scores
    cluster_summary["score"] = (
        cluster_summary["monetary"].rank()          # higher monetary = better
        + cluster_summary["frequency"].rank()       # higher frequency = better
        - cluster_summary["recency"].rank()         # lower recency = better (more recent)
    )
    cluster_summary = cluster_summary.sort_values("score", ascending=False)

    # Map cluster numbers to human-readable labels
    label_map_keys = cluster_summary.index.tolist()
    labels = ["🏆 Champions", "💚 Loyal Customers", "⚠️ At Risk", "😴 Lost/Inactive"]
    label_map = {cluster: label for cluster, label in zip(label_map_keys, labels[:n_clusters])}

    rfm["segment"] = rfm["cluster"].map(label_map)
    return rfm


def get_segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary table showing characteristics of each segment.

    Returns:
        DataFrame showing avg recency, frequency, monetary per segment + customer count
    """
    if "segment" not in rfm.columns:
        return pd.DataFrame()

    summary = rfm.groupby("segment").agg(
        customers=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        total_revenue=("monetary", "sum"),
    ).reset_index()

    summary["avg_recency"] = summary["avg_recency"].round(0).astype(int)
    summary["avg_frequency"] = summary["avg_frequency"].round(1)
    summary["avg_monetary"] = summary["avg_monetary"].round(2)
    return summary.sort_values("total_revenue", ascending=False)


def generate_segment_insights(summary: pd.DataFrame) -> list:
    """
    Generates actionable insights and recommendations for each segment.
    fmt() is imported lazily so it only runs inside a live Streamlit session.
    """
    from utils.currency import fmt  # lazy import - needs active Streamlit session
    insights = []
    if summary.empty:
        return insights

    for _, row in summary.iterrows():
        seg = row["segment"]
        count = int(row["customers"])
        avg_spend = row["avg_monetary"]

        if "Champions" in seg:
            insights.append(
                f"**{seg}** ({count} customers): Your best customers averaging "
                f"{fmt(avg_spend)} spend. Reward them with loyalty perks to retain them."
            )
        elif "Loyal" in seg:
            insights.append(
                f"**{seg}** ({count} customers): Consistent buyers with {fmt(avg_spend)} avg spend. "
                f"Upsell premium products to move them to Champions."
            )
        elif "At Risk" in seg:
            insights.append(
                f"**{seg}** ({count} customers): Previously good customers going quiet. "
                f"Send a win-back campaign with a discount before they churn."
            )
        elif "Lost" in seg:
            insights.append(
                f"**{seg}** ({count} customers): Inactive customers with only {fmt(avg_spend)} avg spend. "
                f"Run a re-engagement email with a strong offer."
            )

    return insights
