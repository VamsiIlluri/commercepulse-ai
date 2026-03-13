# =============================================================================
# utils/churn_model.py
# =============================================================================
# PURPOSE:
#   Predicts which customers are likely to stop buying (churn) using a
#   Random Forest Classifier — a machine learning model.
#
# WHAT IS CHURN?
#   A customer "churns" when they stop purchasing. For e-commerce:
#   If a customer hasn't bought in 90+ days, they're at risk of never returning.
#
# HOW THE MODEL WORKS:
#   1. CREATE LABELS: Customers who haven't bought in last 90 days = churned (1)
#      Customers who bought in last 90 days = active (0)
#   2. CREATE FEATURES: For each customer compute behavioral signals:
#      - days_since_last_order  (recency)
#      - total_orders           (how often they buy)
#      - avg_order_value        (how much they spend per order)
#      - total_revenue          (lifetime value)
#      - days_since_first_order (how long they've been a customer)
#      - order_frequency_rate   (orders per month)
#   3. TRAIN Random Forest on 80% of data
#   4. PREDICT churn probability on remaining 20%
#   5. Return each customer's churn probability (0.0 to 1.0)
#
# RANDOM FOREST:
#   Builds many decision trees (e.g., 100 trees), each trained on a random
#   subset of the data. Final prediction = average of all trees.
#   More robust than a single decision tree.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler


# Threshold: customers not buying in CHURN_DAYS are labeled as churned
CHURN_DAYS = 90


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for each customer that the ML model will use to
    predict churn.

    FEATURE ENGINEERING EXPLAINED:
      - days_since_last_order: More days = higher churn risk
      - total_orders: Fewer orders = less engaged = higher churn risk
      - avg_order_value: Lower value = less invested customer
      - total_revenue: Total lifetime spend
      - customer_age_days: How long since first purchase
      - order_frequency: Orders per 30 days (engagement rate)

    Args:
        df: Main DataFrame with date, customer_id, revenue, order_id

    Returns:
        DataFrame with one row per customer and engineered features
    """
    required = ["date", "customer_id", "revenue"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    reference_date = df["date"].max()

    # Build per-customer aggregations
    agg_dict = {
        "date":    ["max", "min", "count"],
        "revenue": ["sum", "mean"],
    }
    if "order_id" in df.columns:
        agg_dict["order_id"] = "nunique"

    features = df.groupby("customer_id").agg(agg_dict)
    features.columns = ["_".join(c).strip() for c in features.columns]
    features = features.reset_index()

    # Rename to readable names
    rename = {
        "date_max":          "last_order_date",
        "date_min":          "first_order_date",
        "date_count":        "total_transactions",
        "revenue_sum":       "total_revenue",
        "revenue_mean":      "avg_order_value",
    }
    if "order_id_nunique" in features.columns:
        rename["order_id_nunique"] = "total_orders"
    else:
        features["total_orders"] = features["total_transactions"]
    features = features.rename(columns=rename)

    # Compute derived features
    features["days_since_last_order"] = (reference_date - features["last_order_date"]).dt.days
    features["customer_age_days"] = (reference_date - features["first_order_date"]).dt.days
    features["customer_age_days"] = features["customer_age_days"].clip(lower=1)

    # Order frequency = orders per 30 days (normalized by how long they've been a customer)
    features["order_frequency"] = (
        features["total_orders"] / features["customer_age_days"] * 30
    ).round(4)

    # Drop date columns (model doesn't need raw dates, only derived features)
    features = features.drop(columns=["last_order_date", "first_order_date", "total_transactions"], errors="ignore")

    return features


def label_churned_customers(features: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary churn labels:
      - 1 = churned (hasn't ordered in CHURN_DAYS days)
      - 0 = active (ordered within CHURN_DAYS days)

    This is our "ground truth" that the model learns from.
    """
    features = features.copy()
    features["churned"] = (features["days_since_last_order"] >= CHURN_DAYS).astype(int)
    return features


def train_churn_model(features: pd.DataFrame):
    """
    Trains a Random Forest classifier on customer features.

    TRAINING PROCESS:
      1. Split data: 80% training, 20% testing
      2. Scale features (normalize values)
      3. Train Random Forest (100 decision trees)
      4. Evaluate on test set (AUC score)

    Args:
        features: DataFrame from build_customer_features() with churned column

    Returns:
        Tuple of (trained_model, scaler, feature_columns, metrics_dict)
    """
    feature_cols = [
        "days_since_last_order", "total_orders", "avg_order_value",
        "total_revenue", "customer_age_days", "order_frequency"
    ]
    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in features.columns]

    X = features[feature_cols].fillna(0)
    y = features["churned"]

    # Need at least 2 classes and enough samples
    if y.nunique() < 2 or len(X) < 10:
        return None, None, feature_cols, {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    # n_estimators=100 → builds 100 trees (more = more accurate but slower)
    # class_weight="balanced" → handles imbalanced churn data (usually few churners)
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        max_depth=6,  # prevent overfitting
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {}
    try:
        metrics["auc"] = round(roc_auc_score(y_test, y_prob), 3)
    except Exception:
        metrics["auc"] = "N/A"

    metrics["accuracy"] = round((y_pred == y_test).mean(), 3)

    return model, scaler, feature_cols, metrics


def predict_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    MAIN FUNCTION: End-to-end churn prediction pipeline.

    Steps:
      1. Build customer features
      2. Label churned/active
      3. Train model
      4. Predict churn probability for all customers
      5. Assign risk category

    Args:
        df: Main cleaned DataFrame

    Returns:
        DataFrame with customer_id, churn_probability, risk_level, and features
    """
    features = build_customer_features(df)
    if features.empty:
        return pd.DataFrame()

    features = label_churned_customers(features)

    model, scaler, feature_cols, metrics = train_churn_model(features)

    if model is None:
        # Fallback: use rule-based scoring if not enough data for ML
        features["churn_probability"] = (
            features["days_since_last_order"] / features["days_since_last_order"].max()
        ).round(3)
    else:
        X_all = features[feature_cols].fillna(0)
        X_all_scaled = scaler.transform(X_all)
        features["churn_probability"] = model.predict_proba(X_all_scaled)[:, 1].round(3)

    # Assign risk categories based on probability
    def categorize_risk(prob):
        if prob >= 0.75:
            return "🔴 High Risk"
        elif prob >= 0.45:
            return "🟠 Medium Risk"
        elif prob >= 0.20:
            return "🟡 Low Risk"
        else:
            return "🟢 Safe"

    features["risk_level"] = features["churn_probability"].apply(categorize_risk)
    features["model_metrics"] = str(metrics)

    return features.sort_values("churn_probability", ascending=False)


def get_churn_insights(churn_df: pd.DataFrame) -> list:
    """
    Generates actionable insights from churn predictions.

    Returns:
        List of insight strings
    """
    insights = []
    if churn_df.empty:
        return insights

    total = len(churn_df)
    high_risk = (churn_df["churn_probability"] >= 0.75).sum()
    medium_risk = ((churn_df["churn_probability"] >= 0.45) & (churn_df["churn_probability"] < 0.75)).sum()

    insights.append(
        f"🔴 **{high_risk} customers** ({high_risk/total*100:.1f}%) are at HIGH risk of churning. "
        f"Immediate win-back campaign recommended."
    )
    insights.append(
        f"🟠 **{medium_risk} customers** ({medium_risk/total*100:.1f}%) are at MEDIUM risk. "
        f"A targeted discount could re-engage them."
    )

    if "avg_order_value" in churn_df.columns:
        high_value_churners = churn_df[
            (churn_df["churn_probability"] >= 0.75) &
            (churn_df["avg_order_value"] > churn_df["avg_order_value"].median())
        ]
        if len(high_value_churners) > 0:
            insights.append(
                f"💰 **{len(high_value_churners)} high-value customers** are at high churn risk. "
                f"Priority: reach out personally with personalized offers."
            )

    return insights
