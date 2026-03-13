# =============================================================================
# utils/metrics.py  —  KPI Calculations & Auto Insights
# =============================================================================
# NOTE: fmt() uses st.session_state["currency_code"] set on the home page.
#       fmt() is imported LAZILY inside functions (not at module top level)
#       so it only runs inside a live Streamlit session.
# =============================================================================

import pandas as pd
import numpy as np


# =============================================================================
# CORE KPI FUNCTIONS
# =============================================================================

def total_revenue(df):
    try:
        return float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
    except Exception:
        return 0.0


def total_orders(df):
    try:
        if "order_id" in df.columns:
            return int(df["order_id"].nunique())
        return len(df)
    except Exception:
        return 0


def total_customers(df):
    try:
        if "customer_id" in df.columns:
            return int(df["customer_id"].nunique())
        return 0
    except Exception:
        return 0


def average_order_value(df):
    try:
        orders = total_orders(df)
        return total_revenue(df) / orders if orders > 0 else 0.0
    except Exception:
        return 0.0


def revenue_growth_rate(df):
    """Month-over-Month revenue growth %."""
    try:
        if "date" not in df.columns or "revenue" not in df.columns:
            return 0.0
        df = df.copy()
        df["month"] = df["date"].dt.to_period("M")
        monthly = df.groupby("month")["revenue"].sum().sort_index()
        if len(monthly) < 2:
            return 0.0
        current, previous = monthly.iloc[-1], monthly.iloc[-2]
        if previous == 0:
            return 0.0
        return round(((current - previous) / previous) * 100, 2)
    except Exception:
        return 0.0


def customer_growth_rate(df):
    """Month-over-Month new customer growth %."""
    try:
        if "date" not in df.columns or "customer_id" not in df.columns:
            return 0.0
        df = df.copy()
        first_purchase = df.groupby("customer_id")["date"].min().reset_index()
        first_purchase["month"] = first_purchase["date"].dt.to_period("M")
        monthly_new = first_purchase.groupby("month").size().sort_index()
        if len(monthly_new) < 2:
            return 0.0
        current, previous = monthly_new.iloc[-1], monthly_new.iloc[-2]
        if previous == 0:
            return 0.0
        return round(((current - previous) / previous) * 100, 2)
    except Exception:
        return 0.0


# =============================================================================
# TIME-SERIES AGGREGATIONS
# =============================================================================

def daily_revenue(df):
    try:
        if "date" not in df.columns or "revenue" not in df.columns:
            return pd.DataFrame()
        daily = df.groupby(df["date"].dt.date)["revenue"].sum().reset_index()
        daily.columns = ["date", "revenue"]
        daily["date"] = pd.to_datetime(daily["date"])
        return daily.sort_values("date")
    except Exception:
        return pd.DataFrame()


def weekly_revenue(df):
    try:
        if "date" not in df.columns or "revenue" not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df["week"] = df["date"].dt.to_period("W")
        weekly = df.groupby("week")["revenue"].sum().reset_index()
        weekly["week"] = weekly["week"].dt.to_timestamp()
        return weekly.sort_values("week")
    except Exception:
        return pd.DataFrame()


def monthly_revenue(df):
    try:
        if "date" not in df.columns or "revenue" not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df["month"] = df["date"].dt.to_period("M")
        agg = {"revenue": "sum"}
        if "order_id" in df.columns:
            agg["order_id"] = "nunique"
        if "customer_id" in df.columns:
            agg["customer_id"] = "nunique"
        monthly = df.groupby("month").agg(agg).reset_index()
        monthly["month"] = monthly["month"].dt.to_timestamp()
        monthly = monthly.rename(columns={"order_id": "orders", "customer_id": "customers"})
        return monthly.sort_values("month")
    except Exception:
        return pd.DataFrame()


# =============================================================================
# PRODUCT & CATEGORY ANALYSIS
# =============================================================================

def top_products(df, n=10):
    try:
        if "product" not in df.columns or "revenue" not in df.columns:
            return pd.DataFrame()
        agg = {"revenue": "sum"}
        if "quantity" in df.columns:
            agg["quantity"] = "sum"
        if "order_id" in df.columns:
            agg["order_id"] = "nunique"
        top = df.groupby("product").agg(agg).reset_index()
        top = top.rename(columns={"order_id": "orders"})
        return top.sort_values("revenue", ascending=False).head(n)
    except Exception:
        return pd.DataFrame()


def revenue_by_category(df):
    try:
        if "category" not in df.columns or "revenue" not in df.columns:
            return pd.DataFrame()
        cat = df.groupby("category")["revenue"].sum().reset_index()
        cat["percentage"] = (cat["revenue"] / cat["revenue"].sum() * 100).round(2)
        return cat.sort_values("revenue", ascending=False)
    except Exception:
        return pd.DataFrame()


def revenue_by_region(df):
    try:
        if "region" not in df.columns or "revenue" not in df.columns:
            return pd.DataFrame()
        reg = df.groupby("region")["revenue"].sum().reset_index()
        reg["percentage"] = (reg["revenue"] / reg["revenue"].sum() * 100).round(2)
        return reg.sort_values("revenue", ascending=False)
    except Exception:
        return pd.DataFrame()


# =============================================================================
# INSIGHT GENERATOR
# fmt() is imported lazily — only called inside a live Streamlit session
# =============================================================================

def generate_kpi_insights(df):
    """
    Generates plain-English insights from the data.
    Uses lazy import of fmt() so it only runs inside an active Streamlit session
    where st.session_state["currency_code"] is already set.
    """
    from utils.currency import fmt   # <-- lazy import, safe inside Streamlit session

    insights = []

    try:
        growth = revenue_growth_rate(df)
        if growth > 10:
            insights.append(f"🟢 Strong growth! Revenue is up **{growth:.1f}%** vs last month.")
        elif growth > 0:
            insights.append(f"🟡 Modest growth of **{growth:.1f}%** vs last month.")
        elif growth < -10:
            insights.append(f"🔴 Revenue dropped **{abs(growth):.1f}%** vs last month. Needs attention.")
        elif growth < 0:
            insights.append(f"🟠 Slight decline of **{abs(growth):.1f}%** vs last month.")
    except Exception:
        pass

    try:
        monthly = monthly_revenue(df)
        if not monthly.empty:
            best = monthly.loc[monthly["revenue"].idxmax()]
            insights.append(
                f"📅 Best month: **{best['month'].strftime('%B %Y')}** "
                f"with {fmt(best['revenue'])} in revenue."
            )
    except Exception:
        pass

    try:
        top = top_products(df, n=1)
        if not top.empty:
            insights.append(
                f"🏆 Top product: **{top.iloc[0]['product']}** "
                f"generating {fmt(top.iloc[0]['revenue'])} in revenue."
            )
    except Exception:
        pass

    try:
        cust_growth = customer_growth_rate(df)
        if cust_growth > 0:
            insights.append(f"👥 Customer acquisition grew **{cust_growth:.1f}%** this month.")
        elif cust_growth < 0:
            insights.append(f"👥 New customer acquisition down **{abs(cust_growth):.1f}%** this month.")
    except Exception:
        pass

    try:
        aov = average_order_value(df)
        insights.append(f"🛒 Average Order Value: **{fmt(aov, 2)}** per transaction.")
    except Exception:
        pass

    return insights
