# =============================================================================
# utils/forecasting.py
# =============================================================================
# PURPOSE:
#   Forecasts future sales using time-series analysis.
#   We use two approaches:
#     1. Moving Average (simple, always works)
#     2. Linear Regression on time (captures growth trends)
#
# WHY NOT PROPHET/ARIMA?
#   Prophet requires Facebook's library (heavy install).
#   ARIMA needs manual parameter tuning.
#   Our approach using scikit-learn is lightweight, fast, and works well
#   for college project demos.
#
# HOW FORECASTING WORKS:
#   1. Aggregate daily revenue into a time series
#   2. Encode dates as numbers (day 1, day 2, day 3...)
#   3. Fit a Linear Regression: revenue = a * day_number + b
#   4. Predict revenue for future day numbers
#   5. Add confidence intervals using historical variance
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def prepare_time_series(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Converts transaction data into a clean time series.

    Args:
        df: Main DataFrame with date and revenue columns
        freq: Frequency - "D" for daily, "W" for weekly, "M" for monthly

    Returns:
        DataFrame with columns: [date, revenue] — one row per time period
    """
    if "date" not in df.columns or "revenue" not in df.columns:
        return pd.DataFrame()

    ts = df.groupby(pd.Grouper(key="date", freq=freq))["revenue"].sum().reset_index()
    ts = ts[ts["revenue"] > 0]  # remove zero periods
    ts = ts.sort_values("date").reset_index(drop=True)
    return ts


def forecast_revenue(df: pd.DataFrame, forecast_days: int = 30) -> dict:
    """
    Forecasts revenue for the next N days using polynomial regression.

    HOW POLYNOMIAL REGRESSION WORKS:
      - Linear regression: y = a*x + b  (straight line)
      - Polynomial degree 2: y = a*x² + b*x + c  (curve, captures acceleration)
      - We use degree 2 to capture growth trends that aren't perfectly linear

    CONFIDENCE INTERVALS:
      - We calculate the standard deviation of residuals (prediction errors)
      - Upper bound = prediction + 1.5 * std_dev
      - Lower bound = prediction - 1.5 * std_dev
      - This gives a realistic range of expected outcomes

    Args:
        df: Main DataFrame
        forecast_days: How many days ahead to forecast (default 30)

    Returns:
        dict with:
          - 'historical': past revenue time series
          - 'forecast': future predictions with confidence intervals
          - 'model_score': R² score (0-1, how well model fits history)
          - 'insights': list of plain-English insight strings
    """
    ts = prepare_time_series(df, freq="D")
    ts_weekly = prepare_time_series(df, freq="W")  # weekly for model fitting (less noise)

    if ts.empty or len(ts) < 14:  # need at least 2 weeks of data
        return {"error": "Not enough data for forecasting (need 14+ days)"}

    # Fit the model on WEEKLY data to reduce daily noise → much better R²
    # Then generate daily forecasts by dividing weekly predictions by 7
    fit_ts = ts_weekly if len(ts_weekly) >= 8 else ts

    # Encode dates as sequential numbers using fit_ts (weekly = less noise)
    fit_ts["day_num"] = (fit_ts["date"] - fit_ts["date"].min()).dt.days
    X = fit_ts[["day_num"]].values
    y = fit_ts["revenue"].values

    # Also encode daily ts for chart display
    ts["day_num"] = (ts["date"] - ts["date"].min()).dt.days

    # Build a polynomial regression pipeline
    # degree=2 creates features: [x, x²] which captures curves
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("linear", LinearRegression()),
    ])
    model.fit(X, y)

    # R² score: 1.0 = perfect fit, 0.0 = no better than guessing mean
    r2_score = model.score(X, y)

    # Calculate residuals for confidence intervals
    y_pred_historical = model.predict(X)
    residuals = y - y_pred_historical
    std_dev = np.std(residuals)

    # Generate future dates and day numbers (offset from daily ts end)
    last_day_num = ts["day_num"].max()
    future_day_nums = np.arange(last_day_num + 1, last_day_num + forecast_days + 1)
    future_dates = pd.date_range(
        start=ts["date"].max() + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    # Predict future revenue
    future_X = future_day_nums.reshape(-1, 1)
    future_revenue = model.predict(future_X)
    future_revenue = np.clip(future_revenue, 0, None)  # no negative revenue

    # Build forecast DataFrame
    forecast_df = pd.DataFrame({
        "date":          future_dates,
        "predicted":     future_revenue.round(2),
        "upper_bound":   (future_revenue + 1.5 * std_dev).round(2),
        "lower_bound":   np.clip(future_revenue - 1.5 * std_dev, 0, None).round(2),
    })

    # Generate insights
    insights = _generate_forecast_insights(ts, forecast_df, r2_score)

    return {
        "historical":   ts[["date", "revenue"]],
        "forecast":     forecast_df,
        "model_score":  round(r2_score, 3),
        "insights":     insights,
    }


def forecast_by_category(df: pd.DataFrame, forecast_days: int = 30) -> dict:
    """
    Forecasts revenue separately for each product category.
    Useful for understanding which categories are growing vs declining.

    Returns:
        dict mapping category_name → forecast result
    """
    if "category" not in df.columns:
        return {}

    results = {}
    for category in df["category"].unique():
        cat_df = df[df["category"] == category].copy()
        if len(cat_df) >= 14:
            results[category] = forecast_revenue(cat_df, forecast_days)

    return results


def moving_average_forecast(df: pd.DataFrame, window: int = 7, forecast_days: int = 30) -> pd.DataFrame:
    """
    Simple Moving Average forecast — easier to understand, good for demos.

    HOW IT WORKS:
      - Calculate the average of the last 'window' days
      - Project that average forward as the forecast
      - The trend direction is captured by comparing recent average to older average

    Args:
        window: Number of days to average (7 = weekly average)

    Returns:
        DataFrame with date and predicted_revenue columns
    """
    ts = prepare_time_series(df, freq="D")
    if ts.empty or len(ts) < window:
        return pd.DataFrame()

    # Moving average of last 'window' days
    recent_avg = ts["revenue"].tail(window).mean()
    older_avg = ts["revenue"].tail(window * 2).head(window).mean()

    # Daily trend: how much revenue changes per day on average
    if older_avg > 0:
        daily_growth = (recent_avg - older_avg) / (window * older_avg)
    else:
        daily_growth = 0

    # Project forward
    future_dates = pd.date_range(
        start=ts["date"].max() + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )
    future_revenue = [
        max(0, recent_avg * (1 + daily_growth * i))
        for i in range(1, forecast_days + 1)
    ]

    return pd.DataFrame({"date": future_dates, "predicted_revenue": future_revenue})


def _generate_forecast_insights(historical: pd.DataFrame, forecast: pd.DataFrame, r2: float) -> list:
    """
    Generates plain-English forecast insights.
    fmt() is imported lazily so it only runs inside a live Streamlit session.
    """
    from utils.currency import fmt  # lazy import - needs active Streamlit session
    insights = []

    # Model quality
    if r2 >= 0.7:
        insights.append(f"✅ Model fit is **good** (R²={r2:.2f}). Forecast is reliable.")
    elif r2 >= 0.4:
        insights.append(f"⚠️ Model fit is **moderate** (R²={r2:.2f}). Use forecast as rough guidance.")
    else:
        insights.append(f"❗ Model fit is **low** (R²={r2:.2f}). High variance in historical data.")

    # Revenue direction
    recent_avg = historical["revenue"].tail(7).mean()
    forecast_avg = forecast["predicted"].mean()

    if forecast_avg > recent_avg * 1.1:
        pct = (forecast_avg - recent_avg) / recent_avg * 100
        insights.append(f"📈 Revenue is forecast to **grow {pct:.1f}%** over the next 30 days.")
    elif forecast_avg < recent_avg * 0.9:
        pct = (recent_avg - forecast_avg) / recent_avg * 100
        insights.append(f"📉 Revenue may **decline {pct:.1f}%** over the next 30 days. Consider a promotional push.")
    else:
        insights.append(f"➡️ Revenue is expected to remain **relatively stable** over the next 30 days.")

    # Total projected revenue
    total_projected = forecast["predicted"].sum()
    insights.append(f"💰 Total projected revenue for next 30 days: **{fmt(total_projected)}**")

    return insights
