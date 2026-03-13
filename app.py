# =============================================================================
# app.py  —  CommercePulse AI: Main Entry Point
# =============================================================================
# PURPOSE:
#   This is the FIRST page users see. It handles:
#     1. CSV file upload (or sample dataset selection)
#     2. 3-Layer column detection with manual override UI
#     3. Data preview and feature availability summary
#     4. Stores data in session_state so ALL pages can access it
#
# HOW STREAMLIT MULTI-PAGE APPS WORK:
#   - app.py is the home page
#   - Files inside pages/ folder automatically become sidebar navigation items
#   - st.session_state is a dictionary that persists across pages
#
# RUN WITH:
#   streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd

from utils.currency import detect_currency, render_currency_selector, fmt, get_currency_symbol
from utils.data_loader import (
    load_uploaded_file,
    load_sample_dataset,
    get_column_summary,
    render_column_mapper,
    clean_dataframe,
    STANDARD_COLS,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="CommercePulse AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .col-detected { background:#d4edda; color:#155724; padding:3px 10px;
        border-radius:20px; font-size:0.85rem; margin:2px; display:inline-block; }
    .col-missing  { background:#f8d7da; color:#721c24; padding:3px 10px;
        border-radius:20px; font-size:0.85rem; margin:2px; display:inline-block; }
    .layer-badge  { background:#e8f4fd; color:#1a6496; padding:2px 8px;
        border-radius:10px; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<p class="main-header">🛒 CommercePulse AI</p>', unsafe_allow_html=True)
st.markdown("**E-Commerce Sales & Customer Intelligence Platform** — Upload any dataset, get instant ML-powered insights")
st.divider()

# =============================================================================
# DATA SOURCE SELECTION
# =============================================================================
st.subheader("📂 Load Your Data")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### Option 1: Upload Your CSV")
    st.caption("Works with any e-commerce CSV — auto-detects columns using 3-layer matching")

    uploaded_file = st.file_uploader(
        label="Drop your CSV here",
        type=["csv"],
        help="The app uses Exact Match → Fuzzy Match → Content Sniffing to map any column names",
    )

    if uploaded_file is not None:
        with st.spinner("🔄 Loading and detecting columns..."):
            df, detection_report = load_uploaded_file(uploaded_file)

        if df is not None and not df.empty:
            st.session_state["df_raw"] = df
            st.session_state["detection_report"] = detection_report
            st.session_state["data_source"] = uploaded_file.name
            # Detect currency from raw file text + region column
            detected = detect_currency(df, uploaded_file.name, uploaded_file)
            st.session_state["currency_code"] = detected
            st.success(f"✅ Loaded **{len(df):,} rows** from `{uploaded_file.name}`")
        else:
            st.error("❌ Could not process the file.")

with col2:
    st.markdown("#### Option 2: Use a Sample Dataset")
    st.caption("Pre-loaded Kaggle datasets for exploration")

    sample_choice = st.selectbox(
        "Choose a dataset",
        options=["— Select —", "UK Online Retail", "US E-Commerce (Demo)"],
    )

    if st.button("📦 Load Sample Dataset", use_container_width=True):
        if sample_choice == "— Select —":
            st.warning("Please select a dataset first.")
        else:
            with st.spinner(f"Loading {sample_choice}..."):
                df, detection_report = load_sample_dataset(sample_choice)
            if df is not None and not df.empty:
                st.session_state["df_raw"] = df
                st.session_state["detection_report"] = detection_report
                st.session_state["data_source"] = sample_choice
                # Detect currency from region column + source name
                detected = detect_currency(df, source_name=sample_choice)
                st.session_state["currency_code"] = detected
                st.success(f"✅ Loaded **{len(df):,} rows** from {sample_choice}")
            else:
                st.error(
                    "Sample dataset not found. Download from Kaggle and place in `sample_data/`.\n\n"
                    "**UK Online Retail**: https://www.kaggle.com/datasets/carrie1/ecommerce-data\n\n"
                    "**Superstore**: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final"
                )

st.divider()

# =============================================================================
# COLUMN MAPPING UI + DATA PREVIEW
# Only shown after data is loaded
# =============================================================================
if "df_raw" in st.session_state and st.session_state["df_raw"] is not None:
    df = st.session_state["df_raw"]
    report = st.session_state.get("detection_report", {})

    # ── Detection Summary ────────────────────────────────────────────────────
    st.subheader("🔍 Column Detection Results")

    total_mapped = len(report.get("mapped", {})) + len(report.get("fuzzy", {})) + len(report.get("sniffed", {}))
    has_uncertain = bool(report.get("suggestions")) or bool(report.get("unmapped"))

    det_col1, det_col2, det_col3, det_col4 = st.columns(4)
    det_col1.metric("Columns in CSV",     len([c for c in df.columns if not c.startswith("unnamed")]))
    det_col2.metric("Auto-Mapped",        len(report.get("mapped", {})), help="Exact alias matches")
    det_col3.metric("Fuzzy-Matched",      len(report.get("fuzzy", {})),  help="Similar name matches")
    det_col4.metric("Content-Sniffed",    len(report.get("sniffed", {})),help="Detected from data values")

    # Show detection method for each mapped column
    with st.expander("📋 How each column was detected", expanded=False):
        rows = []
        for orig, std in report.get("mapped", {}).items():
            rows.append({"Your Column": orig, "Mapped To": std, "Method": "✅ Exact Match"})
        for orig, std in report.get("fuzzy", {}).items():
            rows.append({"Your Column": orig, "Mapped To": std, "Method": "🔍 Fuzzy Match"})
        for orig, std in report.get("sniffed", {}).items():
            rows.append({"Your Column": orig, "Mapped To": std, "Method": "🧪 Content Sniff"})
        for orig in report.get("unmapped", []):
            rows.append({"Your Column": orig, "Mapped To": "—", "Method": "❌ Not Mapped"})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Manual Override UI (shown when there are uncertain/unmapped columns) ─
    if has_uncertain:
        st.warning("⚠️ Some columns couldn't be mapped automatically. Please review below.")
        df = render_column_mapper(df, report)
        df = clean_dataframe(df)

    # Save final mapped df to session_state["df"] — used by all pages
    st.session_state["df"] = df

    st.divider()

    # ── Data Overview ────────────────────────────────────────────────────────
    st.subheader("📊 Data Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{len(df):,}")
    m2.metric("Columns",    len(df.columns))
    m3.metric("Date Range",
              f"{df['date'].min().strftime('%b %Y')} → {df['date'].max().strftime('%b %Y')}"
              if "date" in df.columns else "N/A")
    m4.metric("Source",     st.session_state.get("data_source", "Unknown"))

    # ── Standard Column Availability ─────────────────────────────────────────
    st.markdown("#### Standard Columns Available")
    col_summary = get_column_summary(df)
    cols_disp = st.columns(4)
    for i, (col_name, detected) in enumerate(col_summary.items()):
        with cols_disp[i % 4]:
            if detected:
                st.markdown(f'<span class="col-detected">✓ {col_name}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="col-missing">✗ {col_name}</span>', unsafe_allow_html=True)

    st.caption("✓ = available for analysis | ✗ = not found (related features will be limited)")

    # ── Data Preview ─────────────────────────────────────────────────────────
    st.markdown("#### 👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    # ── Available Modules ────────────────────────────────────────────────────
    st.markdown("#### 🚀 Available Analysis Modules")
    f1, f2, f3 = st.columns(3)
    with f1:
        if col_summary.get("date") and col_summary.get("revenue"):
            st.success("✅ Sales Dashboard")
            st.success("✅ Forecasting")
        else:
            st.error("❌ Sales Dashboard (needs date + revenue)")
            st.error("❌ Forecasting (needs date + revenue)")
    with f2:
        if col_summary.get("customer_id"):
            st.success("✅ Customer Intelligence")
            st.success("✅ Churn Prediction")
        else:
            st.warning("⚠️ Customer Intelligence (needs customer_id)")
            st.warning("⚠️ Churn Prediction (needs customer_id)")
    with f3:
        if col_summary.get("product"):
            st.success("✅ Product Intelligence")
        else:
            st.warning("⚠️ Product Intelligence (needs product)")
        if col_summary.get("product") and col_summary.get("order_id"):
            st.success("✅ Market Basket")
        else:
            st.warning("⚠️ Market Basket (needs product + order_id)")

    # Currency override widget — user can correct if auto-detection is wrong
    final_code = render_currency_selector(st.session_state.get("currency_code", "USD"))
    st.session_state["currency_code"] = final_code

    st.divider()
    st.info("👈 Use the **sidebar** to navigate to any analysis module!")

else:
    # Welcome screen
    st.markdown("""
    ### 👋 Welcome to CommercePulse AI!

    Upload **any** e-commerce CSV — the app intelligently maps columns using a 3-layer system:

    | Layer | Method | Example |
    |-------|--------|---------|
    | 1️⃣ Exact Match | Checks 80+ known column aliases | `InvoiceDate` → `date` |
    | 2️⃣ Fuzzy Match | String similarity (≥82% match) | `order_dt` → `date` |
    | 3️⃣ Content Sniff | Reads actual values for clues | Column with dates → `date` |

    Then provides ML-powered insights — no AI API needed!
    """)

    with st.expander("📋 Recognized column name variations"):
        st.markdown("""
        | Standard | Recognized variations |
        |----------|-----------------------|
        | `date` | OrderDate, purchase_date, InvoiceDate, sale_date, created_at, timestamp... |
        | `revenue` | Amount, Total, UnitPrice, TotalPrice, Sales, net_sales, payment, value... |
        | `customer_id` | CustomerID, client_id, user_id, customer_name, account_id, member_id... |
        | `product` | Description, ProductName, SKU, StockCode, item_name, title, ASIN... |
        | `category` | Category, Department, Sub-Category, product_type, brand, division... |
        | `region` | Country, State, City, Location, territory, ship_country, geography... |
        | `order_id` | OrderID, InvoiceNo, transaction_id, receipt_id, basket_id... |
        """)

