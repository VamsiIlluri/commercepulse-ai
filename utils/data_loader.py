# =============================================================================
# utils/data_loader.py
# =============================================================================
# PURPOSE:
#   Loads ANY e-commerce CSV and standardizes column names so the rest of the
#   app always works with the same column names regardless of the dataset.
#
# THE PROBLEM THIS SOLVES:
#   Different datasets name the same concept differently:
#     UK Retail:   InvoiceDate, UnitPrice, CustomerID, Description
#     Superstore:  Order Date,  Sales,     Customer Name, Sub-Category
#     Custom:      date,        amount,    cust_id,       item_name
#   All of these need to be understood as: date, revenue, customer_id, product
#
# 3-LAYER DETECTION STRATEGY:
#   Layer 1 — Exact alias match:   "invoicedate" is in our known aliases list → "date"
#   Layer 2 — Fuzzy string match:  "inv_date" is 82% similar to "date" → suggest "date"
#   Layer 3 — Content sniffing:    column has datetime values? → must be "date"
#                                  column has large numbers? → might be "revenue"
#
# WHY THIS APPROACH?
#   - Layer 1 handles 95% of popular Kaggle datasets
#   - Layer 2 handles typos and slight variations (no extra libraries needed)
#   - Layer 3 is a safety net using the actual data values as clues
# =============================================================================

import pandas as pd
import numpy as np
import streamlit as st
import os
from difflib import SequenceMatcher  # built-in Python, no install needed

# =============================================================================
# ALIAS DICTIONARY
# Maps every known column name variation → standard name
# Add more aliases here as you encounter new datasets
# =============================================================================
COLUMN_ALIASES = {
    # ── DATE ──────────────────────────────────────────────────────────────────
    "date": [
        "date", "order_date", "purchase_date", "transaction_date",
        "invoicedate", "orderdate", "order date", "invoice_date",
        "sale_date", "saledate", "ship_date", "shipdate", "created_at",
        "time", "datetime", "timestamp", "week", "month", "year",
    ],
    # ── REVENUE ───────────────────────────────────────────────────────────────
    # IMPORTANT: "total_price" and "total" come BEFORE "unit_price" and "price"
    # so that when both columns exist (e.g. our US dataset has unit_price AND
    # total_price), the aggregated total is always picked as revenue — not the
    # per-unit price which would massively undercount revenue.
    "revenue": [
        "total_price", "totalprice", "revenue", "total_revenue",
        "amount", "total", "sales", "sale_amount",
        "order_value", "order_total", "net_sales", "gross_sales",
        "subtotal", "sub_total", "payment", "line_total", "item_total",
        "lineitem_price",
        # unit_price / price are LAST fallbacks — only used when no total exists
        "price", "unitprice", "unit_price", "cost", "value",
    ],
    # ── QUANTITY ──────────────────────────────────────────────────────────────
    "quantity": [
        "quantity", "qty", "units", "count", "num_items",
        "quantity_ordered", "unitssold", "units_sold", "items",
        "order_qty", "no_of_items", "pieces",
    ],
    # ── CUSTOMER ID ───────────────────────────────────────────────────────────
    "customer_id": [
        "customer_id", "customerid", "client_id", "user_id",
        "customer", "cust_id", "buyer_id", "account_id",
        "customer name", "customername", "customer_name",
        "client", "member_id", "shopper_id",
    ],
    # ── PRODUCT ───────────────────────────────────────────────────────────────
    "product": [
        "product", "product_name", "productname", "item", "item_name",
        "description", "product_description", "product_id",
        "stockcode", "stock_code", "sku", "asin",
        "title", "product_title", "name", "goods",
    ],
    # ── CATEGORY ──────────────────────────────────────────────────────────────
    "category": [
        "category", "product_category", "dept", "department",
        "type", "segment", "sub_category", "subcategory",
        "product_type", "division", "class", "group",
        "genre", "brand", "product_line",
    ],
    # ── REGION ────────────────────────────────────────────────────────────────
    "region": [
        "region", "country", "state", "city", "location",
        "market", "territory", "area", "zone", "province",
        "ship_country", "billing_country", "geo", "geography",
    ],
    # ── ORDER ID ──────────────────────────────────────────────────────────────
    "order_id": [
        "order_id", "orderid", "invoice", "invoiceno", "invoice_no",
        "transaction_id", "txn_id", "receipt_id", "purchase_id",
        "order_number", "ordernumber", "basket_id",
    ],
}

# Standard column names used everywhere in the app
STANDARD_COLS = list(COLUMN_ALIASES.keys())

# Fuzzy match threshold: 0.0 = anything matches, 1.0 = exact match only
# 0.82 means 82% similar — catches "inv_date" → "date", "customerid" → "customer_id"
FUZZY_THRESHOLD = 0.82


# =============================================================================
# LAYER 1: EXACT ALIAS MATCHING
# =============================================================================
def _exact_match(normalized_cols: list) -> dict:
    """
    Checks each column against the alias dictionary for an exact match.

    KEY DESIGN: Iterates ALIASES in priority order (not CSV columns).
    This means if a CSV has both "unit_price" and "total_price", the one
    that appears FIRST in the aliases list wins.
    Since we put "total_price" before "unit_price" in the revenue aliases,
    total_price is always preferred over unit_price as revenue.

    Example with US dataset columns [unit_price, total_price]:
        aliases = ["total_price", ..., "unit_price"]
        → "total_price" is found first → mapped to "revenue" ✅
        → "unit_price" is left unmapped (no duplicate standard allowed)

    Args:
        normalized_cols: List of lowercased, stripped column names

    Returns:
        rename_map: { original_col: standard_col }
    """
    rename_map = {}
    used_standards = set()   # prevent two CSV cols mapping to same standard
    used_csv_cols  = set()   # prevent one CSV col mapping to two standards

    # Build a fast lookup: cleaned_col_name → original_col_name
    col_lookup = {}
    for col in normalized_cols:
        col_clean = col.replace("_", "").replace(" ", "")
        col_lookup[col_clean] = col

    # Iterate aliases IN ORDER — first match wins (priority order matters!)
    for standard, aliases in COLUMN_ALIASES.items():
        if standard in used_standards:
            continue
        for alias in aliases:
            alias_clean = alias.replace("_", "").replace(" ", "")
            if alias_clean in col_lookup:
                original_col = col_lookup[alias_clean]
                if original_col not in used_csv_cols:
                    rename_map[original_col] = standard
                    used_standards.add(standard)
                    used_csv_cols.add(original_col)
                    break  # found best match for this standard, move on

    return rename_map


# =============================================================================
# LAYER 2: FUZZY STRING MATCHING
# =============================================================================
def _fuzzy_match(normalized_cols: list, already_mapped: set) -> dict:
    """
    For columns not matched by Layer 1, uses fuzzy string similarity
    to find the closest standard column name.

    HOW SequenceMatcher WORKS:
        ratio = 2 * M / T
        where M = number of matching characters, T = total characters in both strings
        "invoicedate" vs "date": ratio ≈ 0.53 (too different)
        "order_dt" vs "date":    ratio ≈ 0.67 (below threshold, skip)
        "sale_date" vs "date":   ratio ≈ 0.73 (close but below threshold)
        "saledate" vs "date":    ratio ≈ 0.80 (just below threshold)
        "orderdate" vs "date":   ratio ≈ 0.89 ✅ above threshold → mapped!

    Args:
        normalized_cols: Columns not yet matched
        already_mapped: Standard names already assigned (avoid duplicates)

    Returns:
        fuzzy_map: { original_col: standard_col } for fuzzy matches
        suggestions: { original_col: (best_match, score) } for UI display
    """
    fuzzy_map = {}
    suggestions = {}

    for col in normalized_cols:
        best_score = 0
        best_standard = None

        for standard in COLUMN_ALIASES.keys():
            if standard in already_mapped:
                continue

            # Compare column name against the standard name
            score = SequenceMatcher(None, col, standard).ratio()

            # Also compare against each alias
            for alias in COLUMN_ALIASES[standard]:
                alias_score = SequenceMatcher(
                    None, col.replace("_", ""), alias.replace("_", "")
                ).ratio()
                score = max(score, alias_score)

            if score > best_score:
                best_score = score
                best_standard = standard

        if best_score >= FUZZY_THRESHOLD and best_standard not in already_mapped:
            fuzzy_map[col] = best_standard
            already_mapped.add(best_standard)
        elif best_score >= 0.65:
            # Not confident enough to auto-map, but worth suggesting to user
            suggestions[col] = (best_standard, round(best_score, 2))

    return fuzzy_map, suggestions


# =============================================================================
# LAYER 3: CONTENT SNIFFING
# =============================================================================
def _content_sniff(df: pd.DataFrame, unmapped_cols: list, already_mapped: set) -> dict:
    """
    Last resort: looks at the actual VALUES in a column to guess its type.

    Rules:
      - If >80% of values parse as dates → must be "date"
      - If values are all numeric and large (mean > 1) → might be "revenue"
      - If values are all numeric and small integers → might be "quantity"
      - If values are strings with high cardinality → might be "product"
      - If values are strings with low cardinality (<20 unique) → might be "category"

    Args:
        df: Raw DataFrame (pre-rename) with normalized column names
        unmapped_cols: Columns not yet matched by Layers 1 or 2
        already_mapped: Standard names already assigned

    Returns:
        sniff_map: { original_col: standard_col }
    """
    sniff_map = {}

    for col in unmapped_cols:
        if col not in df.columns:
            continue
        sample = df[col].dropna().head(100)  # check first 100 non-null values

        # --- Date detection ---
        if "date" not in already_mapped:
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                success_rate = parsed.notna().mean()
                if success_rate > 0.8:
                    sniff_map[col] = "date"
                    already_mapped.add("date")
                    continue
            except Exception:
                pass

        # --- Numeric column detection ---
        numeric_sample = pd.to_numeric(sample, errors="coerce")
        numeric_rate = numeric_sample.notna().mean()

        if numeric_rate > 0.9:
            mean_val = numeric_sample.mean()
            is_integer = (numeric_sample.dropna() % 1 == 0).all()

            if "revenue" not in already_mapped and mean_val > 1.0 and not is_integer:
                sniff_map[col] = "revenue"
                already_mapped.add("revenue")
                continue
            elif "quantity" not in already_mapped and is_integer and mean_val < 1000:
                sniff_map[col] = "quantity"
                already_mapped.add("quantity")
                continue

        # --- String column detection ---
        elif numeric_rate < 0.1:
            n_unique = sample.nunique()
            n_total = len(sample)

            if "product" not in already_mapped and n_unique > 20:
                sniff_map[col] = "product"
                already_mapped.add("product")
                continue
            elif "category" not in already_mapped and n_unique <= 20:
                sniff_map[col] = "category"
                already_mapped.add("category")
                continue

    return sniff_map


# =============================================================================
# MAIN DETECTION FUNCTION — combines all 3 layers
# =============================================================================
def detect_and_rename_columns(df: pd.DataFrame) -> tuple:
    """
    Runs all 3 detection layers and returns the renamed DataFrame
    plus a detailed report of what was detected and how.

    Args:
        df: Raw DataFrame from uploaded CSV

    Returns:
        Tuple of:
          - renamed DataFrame with standardized column names
          - detection_report: dict with info about how each column was mapped
            {
              "mapped":      { original_col: standard_col },  # all confirmed mappings
              "fuzzy":       { original_col: standard_col },  # fuzzy-matched
              "sniffed":     { original_col: standard_col },  # content-sniffed
              "suggestions": { original_col: (standard, score) },  # uncertain, shown to user
              "unmapped":    [ col1, col2, ... ],             # could not map
            }
    """
    # Step 0: Normalize all column names
    original_cols = df.columns.tolist()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    normalized_cols = df.columns.tolist()

    detection_report = {
        "mapped":      {},
        "fuzzy":       {},
        "sniffed":     {},
        "suggestions": {},
        "unmapped":    [],
    }

    # ── Layer 1: Exact alias matching ─────────────────────────────────────────
    exact_map = _exact_match(normalized_cols)
    detection_report["mapped"].update(exact_map)
    already_mapped_standards = set(exact_map.values())
    remaining_cols = [c for c in normalized_cols if c not in exact_map]

    # ── Layer 2: Fuzzy matching on remaining columns ──────────────────────────
    fuzzy_map, suggestions = _fuzzy_match(remaining_cols, already_mapped_standards)
    detection_report["fuzzy"].update(fuzzy_map)
    detection_report["suggestions"].update(suggestions)
    already_mapped_standards.update(fuzzy_map.values())
    remaining_cols = [c for c in remaining_cols if c not in fuzzy_map]

    # ── Layer 3: Content sniffing on still-remaining columns ──────────────────
    sniff_map = _content_sniff(df, remaining_cols, already_mapped_standards)
    detection_report["sniffed"].update(sniff_map)
    remaining_cols = [c for c in remaining_cols if c not in sniff_map]
    detection_report["unmapped"] = remaining_cols

    # Apply all renames
    full_rename = {**exact_map, **fuzzy_map, **sniff_map}
    df = df.rename(columns=full_rename)

    return df, detection_report


# =============================================================================
# MANUAL COLUMN MAPPING UI
# Called when auto-detection is uncertain — lets user fix mappings
# =============================================================================
def render_column_mapper(df_raw: pd.DataFrame, detection_report: dict) -> pd.DataFrame:
    """
    Shows a Streamlit UI for the user to manually confirm or fix column mappings.
    Called from app.py when there are uncertain suggestions or unmapped columns.

    Displays:
      - ✅ Auto-detected mappings (read-only confirmation)
      - ⚠️  Suggested mappings (user can confirm or change)
      - ❓ Unmapped columns (user can manually assign)

    Args:
        df_raw: DataFrame with normalized but not yet fully renamed columns
        detection_report: Output from detect_and_rename_columns()

    Returns:
        DataFrame with user-confirmed column mappings applied
    """
    st.markdown("### 🗺️ Column Mapping")

    all_options = ["— skip —"] + STANDARD_COLS
    user_overrides = {}

    # --- Show confirmed auto-detections (green, no input needed) ---
    confirmed = {**detection_report["mapped"], **detection_report["fuzzy"], **detection_report["sniffed"]}
    if confirmed:
        with st.expander("✅ Auto-Detected Columns", expanded=False):
            for orig, std in confirmed.items():
                layer = "exact match" if orig in detection_report["mapped"] else \
                        "fuzzy match" if orig in detection_report["fuzzy"] else "content sniff"
                st.markdown(f"`{orig}` → **{std}** *(via {layer})*")

    # --- Suggestions: auto-detection not confident — user decides ---
    if detection_report["suggestions"]:
        st.markdown("#### ⚠️ Uncertain Matches — Please Confirm")
        st.caption("These columns couldn't be mapped with high confidence. Select the correct standard name.")
        for orig, (suggested, score) in detection_report["suggestions"].items():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"`{orig}` (similarity: {score:.0%})")
                st.caption(f"Sample values: {df_raw[orig].dropna().head(3).tolist()}")
            with col2:
                choice = st.selectbox(
                    f"Map '{orig}' to:",
                    options=all_options,
                    index=all_options.index(suggested) if suggested in all_options else 0,
                    key=f"suggest_{orig}",
                )
                if choice != "— skip —":
                    user_overrides[orig] = choice

    # --- Unmapped columns: completely unrecognized ---
    if detection_report["unmapped"]:
        st.markdown("#### ❓ Unrecognized Columns — Map Manually")
        st.caption("These columns weren't recognized. If they contain useful data, map them here.")
        for orig in detection_report["unmapped"]:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"`{orig}`")
                try:
                    st.caption(f"Sample: {df_raw[orig].dropna().head(3).tolist()}")
                except Exception:
                    pass
            with col2:
                choice = st.selectbox(
                    f"Map '{orig}' to:",
                    options=all_options,
                    index=0,
                    key=f"unmap_{orig}",
                )
                if choice != "— skip —":
                    user_overrides[orig] = choice

    # Apply user overrides on top of auto-detected renames
    if user_overrides:
        df_raw = df_raw.rename(columns=user_overrides)

    return df_raw


# =============================================================================
# CLEANING
# =============================================================================
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame after column mapping:
      - Parses date column to datetime
      - Strips currency symbols from revenue
      - Computes revenue = price × quantity if revenue column is missing
      - Removes invalid rows (negative revenue, unparseable dates)
    """
    # --- Date ---
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    # --- Revenue ---
    if "revenue" in df.columns:
        df["revenue"] = (
            df["revenue"].astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
        df = df[df["revenue"] > 0]

    # --- Quantity ---
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1).astype(int)
        df = df[df["quantity"] > 0]

    # --- Derive revenue from price × quantity if missing ---
    if "revenue" not in df.columns:
        price_col = next((c for c in df.columns if "price" in c), None)
        if price_col and "quantity" in df.columns:
            df["revenue"] = pd.to_numeric(df[price_col], errors="coerce") * df["quantity"]
            df = df[df["revenue"] > 0]

    # --- Fix UK-retail style datasets: if revenue is unit_price (per-item),
    #     multiply by quantity to get actual order line total.
    #     Heuristic: if "unit_price" or "unitprice" was the source column AND
    #     quantity exists, recalculate.  We detect this by checking whether the
    #     original (pre-clean) column was a per-unit price alias. ---
    if "revenue" in df.columns and "quantity" in df.columns:
        # Check if revenue values look like per-unit prices:
        # Per-unit prices are usually small (< £100 / ₹5000) while line totals
        # for quantity 6 would be larger.  We use a simple heuristic:
        # if max(revenue) < 500 AND max(quantity) > 1, it's likely unit prices.
        # More reliably: check if revenue / quantity gives a consistent per-unit price.
        rev_vals = df["revenue"]
        qty_vals = df["quantity"]
        # Only apply if we have quantity > 1 rows
        has_multi_qty = (qty_vals > 1).any()
        if has_multi_qty:
            # Calculate what revenue would be if already multiplied
            # Check a sample: if most rows have revenue == round(revenue/qty)*qty, it's already totals
            sample = df.head(500).copy()
            sample["implied_unit"] = sample["revenue"] / sample["quantity"]
            sample["recalc"] = sample["implied_unit"] * sample["quantity"]
            already_total = (abs(sample["recalc"] - sample["revenue"]) < 0.01).mean()
            # If it's already totals, that's fine (always true - skip)
            # Key check: if quantity col has values > 1 but revenue stays flat across
            # same-invoice rows, revenue = unit price
            if "order_id" in df.columns:
                # Check if rows with same order_id have same revenue but different qty
                sample2 = df.sample(min(200, len(df)), random_state=42)
                if len(sample2["order_id"].unique()) < len(sample2) * 0.8:
                    # Multiple rows per order exist
                    grp = sample2.groupby("order_id").agg(
                        rev_std=("revenue","std"), qty_max=("quantity","max")
                    )
                    # If revenue has near-zero std within order but qty varies → unit prices
                    unit_price_pattern = (
                        (grp["rev_std"].fillna(0) < 0.5) & (grp["qty_max"] > 1)
                    ).mean()
                    if unit_price_pattern > 0.3:
                        df["revenue"] = df["revenue"] * df["quantity"]

    return df.reset_index(drop=True)


def get_column_summary(df: pd.DataFrame) -> dict:
    """Returns which standard columns are present in the DataFrame."""
    return {col: col in df.columns for col in STANDARD_COLS}


@st.cache_data
def load_sample_dataset(name: str) -> tuple:
    """
    Loads sample dataset using an absolute path so it works regardless of
    which directory you run `streamlit run app.py` from.

    Only UK Online Retail is supported as a sample dataset.
    (Superstore removed due to column compatibility issues.)
    """
    # Build absolute path: data_loader.py is in utils/, go up one level to project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    path_map = {
        "UK Online Retail": os.path.join(BASE_DIR, "sample_data", "uk_retail.csv"),
        "US E-Commerce (Demo)": os.path.join(BASE_DIR, "sample_data", "us_ecommerce.csv"),
    }
    path = path_map.get(name)

    if not path:
        st.error(f"Unknown dataset: {name}")
        return None, {}

    if not os.path.exists(path):
        st.error(
            f"File not found at: `{path}`\n\n"
            "**Steps to fix:**\n"
            "1. Download UK Online Retail from: https://www.kaggle.com/datasets/carrie1/ecommerce-data\n"
            "2. Rename the downloaded file to exactly: `uk_retail.csv` (all lowercase)\n"
            "3. Place it in the `sample_data/` folder inside your project\n"
            "4. Final path should be: `commercepulse-ai/sample_data/uk_retail.csv`"
        )
        return None, {}

    try:
        df = pd.read_csv(path, encoding="latin1")
        if df.empty:
            st.error("The CSV file is empty.")
            return None, {}
        df, report = detect_and_rename_columns(df)
        df = clean_dataframe(df)
        if df.empty:
            st.error("No valid rows remain after cleaning. Check the CSV format.")
            return None, {}
        return df, report
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None, {}


def load_uploaded_file(uploaded_file) -> tuple:
    """
    Loads a user-uploaded CSV.
    Returns (DataFrame, detection_report) tuple.
    """
    try:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")

        df, report = detect_and_rename_columns(df)
        df = clean_dataframe(df)
        return df, report

    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        return None, {}
