# =============================================================================
# utils/currency.py
# =============================================================================
# PURPOSE:
#   Automatically detects the currency of any uploaded dataset and provides
#   a formatter so every page displays the correct symbol (£, $, €, ₹ etc.)
#
# HOW CURRENCY DETECTION WORKS (3 methods, tried in order):
#
#   Method 1 — Scan raw CSV text for currency symbols
#     Reads the first 50 rows as raw text and counts occurrences of
#     £, $, €, ₹, ¥ etc. The most frequent one wins.
#     Example: UnitPrice column has "£2.50", "£14.99" → detected as GBP (£)
#
#   Method 2 — Match by known dataset / country column
#     If a 'region' column exists and most values are "United Kingdom" → GBP
#     If most values are "United States" → USD, "France/Germany" → EUR etc.
#
#   Method 3 — Fallback to user selection
#     If neither method works, show a dropdown in the UI so the user picks.
#
# USAGE IN OTHER FILES:
#   from utils.currency import fmt, get_currency_symbol
#
#   fmt(1234.5)        → "£1,234.50"  (or "$1,234.50" etc.)
#   fmt(1234.5, 0)     → "£1,235"     (no decimal places)
#   get_currency_symbol() → "£"
# =============================================================================

import streamlit as st
import pandas as pd
import re

# =============================================================================
# CURRENCY REGISTRY
# Maps currency code → (symbol, name, countries/regions that use it)
# =============================================================================
CURRENCIES = {
    "GBP": {
        "symbol": "£",
        "name": "British Pound",
        "regions": ["united kingdom", "uk", "england", "scotland", "wales", "great britain"],
    },
    "USD": {
        "symbol": "$",
        "name": "US Dollar",
        "regions": ["united states", "usa", "us", "america"],
    },
    "EUR": {
        "symbol": "€",
        "name": "Euro",
        "regions": ["france", "germany", "italy", "spain", "netherlands", "belgium",
                    "portugal", "austria", "finland", "ireland", "greece", "europe"],
    },
    "INR": {
        "symbol": "₹",
        "name": "Indian Rupee",
        "regions": ["india"],
    },
    "JPY": {
        "symbol": "¥",
        "name": "Japanese Yen",
        "regions": ["japan"],
    },
    "AUD": {
        "symbol": "A$",
        "name": "Australian Dollar",
        "regions": ["australia"],
    },
    "CAD": {
        "symbol": "C$",
        "name": "Canadian Dollar",
        "regions": ["canada"],
    },
    "BRL": {
        "symbol": "R$",
        "name": "Brazilian Real",
        "regions": ["brazil", "brasil"],
    },
}

# Symbol → currency code lookup (used in raw text scanning)
SYMBOL_TO_CODE = {
    "£":  "GBP",
    "$":  "USD",
    "€":  "EUR",
    "₹":  "INR",
    "¥":  "JPY",
    "A$": "AUD",
    "C$": "CAD",
    "R$": "BRL",
}

DEFAULT_CURRENCY = "USD"


# =============================================================================
# METHOD 1: Scan raw CSV text for currency symbols
# =============================================================================
def detect_from_raw_text(uploaded_file) -> str | None:
    """
    Reads the raw CSV text (first 5000 chars) and counts currency symbols.
    The most frequent symbol wins.

    Args:
        uploaded_file: Streamlit UploadedFile (before pd.read_csv)

    Returns:
        Currency code like "GBP" or None if nothing found
    """
    try:
        uploaded_file.seek(0)
        raw = uploaded_file.read(5000).decode("latin1", errors="ignore")
        uploaded_file.seek(0)  # reset for later pd.read_csv

        counts = {}
        for symbol, code in SYMBOL_TO_CODE.items():
            count = raw.count(symbol)
            if count > 0:
                counts[code] = count

        if counts:
            return max(counts, key=counts.get)
    except Exception:
        pass
    return None


# =============================================================================
# METHOD 2: Match by region/country column values
# =============================================================================
def detect_from_region_column(df: pd.DataFrame) -> str | None:
    """
    Looks at the 'region' column (if present) and matches country names
    to known currency regions.

    HOW IT WORKS:
      1. Get the most common value in the region column
      2. Check if it matches any country in our CURRENCIES registry
      3. Return that currency code

    Args:
        df: Cleaned DataFrame (after column renaming)

    Returns:
        Currency code or None
    """
    if "region" not in df.columns:
        return None

    # Get top 3 most common regions
    top_regions = df["region"].dropna().str.lower().value_counts().head(3).index.tolist()

    for region in top_regions:
        for code, info in CURRENCIES.items():
            if any(r in region or region in r for r in info["regions"]):
                return code

    return None


# =============================================================================
# METHOD 3: Detect from dataset/source name
# =============================================================================
def detect_from_source_name(source_name: str) -> str | None:
    """
    Checks the filename or dataset name for clues.

    Example:
        "uk_retail.csv"   → GBP
        "superstore.csv"  → USD (US-based dataset)
        "olist_brazil"    → BRL
    """
    if not source_name:
        return None

    name = source_name.lower()

    source_hints = {
        "GBP": ["uk", "united_kingdom", "britain", "retail"],  # uk_retail is GBP
        "USD": ["superstore", "us_", "_us", "amazon", "walmart"],
        "EUR": ["eu_", "europe", "french", "german"],
        "INR": ["india", "flipkart", "amazon_in"],
        "BRL": ["brazil", "olist", "brasil"],
    }

    for code, hints in source_hints.items():
        if any(hint in name for hint in hints):
            return code

    return None


# =============================================================================
# MAIN DETECTION FUNCTION
# =============================================================================
def detect_currency(df: pd.DataFrame, source_name: str = "", uploaded_file=None) -> str:
    """
    Runs all detection methods in priority order and returns the currency code.

    Priority:
      1. Raw text scan (most reliable — actual symbols in data)
      2. Region column match
      3. Source/filename hint
      4. Default to USD

    Args:
        df: Cleaned DataFrame
        source_name: Filename or dataset name string
        uploaded_file: Original Streamlit UploadedFile (for raw text scan)

    Returns:
        Currency code string like "GBP", "USD", "EUR"
    """
    # Method 1: Raw text scan
    if uploaded_file is not None:
        code = detect_from_raw_text(uploaded_file)
        if code:
            return code

    # Method 2: Region column
    code = detect_from_region_column(df)
    if code:
        return code

    # Method 3: Source name hint
    code = detect_from_source_name(source_name)
    if code:
        return code

    return DEFAULT_CURRENCY


# =============================================================================
# CURRENCY FORMATTER
# The main function used across all pages
# =============================================================================
def fmt(value: float, decimals: int = 0) -> str:
    """
    Formats a number as currency using the currently detected currency.

    Reads currency from st.session_state["currency_code"] which is set
    on the home page after upload.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places (default 0 for whole numbers)

    Returns:
        Formatted string like "£1,234", "$2,500.50", "€890"

    Examples:
        fmt(1234)        → "£1,234"
        fmt(1234.5, 2)   → "£1,234.50"
        fmt(0)           → "£0"
    """
    symbol = get_currency_symbol()
    try:
        return f"{symbol}{value:,.{decimals}f}"
    except (ValueError, TypeError):
        return f"{symbol}0"


def get_currency_symbol() -> str:
    """Returns the symbol for the currently detected currency (e.g. '£')."""
    code = st.session_state.get("currency_code", DEFAULT_CURRENCY)
    return CURRENCIES.get(code, CURRENCIES[DEFAULT_CURRENCY])["symbol"]


def get_currency_name() -> str:
    """Returns the full name (e.g. 'British Pound')."""
    code = st.session_state.get("currency_code", DEFAULT_CURRENCY)
    return CURRENCIES.get(code, CURRENCIES[DEFAULT_CURRENCY])["name"]


def get_currency_code() -> str:
    """Returns the currency code (e.g. 'GBP')."""
    return st.session_state.get("currency_code", DEFAULT_CURRENCY)


# =============================================================================
# UI WIDGET — shown on home page so user can override if detection is wrong
# =============================================================================
def render_currency_selector(detected_code: str) -> str:
    """
    Shows a small currency selector on the home page.
    Pre-selects the auto-detected currency but lets user override.

    Args:
        detected_code: Auto-detected currency code

    Returns:
        Final selected currency code (user's choice)
    """
    options = {f"{info['symbol']} {info['name']} ({code})": code
               for code, info in CURRENCIES.items()}

    # Find default index based on detected code
    detected_label = next(
        (label for label, code in options.items() if code == detected_code),
        list(options.keys())[0]
    )

    st.markdown("#### 💱 Currency")
    st.caption("Auto-detected from your data — change if incorrect")

    selected_label = st.selectbox(
        "Currency",
        options=list(options.keys()),
        index=list(options.keys()).index(detected_label),
        label_visibility="collapsed",
    )

    return options[selected_label]
