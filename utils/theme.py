# =============================================================================
# utils/theme.py
# =============================================================================
# PURPOSE:
#   Central place for all UI theming, shared formatters, and CSS injection.
#   Import this at the top of every page to get consistent styling.
#
# USAGE:
#   from utils.theme import apply_theme, smart_fmt, card_metric
# =============================================================================

import streamlit as st
from utils.currency import fmt, get_currency_symbol


# =============================================================================
# GLOBAL CSS — injected once per page via apply_theme()
# =============================================================================
_CSS = """
<style>
/* ── Import Google Font ───────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Gradient page header ─────────────────────────────────────────────────── */
.cp-header {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.cp-subheader {
    font-size: 1.5rem;
    font-weight: 700;
    color: #94a3b8;
    margin-bottom: 1rem;
}
.cp-caption {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 1rem;
}

/* ── KPI cards ────────────────────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #6366f1; }
.kpi-label  { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem; }
.kpi-value  { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }
.kpi-delta  { font-size: 0.8rem; margin-top: 0.2rem; }
.kpi-delta.pos { color: #22c55e; }
.kpi-delta.neg { color: #ef4444; }

/* ── Section headers ──────────────────────────────────────────────────────── */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    padding: 0.4rem 0;
    border-bottom: 2px solid #6366f1;
    margin-bottom: 1rem;
}

/* ── Insight cards ────────────────────────────────────────────────────────── */
.insight-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-left: 4px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    color: #cbd5e1;
    font-size: 0.9rem;
}
.insight-card.warning { border-left-color: #f59e0b; }
.insight-card.success { border-left-color: #22c55e; }
.insight-card.danger  { border-left-color: #ef4444; }

/* ── Pill badges ──────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.badge-green  { background: #052e16; color: #22c55e; }
.badge-red    { background: #450a0a; color: #ef4444; }
.badge-blue   { background: #0c1a47; color: #6366f1; }
.badge-yellow { background: #431407; color: #f59e0b; }

/* ── Sidebar branding ─────────────────────────────────────────────────────── */
.sidebar-brand {
    text-align: center;
    padding: 1rem 0;
    border-bottom: 1px solid #334155;
    margin-bottom: 1rem;
}
.sidebar-logo {
    font-size: 2rem;
    margin-bottom: 0.2rem;
}
.sidebar-title {
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar-subtitle {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* ── Streamlit metric overrides for dark theme ────────────────────────────── */
[data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="metric-container"]:hover {
    border-color: #6366f1;
    transition: border-color 0.2s;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; }

/* ── DataFrame styling ────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* ── Hide Streamlit branding ──────────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Scrollbar styling ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }
</style>
"""

# Plotly chart defaults — consistent dark theme across all charts
PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter"),
    title_font=dict(color="#e2e8f0", size=14),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickfont=dict(color="#64748b")),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155", font=dict(color="#94a3b8")),
    margin=dict(l=10, r=10, t=40, b=10),
)

PLOTLY_COLORS = [
    "#6366f1", "#06b6d4", "#22c55e", "#f59e0b",
    "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6",
]


def apply_theme():
    """Call at the top of every page to inject global CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)
    # Sidebar branding
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-logo">🛒</div>
            <div class="sidebar-title">CommercePulse AI</div>
            <div class="sidebar-subtitle">E-Commerce Intelligence Platform</div>
        </div>
        """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = ""):
    """Renders a gradient page title with optional subtitle."""
    display = f"{icon} {title}" if icon else title
    st.markdown(f'<div class="cp-header">{display}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="cp-caption">{subtitle}</div>', unsafe_allow_html=True)


def section_header(title: str):
    """Renders a styled section divider."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def insight_card(text: str, kind: str = "info"):
    """
    Renders a styled insight card.
    kind: 'info' | 'warning' | 'success' | 'danger'
    """
    cls_map = {"info": "", "warning": " warning", "success": " success", "danger": " danger"}
    cls = cls_map.get(kind, "")
    st.markdown(f'<div class="insight-card{cls}">{text}</div>', unsafe_allow_html=True)


def smart_fmt(value: float) -> str:
    """
    Shortens large currency values for metric cards.
      >= 1,000,000  → ₹2.48M
      >= 1,000      → ₹24.8K
      < 1,000       → ₹248.50
    """
    sym = get_currency_symbol()
    if value >= 1_000_000:
        return f"{sym}{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{sym}{value / 1_000:.1f}K"
    else:
        return fmt(value, 2)


def plotly_defaults(fig, height: int = 400, hovermode: str = "x unified"):
    """Apply consistent dark-theme styling to any Plotly figure."""
    fig.update_layout(height=height, hovermode=hovermode, **PLOTLY_LAYOUT)
    return fig
