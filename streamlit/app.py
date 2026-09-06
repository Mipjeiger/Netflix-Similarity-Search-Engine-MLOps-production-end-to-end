"""
Netflix Engineering — Streamlit Platform UI
Company Standard: Netflix MLOps Design System
- Theme: Netflix Dark ( #141414 / #E50914 )
- Layout: Wide, responsive, card-based
- Typography: Inter / Helvetica Neue
- Components: Metric cards, Plotly dark, API-aware recommendations
"""
import os
import pickle
from pathlib import Path

import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Netflix Engineering Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")

# Resolve data path — works both inside Docker and locally
CANDIDATE_DATA_PATHS = [
    Path("data/processed/netflix_processed.csv"),
    Path("/app/data/processed/netflix_processed.csv"),
    Path("data/raw/netflix_titles.csv"),
    Path("/app/data/raw/netflix_titles.csv"),
    Path("airflow/data/raw/netflix_titles.csv"),
    Path("/opt/airflow/data/raw/netflix_titles.csv"),
    Path("../airflow/data/raw/netflix_titles.csv"),
]

COLORS = {
    "netflix_red": "#E50914",
    "netflix_red_hover": "#B81D24",
    "bg": "#141414",
    "bg_sidebar": "#0F0F0F",
    "surface": "#1E1E1E",
    "surface_2": "#2A2A2A",
    "border": "#333333",
    "text": "#FFFFFF",
    "muted": "#B3B3B3",
    "muted_2": "#808080",
    "success": "#46D369",
    "warning": "#FFC107",
}

PLOTLY_TEMPLATE = "plotly_dark"

# ---------------------------------------------------------------------------
# Global CSS — Company Design System
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Bebas+Neue&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}

    /* App background */
    .stApp {{
        background: {COLORS['bg']};
        color: {COLORS['text']};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {COLORS['bg_sidebar']};
        border-right: 1px solid {COLORS['border']};
    }}
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {{
        color: {COLORS['text']} !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: {COLORS['border']};
    }}

    /* Brand header in sidebar */
    .brand {{
        display: flex; align-items: center; gap: 12px;
        padding: 6px 0 14px 0; margin-bottom: 6px;
        border-bottom: 1px solid {COLORS['border']};
    }}
    .brand-mark {{
        width: 36px; height: 36px; border-radius: 8px;
        background: {COLORS['netflix_red']};
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 16px; color: white; letter-spacing: 0.5px;
        box-shadow: 0 4px 16px rgba(229,9,20,0.35);
    }}
    .brand-title {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 22px; letter-spacing: 1.2px; color: {COLORS['netflix_red']};
        line-height: 1; margin: 0;
    }}
    .brand-sub {{
        font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
        color: {COLORS['muted']}; margin-top: -2px; font-weight: 600;
    }}

    /* Top header bar */
    .page-header {{
        padding: 10px 0 4px 0;
        border-bottom: 1px solid {COLORS['border']};
        margin-bottom: 20px;
    }}
    .page-kicker {{
        font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase;
        color: {COLORS['netflix_red']}; font-weight: 700; margin-bottom: 4px;
    }}
    .page-title {{
        font-size: 32px; font-weight: 800; letter-spacing: -0.02em;
        color: {COLORS['text']}; margin: 0 0 4px 0; line-height: 1.1;
    }}
    .page-subtitle {{
        font-size: 13.5px; color: {COLORS['muted']}; margin: 0;
    }}

    /* Metric cards */
    .metric-card {{
        background: linear-gradient(180deg, {COLORS['surface']} 0%, #181818 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 16px 18px;
        position: relative;
        overflow: hidden;
    }}
    .metric-card::before {{
        content: "";
        position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: {COLORS['netflix_red']};
        opacity: 0.9;
    }}
    .metric-label {{
        font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase;
        color: {COLORS['muted']}; font-weight: 600; margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 28px; font-weight: 800; color: {COLORS['text']};
        letter-spacing: -0.02em; line-height: 1;
    }}
    .metric-delta {{
        font-size: 12px; color: {COLORS['muted_2']}; margin-top: 6px;
    }}
    .metric-delta strong {{ color: {COLORS['success']}; }}

    /* Section cards */
    .section-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 18px 18px 14px 18px;
    }}
    .section-card h4 {{
        font-size: 14px; font-weight: 700; color: {COLORS['text']};
        margin: 0 0 2px 0; letter-spacing: -0.01em;
    }}
    .section-card p.desc {{
        font-size: 12.5px; color: {COLORS['muted']}; margin: 0 0 12px 0;
    }}

    /* Recommendation cards */
    .rec-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 14px 16px;
        display: flex; justify-content: space-between; align-items: flex-start;
        gap: 12px;
        transition: border-color 0.15s;
    }}
    .rec-card:hover {{ border-color: #4A4A4A; }}
    .rec-title {{ font-size: 14px; font-weight: 700; color: {COLORS['text']}; margin: 0; }}
    .rec-meta {{ font-size: 12px; color: {COLORS['muted']}; margin-top: 2px; }}
    .rec-badge {{
        background: rgba(229,9,20,0.14);
        color: {COLORS['netflix_red']};
        border: 1px solid rgba(229,9,20,0.3);
        font-size: 11px; font-weight: 700; letter-spacing: 0.04em;
        padding: 4px 8px; border-radius: 999px; white-space: nowrap;
    }}

    /* Buttons */
    .stButton > button {{
        background: {COLORS['netflix_red']} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        padding: 0.55rem 1.1rem !important;
        box-shadow: 0 4px 14px rgba(229,9,20,0.35) !important;
    }}
    .stButton > button:hover {{
        background: {COLORS['netflix_red_hover']} !important;
        box-shadow: 0 6px 18px rgba(229,9,20,0.45) !important;
    }}
    .stButton > button:disabled {{
        opacity: 0.55;
    }}

    /* Inputs */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {{
        background: {COLORS['surface_2']} !important;
        border: 1px solid {COLORS['border']} !important;
        color: {COLORS['text']} !important;
        border-radius: 8px !important;
    }}
    .stSlider [data-testid="stThumbValue"] {{ color: {COLORS['text']}; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px; padding: 6px 14px; font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS['surface_2']} !important;
        color: {COLORS['text']} !important;
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{
        border: 1px solid {COLORS['border']};
        border-radius: 12px; overflow: hidden;
    }}

    /* Alerts */
    [data-testid="stAlert"] {{
        border-radius: 10px;
    }}

    /* Footer */
    .footer {{
        margin-top: 28px; padding: 16px 0 8px 0;
        border-top: 1px solid {COLORS['border']};
        display: flex; justify-content: space-between; align-items: center;
        font-size: 12px; color: {COLORS['muted_2']};
    }}
    .footer a {{ color: {COLORS['muted']}; text-decoration: none; }}
    .footer a:hover {{ color: {COLORS['text']}; }}

    /* Hide Streamlit decoration */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{ padding-top: 1.2rem; }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame | None:
    for p in CANDIDATE_DATA_PATHS:
        try:
            if p.exists():
                df = pd.read_csv(p)
                # Normalize columns if needed
                if "show_id" in df.columns:
                    return df
                return df
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False, ttl=300)
def api_health() -> dict:
    for endpoint in ["/health", "/healthcheck", "/", "/docs"]:
        try:
            r = requests.get(f"{API_URL}{endpoint}", timeout=3)
            if r.status_code < 500:
                return {"ok": r.status_code == 200, "status": r.status_code, "url": API_URL, "endpoint": endpoint}
        except Exception:
            continue
    # try without path
    try:
        r = requests.get(API_URL, timeout=3)
        return {"ok": r.status_code == 200, "status": r.status_code, "url": API_URL, "endpoint": "/"}
    except Exception as e:
        return {"ok": False, "status": None, "url": API_URL, "error": str(e)}


def call_recommend_api(title: str, n: int):
    """Try multiple endpoint conventions used in this repo."""
    payloads = [
        (f"{API_URL}/recommendations", {"title": title, "n_recommendations": n, "n_recommendation": n}),
        (f"{API_URL}/recommend", {"title": title, "n_recommendations": n}),
        (f"{API_URL}/api/recommend", {"title": title, "n_recommendations": n}),
    ]
    last_err = None
    for url, payload in payloads:
        try:
            r = requests.post(url, json=payload, timeout=8)
            if r.status_code == 200:
                return r.json(), None, url
            elif r.status_code == 404:
                last_err = r.text
                continue
            else:
                last_err = f"{r.status_code}: {r.text[:300]}"
        except Exception as e:
            last_err = str(e)
    return None, last_err, None


def metric_card(label: str, value: str, delta: str = ""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta">{delta}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(kicker: str, title: str, subtitle: str):
    st.markdown(f"""
    <div class="page-header">
        <div class="page-kicker">{kicker}</div>
        <div class="page-title">{title}</div>
        <p class="page-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def plot_style(fig):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
inject_css()

# Sidebar — brand + navigation + system status
with st.sidebar:
    st.markdown(f"""
    <div class="brand">
        <div class="brand-mark">N</div>
        <div>
            <div class="brand-title">NETFLIX</div>
            <div class="brand-sub">Engineering Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:11px; letter-spacing:0.12em; text-transform:uppercase; color:{COLORS['muted']}; font-weight:700; margin: 4px 0 8px 0;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Dashboard", "Recommendations", "Network Analysis", "Data Explorer"],
        label_visibility="collapsed",
    )

    st.markdown(f"<hr style='margin:14px 0; border:none; border-top:1px solid {COLORS['border']};'/>", unsafe_allow_html=True)

    # System status
    health = api_health()
    df_preview = load_data()
    st.markdown(f"<div style='font-size:11px; letter-spacing:0.12em; text-transform:uppercase; color:{COLORS['muted']}; font-weight:700; margin-bottom:8px;'>System</div>", unsafe_allow_html=True)
    api_dot = COLORS["success"] if health.get("ok") else COLORS["warning"]
    api_label = "API Online" if health.get("ok") else "API Offline"
    data_label = f"{len(df_preview):,} titles" if df_preview is not None else "No dataset found"

    st.markdown(f"""
    <div style="display:flex; flex-direction:column; gap:8px; font-size:12.5px;">
        <div style="display:flex; align-items:center; gap:8px; color:{COLORS['muted']}">
            <span style="width:8px; height:8px; border-radius:999px; background:{api_dot}; display:inline-block; box-shadow:0 0 8px {api_dot};"></span>
            <span style="color:{COLORS['text']}; font-weight:600;">{api_label}</span>
            <span style="color:{COLORS['muted_2']}; font-size:11px;">· {API_URL}</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; color:{COLORS['muted']}">
            <span style="width:8px; height:8px; border-radius:999px; background:{COLORS['success'] if df_preview is not None else COLORS['warning']}; display:inline-block;"></span>
            <span style="color:{COLORS['text']}; font-weight:600;">Dataset</span>
            <span style="color:{COLORS['muted_2']}; font-size:11px;">· {data_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='margin-top:16px; padding:10px 12px; background:{COLORS['surface']}; border:1px solid {COLORS['border']}; border-radius:10px; font-size:11.5px; color:{COLORS['muted']}; line-height:1.4;'>"
                f"<strong style='color:{COLORS['text']};'>MLOps Stack</strong><br/>"
                f"FastAPI · MLflow · Airflow · Postgres · Redis<br/>"
                f"<span style='color:{COLORS['muted_2']};'>TF-IDF + Cosine Similarity · 5k features</span>"
                f"</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="footer" style="margin-top:18px; border-top:none; padding-top:0;">
        <span>© 2026 Netflix Engineering</span>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
if page == "Dashboard":
    section_header("Platform Overview", "Content Intelligence Dashboard", "Catalog health, trends, and distribution across 8,800+ titles.")

    df = load_data()
    if df is None:
        st.error("Dataset not found. Expected at `data/processed/netflix_processed.csv` or `airflow/data/raw/netflix_titles.csv`. Run the Airflow preprocessing DAG or check the mounted volume.")
        st.stop()

    # Normalize expected columns
    # Ensure country / rating / release_year exist
    for col, default in [("country", "Unknown"), ("rating", "Unknown"), ("release_year", 0), ("type", "Unknown"), ("listed_in", "Unknown")]:
        if col not in df.columns:
            df[col] = default

    # Metrics
    total = len(df)
    movies = len(df[df["type"] == "Movie"])
    tv = len(df[df["type"] == "TV Show"])
    # Most common rating safely
    try:
        top_rating = df["rating"].mode().iloc[0] if not df["rating"].mode().empty else "—"
    except Exception:
        top_rating = "—"
    pct_movies = f"{movies/total*100:.1f}%" if total else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Titles", f"{total:,}", f"<strong>{pct_movies}</strong> movies · {100 - float(pct_movies.strip('%')):.1f}% TV" if total else "")
    with c2: metric_card("Movies", f"{movies:,}", f"<strong>{movies/total*100:.0f}%</strong> of catalog" if total else "")
    with c3: metric_card("TV Shows", f"{tv:,}", f"<strong>{tv/total*100:.0f}%</strong> of catalog" if total else "")
    with c4: metric_card("Top Rating", str(top_rating), f"<strong>{(df['rating']==top_rating).sum():,}</strong> titles" if top_rating != "—" else "")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Two-column charts
    left, right = st.columns([1.55, 1])

    with left:
        st.markdown(f"""
        <div class="section-card">
            <h4>Catalog Growth by Release Year</h4>
            <p class="desc">Annual additions split by Movies vs TV Shows — highlights content strategy shifts.</p>
        </div>
        """, unsafe_allow_html=True)
        try:
            yearly = df.groupby(["release_year", "type"]).size().reset_index(name="count")
            yearly = yearly[yearly["release_year"] > 1940].sort_values("release_year")
            fig = px.line(yearly, x="release_year", y="count", color="type",
                          color_discrete_map={"Movie": COLORS["netflix_red"], "TV Show": "#FFFFFF"},
                          markers=False)
            fig.update_traces(line=dict(width=2.5))
            fig = plot_style(fig)
            fig.update_layout(height=320, xaxis_title="Release Year", yaxis_title="Titles", legend_title="")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Could not render trend chart: {e}")

    with right:
        st.markdown(f"""
        <div class="section-card">
            <h4>Top Production Countries</h4>
            <p class="desc">Where Netflix content originates — top 10 by title count.</p>
        </div>
        """, unsafe_allow_html=True)
        try:
            # Country may contain comma-separated values — take primary
            country_series = df["country"].fillna("Unknown").astype(str).str.split(",").str[0].str.strip()
            top = country_series.value_counts().head(10).sort_values(ascending=True)
            fig2 = px.bar(x=top.values, y=top.index, orientation="h",
                          color=top.values, color_continuous_scale=[[0, "#3A3A3A"], [1, COLORS["netflix_red"]]],
                          )
            fig2 = plot_style(fig2)
            fig2.update_layout(height=320, xaxis_title="Titles", yaxis_title="", coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
            fig2.update_traces(marker_line_width=0, hovertemplate="%{y}: %{x}<extra></extra>")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Could not render country chart: {e}")

    # Second row
    cA, cB = st.columns(2)
    with cA:
        st.markdown("""
        <div class="section-card">
            <h4>Content Mix</h4>
            <p class="desc">Movies vs TV Shows share.</p>
        </div>
        """, unsafe_allow_html=True)
        try:
            mix = df["type"].value_counts()
            fig3 = go.Figure(data=[go.Pie(
                labels=mix.index, values=mix.values, hole=0.58,
                marker=dict(colors=[COLORS["netflix_red"], "#2A2A2A"], line=dict(color=COLORS["border"], width=1)),
                textinfo="label+percent", textposition="outside",
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            )])
            fig3 = plot_style(fig3)
            fig3.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            fig3.add_annotation(text=f"<b>{total:,}</b><br><span style='color:{COLORS['muted']}; font-size:11px;'>TOTAL</span>",
                                x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=COLORS["text"]), align="center")
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Could not render mix chart: {e}")

    with cB:
        st.markdown("""
        <div class="section-card">
            <h4>Ratings Distribution</h4>
            <p class="desc">Maturity ratings across the catalog.</p>
        </div>
        """, unsafe_allow_html=True)
        try:
            rcount = df["rating"].value_counts().head(8).sort_values(ascending=True)
            fig4 = px.bar(x=rcount.values, y=rcount.index, orientation="h",
                          color_discrete_sequence=[COLORS["netflix_red"]])
            fig4 = plot_style(fig4)
            fig4.update_layout(height=300, xaxis_title="Titles", yaxis_title="", margin=dict(l=10, r=10, t=10, b=10))
            fig4.update_traces(marker_line_width=0)
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Could not render ratings chart: {e}")

    # Genre breakdown
    st.markdown("""
    <div class="section-card" style="margin-top:6px;">
        <h4>Top Genres</h4>
        <p class="desc">Aggregated from <code>listed_in</code> — counts the primary genre per title.</p>
    </div>
    """, unsafe_allow_html=True)
    try:
        genres = df["listed_in"].fillna("Unknown").astype(str).str.split(",").str[0].str.strip().value_counts().head(12)
        fig5 = px.bar(x=genres.index, y=genres.values, color=genres.values,
                      color_continuous_scale=[[0, "#2A2A2A"], [1, COLORS["netflix_red"]]])
        fig5 = plot_style(fig5)
        fig5.update_layout(height=300, xaxis_title="", yaxis_title="Titles", coloraxis_showscale=False, xaxis_tickangle=-22)
        fig5.update_traces(hovertemplate="%{x}: %{y}<extra></extra>")
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.caption(f"Genre chart unavailable: {e}")

# ---------------------------------------------------------------------------
# Page: Recommendations
# ---------------------------------------------------------------------------
elif page == "Recommendations":
    section_header("Discovery", "Content Recommendations", "TF-IDF + Cosine Similarity over director, cast, genre, and description. Powered by the FastAPI recommendation service.")

    # Controls
    df = load_data()
    titles = sorted(df["title"].dropna().astype(str).unique().tolist()) if df is not None else []

    c1, c2, c3 = st.columns([2.2, 0.7, 0.9])
    with c1:
        # Use selectbox with search when catalog is available, fallback to text_input
        if titles:
            title = st.selectbox("Select a title", options=[""] + titles, index=0, placeholder="Search — e.g. Stranger Things")
            # Allow free text override
            with st.expander("Or type a title manually", expanded=False):
                manual = st.text_input("Manual title", placeholder="Exact title as in catalog")
                if manual.strip():
                    title = manual.strip()
        else:
            title = st.text_input("Enter a title", placeholder="e.g. Stranger Things")
    with c2:
        n = st.slider("Results", 1, 10, 5, help="Number of similar titles to return")
    with c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        go_btn = st.button("Get Recommendations  →", use_container_width=True, type="primary")

    # Info strip
    st.markdown(f"""
    <div style="display:flex; gap:8px; flex-wrap:wrap; margin: 6px 0 14px 0; font-size:11px;">
        <span style="background:{COLORS['surface']}; border:1px solid {COLORS['border']}; color:{COLORS['muted']}; padding:6px 10px; border-radius:999px;">Model: <strong style="color:{COLORS['text']};">TF-IDF (5k features)</strong></span>
        <span style="background:{COLORS['surface']}; border:1px solid {COLORS['border']}; color:{COLORS['muted']}; padding:6px 10px; border-radius:999px;">Metric: <strong style="color:{COLORS['text']};">Cosine Similarity</strong></span>
        <span style="background:{COLORS['surface']}; border:1px solid {COLORS['border']}; color:{COLORS['muted']}; padding:6px 10px; border-radius:999px;">API: <strong style="color:{COLORS['text']};">{API_URL}</strong></span>
    </div>
    """, unsafe_allow_html=True)

    if go_btn:
        if not title or not str(title).strip():
            st.warning("Please select or enter a title.")
        else:
            title_clean = str(title).strip()
            with st.spinner(f"Finding titles similar to “{title_clean}” …"):
                data, err, used_url = call_recommend_api(title_clean, n)

            if data is not None:
                # Normalize response shape
                items = data if isinstance(data, list) else data.get("recommendations") or data.get("results") or data.get("data") or []
                if not items:
                    st.info("No recommendations returned. Try another title.")
                else:
                    st.success(f"Top {len(items)} titles similar to **{title_clean}**  ·  via `{used_url}`")
                    for idx, r in enumerate(items, 1):
                        r_title = r.get("title", "Untitled")
                        r_type = r.get("type", r.get("content_type", "—"))
                        r_genre = r.get("genre", r.get("listed_in", r.get("genres", "—")))
                        score = r.get("similarity_score", r.get("similarity", r.get("score", None)))
                        score_str = f"{float(score):.3f}" if isinstance(score, (int, float)) else "—"
                        st.markdown(f"""
                        <div class="rec-card" style="margin-bottom:10px;">
                            <div style="flex:1;">
                                <div style="display:flex; align-items:center; gap:8px;">
                                    <span style="width:26px; height:26px; border-radius:999px; background:{COLORS['surface_2']}; border:1px solid {COLORS['border']}; display:inline-flex; align-items:center; justify-content:center; font-size:11px; font-weight:800; color:{COLORS['muted']};">{idx}</span>
                                    <p class="rec-title">{r_title}</p>
                                </div>
                                <div class="rec-meta">{r_type} &nbsp;·&nbsp; {r_genre}</div>
                            </div>
                            <span class="rec-badge">◉ {score_str}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    # Raw JSON for debugging
                    with st.expander("View raw API response"):
                        st.json(data)
            else:
                st.error(f"Recommendation request failed. Tried multiple endpoints.\n\nLast error: `{err}`")
                st.caption("Falling back to local similarity is unavailable in this UI build — ensure the API service is running (`docker-compose up api`) and that models are built via the training DAG.")
                if df is not None and title_clean in df["title"].values:
                    st.info("Tip: the title exists in the local catalog, so the failure is API connectivity — not a missing title.")

    else:
        # Empty state / how-to
        st.markdown(f"""
        <div style="background:{COLORS['surface']}; border:1px solid {COLORS['border']}; border-radius:12px; padding:18px; display:flex; gap:16px; align-items:flex-start;">
            <div style="width:40px; height:40px; border-radius:10px; background:rgba(229,9,20,0.14); border:1px solid rgba(229,9,20,0.3); display:flex; align-items:center; justify-content:center; font-size:18px; flex-shrink:0;">✦</div>
            <div>
                <div style="font-weight:700; color:{COLORS['text']}; font-size:13.5px; margin-bottom:4px;">How it works</div>
                <div style="font-size:12.5px; color:{COLORS['muted']}; line-height:1.5;">
                    The engine vectorizes <code>director + cast + listed_in + description</code> with TF-IDF (5,000 features) and ranks by cosine similarity to the query title.
                    Select a title above and click <strong style="color:{COLORS['text']};">Get Recommendations</strong>. Results include similarity scores — higher means closer content match.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if df is not None:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:12px; font-weight:700; color:{COLORS['text']}; margin-bottom:8px;'>Try these titles</div>", unsafe_allow_html=True)
            # Show 6 random sample titles as quick pills
            try:
                samples = df["title"].dropna().sample(min(6, len(df)), random_state=42).tolist()
                cols = st.columns(len(samples))
                for col, t in zip(cols, samples):
                    with col:
                        if st.button(t, key=f"sample_{t}", use_container_width=True):
                            st.session_state["_prefill_title"] = t
                            st.rerun()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Page: Network Analysis
# ---------------------------------------------------------------------------
elif page == "Network Analysis":
    section_header("Graph Intelligence", "Director–Actor Collaboration Network", "Explore creative clusters and frequent collaborators. Built from the catalog's director and cast fields.")

    df = load_data()
    if df is None:
        st.error("Dataset not found — network analysis requires the Netflix titles CSV.")
        st.stop()

    # Controls
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        min_titles = st.slider("Min titles per person", 2, 10, 3, help="Filter to prolific directors/actors only")
    with c2:
        top_n = st.slider("Top collaborators shown", 10, 80, 30)
    with c3:
        layout_choice = st.selectbox("Layout", ["Force", "Circular", "Grid"], index=0)

    # Build co-occurrence — lightweight, no networkx hard dependency for layout
    try:
        import networkx as nx  # type: ignore
        has_nx = True
    except Exception:
        has_nx = False

    # Prepare edges: director <-> top cast members
    edges = []
    node_counts = {}
    for _, row in df.iterrows():
        director = str(row.get("director", "")).strip()
        cast_raw = str(row.get("cast", "")).strip()
        if not director or director.lower() in ("nan", "unknown", ""):
            continue
        # Take top 3 cast members to keep graph readable
        cast_members = [c.strip() for c in cast_raw.split(",") if c.strip() and c.strip().lower() != "unknown"][:3]
        for actor in cast_members:
            edges.append((director, actor))
            node_counts[director] = node_counts.get(director, 0) + 1
            node_counts[actor] = node_counts.get(actor, 0) + 1

    # Filter by min_titles
    keep_nodes = {n for n, c in node_counts.items() if c >= min_titles}
    filtered_edges = [(a, b) for a, b in edges if a in keep_nodes and b in keep_nodes]

    if not filtered_edges:
        st.warning("No collaborations meet the current threshold. Lower the 'Min titles' slider.")
        st.stop()

    # Aggregate edge weights
    from collections import Counter
    edge_weights = Counter(filtered_edges)
    # Keep top_n edges by weight
    top_edges = edge_weights.most_common(top_n)
    nodes = sorted({n for e, _ in top_edges for n in e})

    # Stats
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Nodes", f"{len(nodes):,}", "directors & actors")
    with m2: metric_card("Edges", f"{len(top_edges):,}", "collaborations")
    with m3:
        # Most connected
        deg = Counter()
        for (a, b), w in top_edges:
            deg[a] += w; deg[b] += w
        hub, hub_w = deg.most_common(1)[0] if deg else ("—", 0)
        metric_card("Top Hub", hub[:18], f"<strong>{hub_w}</strong> links" if hub != "—" else "")
    with m4:
        # Directors vs actors estimate: directors are those that appeared as first element more often
        directors_in_graph = len({a for (a, _), _ in top_edges})
        metric_card("Directors", f"{directors_in_graph}", f"{len(nodes)-directors_in_graph} actors")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.7, 1])

    with left:
        st.markdown("""
        <div class="section-card">
            <h4>Collaboration Graph</h4>
            <p class="desc">Node size = degree, edge width = number of shared titles. Hover for details.</p>
        </div>
        """, unsafe_allow_html=True)

        # Build positions
        import math, random
        random.seed(42)
        pos = {}
        if layout_choice == "Circular":
            for i, n in enumerate(nodes):
                ang = 2 * math.pi * i / max(1, len(nodes))
                pos[n] = (math.cos(ang), math.sin(ang))
        elif layout_choice == "Grid":
            cols = math.ceil(math.sqrt(len(nodes)))
            for i, n in enumerate(nodes):
                pos[n] = (i % cols, i // cols)
        else:  # Force-like: random + slight repulsion via circular jitter
            for i, n in enumerate(nodes):
                ang = 2 * math.pi * i / max(1, len(nodes))
                r = 0.7 + random.random() * 0.6
                pos[n] = (r * math.cos(ang) + random.uniform(-0.15, 0.15),
                          r * math.sin(ang) + random.uniform(-0.15, 0.15))

        # Normalize to [0,1]
        xs = [pos[n][0] for n in nodes]; ys = [pos[n][1] for n in nodes]
        min_x, max_x = min(xs), max(xs); min_y, max_y = min(ys), max(ys)
        span_x = (max_x - min_x) or 1; span_y = (max_y - min_y) or 1
        for n in nodes:
            x, y = pos[n]
            pos[n] = ((x - min_x) / span_x, (y - min_y) / span_y)

        # Degrees for sizing
        deg = Counter()
        for (a, b), w in top_edges:
            deg[a] += w; deg[b] += w
        max_deg = max(deg.values()) if deg else 1

        # Edge trace
        edge_x, edge_y = [], []
        for (a, b), w in top_edges:
            x0, y0 = pos[a]; x1, y1 = pos[b]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                line=dict(width=1, color="rgba(255,255,255,0.18)"),
                                hoverinfo="none", showlegend=False)

        node_x = [pos[n][0] for n in nodes]
        node_y = [pos[n][1] for n in nodes]
        node_size = [10 + 18 * (deg[n] / max_deg) for n in nodes]
        # Color directors vs actors: directors are those who appear as source in edges
        director_set = {a for (a, _), _ in top_edges}
        node_color = [COLORS["netflix_red"] if n in director_set else "#E5E5E5" for n in nodes]
        node_text = [f"{n}<br>Degree: {deg[n]}<br>{'Director' if n in director_set else 'Actor'}" for n in nodes]

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            text=[n if deg[n] >= sorted(deg.values(), reverse=True)[min(4, len(deg)-1)] else "" for n in nodes],
            textposition="top center", textfont=dict(size=9, color=COLORS["muted"]),
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color=COLORS["border"]), opacity=0.95),
            hovertext=node_text, hoverinfo="text", showlegend=False,
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig = plot_style(fig)
        fig.update_layout(height=520, xaxis=dict(visible=False), yaxis=dict(visible=False),
                          margin=dict(l=10, r=10, t=10, b=10),
                          annotations=[dict(text=f"<span style='color:{COLORS['netflix_red']};'>●</span> Director &nbsp; <span style='color:#E5E5E5;'>●</span> Actor",
                                            xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False,
                                            font=dict(size=11, color=COLORS["muted"]), align="left")])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown("""
        <div class="section-card">
            <h4>Top Collaborations</h4>
            <p class="desc">Most frequent director–actor pairings.</p>
        </div>
        """, unsafe_allow_html=True)
        # Table of top edges
        import pandas as pd  # already imported
        top_df = pd.DataFrame([{"Director": a, "Actor": b, "Titles Together": w} for (a, b), w in top_edges])
        top_df = top_df.sort_values("Titles Together", ascending=False).head(12)
        st.dataframe(top_df, use_container_width=True, hide_index=True, height=320)

        st.markdown("""
        <div class="section-card" style="margin-top:12px;">
            <h4>Most Connected</h4>
            <p class="desc">By total collaboration weight.</p>
        </div>
        """, unsafe_allow_html=True)
        hub_df = pd.DataFrame([{"Name": n, "Links": c, "Role": "Director" if n in director_set else "Actor"} for n, c in deg.most_common(10)])
        st.dataframe(hub_df, use_container_width=True, hide_index=True, height=260)

    if not has_nx:
        st.caption("Tip: install `networkx` for advanced centrality metrics (betweenness, community detection). Current view uses a lightweight in-app layout.")

# ---------------------------------------------------------------------------
# Page: Data Explorer
# ---------------------------------------------------------------------------
else:  # Data Explorer
    section_header("Catalog", "Data Explorer", "Search, filter, and export the Netflix catalog. Filters combine with AND logic.")

    df = load_data()
    if df is None:
        st.error("Dataset not found. Check `airflow/data/raw/netflix_titles.csv` or the mounted `data/` volume.")
        st.stop()

    # Filters
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1.2])
    with f1:
        type_opts = ["All"] + sorted(df["type"].dropna().astype(str).unique().tolist())
        sel_type = st.selectbox("Type", type_opts, index=0)
    with f2:
        # Country primary
        country_primary = df["country"].fillna("Unknown").astype(str).str.split(",").str[0].str.strip()
        country_opts = ["All"] + sorted(country_primary.unique().tolist())[:40]
        sel_country = st.selectbox("Country (primary)", country_opts, index=0)
    with f3:
        rating_opts = ["All"] + sorted(df["rating"].dropna().astype(str).unique().tolist())
        sel_rating = st.selectbox("Rating", rating_opts, index=0)
    with f4:
        q = st.text_input("Search", placeholder="Title, director, cast, genre…", label_visibility="visible")

    # Year range
    try:
        ymin, ymax = int(df["release_year"].min()), int(df["release_year"].max())
    except Exception:
        ymin, ymax = 1940, 2025
    yr = st.slider("Release year", ymin, ymax, (max(ymin, 2010), ymax))

    # Apply filters
    filtered = df.copy()
    if sel_type != "All":
        filtered = filtered[filtered["type"].astype(str) == sel_type]
    if sel_country != "All":
        filtered = filtered[filtered["country"].fillna("Unknown").astype(str).str.contains(sel_country, na=False)]
    if sel_rating != "All":
        filtered = filtered[filtered["rating"].astype(str) == sel_rating]
    filtered = filtered[(filtered["release_year"] >= yr[0]) & (filtered["release_year"] <= yr[1])]
    if q.strip():
        ql = q.strip().lower()
        # Search across key text columns
        cols_to_search = [c for c in ["title", "director", "cast", "listed_in", "description", "country"] if c in filtered.columns]
        mask = False
        for c in cols_to_search:
            mask = mask | filtered[c].astype(str).str.lower().str.contains(ql, na=False)
        filtered = filtered[mask]

    # Summary bar
    st.markdown(f"""
    <div style="display:flex; gap:10px; flex-wrap:wrap; margin: 10px 0 12px 0; align-items:center; font-size:12px;">
        <span style="background:{COLORS['surface']}; border:1px solid {COLORS['border']}; color:{COLORS['text']}; padding:7px 12px; border-radius:999px; font-weight:700;">{len(filtered):,} / {len(df):,} titles</span>
        <span style="color:{COLORS['muted']};">Showing filtered results · sorted by release year</span>
        <span style="flex:1;"></span>
    </div>
    """, unsafe_allow_html=True)

    # Columns to show — keep it tidy
    preferred_cols = ["title", "type", "director", "country", "release_year", "rating", "listed_in", "duration"]
    show_cols = [c for c in preferred_cols if c in filtered.columns]
    # Sort by release_year desc then title
    try:
        filtered = filtered.sort_values(["release_year", "title"], ascending=[False, True])
    except Exception:
        pass

    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True, height=520)

    # Export
    dl1, dl2 = st.columns([1, 4])
    with dl1:
        csv = filtered[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇  Download CSV", data=csv, file_name="netflix_filtered.csv", mime="text/csv", use_container_width=True)
    with dl2:
        st.caption("Export respects all active filters and search. Use the column menu (⋮) to sort and resize.")

# ---------------------------------------------------------------------------
# Footer (all pages)
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="footer">
    <span>Netflix Engineering Platform · Streamlit UI · <span style="color:{COLORS['muted']};">Design System v1 — Dark / Netflix Red</span></span>
    <span><a href="{API_URL}/docs" target="_blank">API Docs</a> &nbsp;·&nbsp; <a href="http://localhost:5001" target="_blank">MLflow</a> &nbsp;·&nbsp; <a href="http://localhost:3000" target="_blank">Grafana</a></span>
</div>
""", unsafe_allow_html=True)
