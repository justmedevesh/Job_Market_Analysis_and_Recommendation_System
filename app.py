import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIG — must be the very first Streamlit command
# ============================================================
st.set_page_config(
    page_title="JobLens — Job Market Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — Premium dark theme
# ============================================================
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── CSS Variables ── */
:root {
    --bg-primary:    #0a0e1a;
    --bg-card:       #12172b;
    --bg-card-hover: #171e36;
    --accent:        #6366f1;
    --accent-light:  #818cf8;
    --accent-glow:   rgba(99,102,241,.25);
    --purple:        #a78bfa;
    --green:         #10b981;
    --green-light:   #34d399;
    --amber:         #f59e0b;
    --rose:          #f43f5e;
    --text:          #f1f5f9;
    --text-dim:      #94a3b8;
    --border:        rgba(99,102,241,.18);
    --radius:        16px;
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #12172b 0%, #0d1117 40%, #17122b 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: -60%;
    background:
        radial-gradient(circle at 25% 40%, rgba(99,102,241,.10) 0%, transparent 55%),
        radial-gradient(circle at 75% 60%, rgba(167,139,250,.08) 0%, transparent 55%);
    animation: hero-glow 10s ease-in-out infinite alternate;
    pointer-events: none;
}
@keyframes hero-glow { 0%{opacity:.4} 100%{opacity:1} }

.hero h1 {
    font-size: 2.3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #c7d2fe, #a78bfa, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 .4rem;
    position: relative;
}
.hero p {
    color: var(--text-dim);
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
    position: relative;
}

/* ── Metric cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.8rem; }
.kpi {
    background: linear-gradient(145deg, #12172b, #0f1322);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: all .3s ease;
}
.kpi:hover {
    border-color: var(--accent);
    transform: translateY(-3px);
    box-shadow: 0 10px 30px var(--accent-glow);
}
.kpi-val {
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-light), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-label {
    color: var(--text-dim);
    font-size: .78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: .3rem;
}

/* ── Section titles ── */
.sec-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    margin: 1.5rem 0 .8rem;
    padding-bottom: .45rem;
    border-bottom: 2px solid var(--border);
}

/* ── Job result cards ── */
.job-card {
    background: linear-gradient(145deg, #12172b, #0f1322);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin-bottom: .7rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all .3s ease;
}
.job-card:hover {
    border-color: var(--accent);
    box-shadow: 0 4px 20px var(--accent-glow);
    transform: translateX(5px);
}
.job-card .jt {
    font-weight: 600;
    color: var(--text);
    font-size: .95rem;
    flex: 1;
}
.job-card .jc {
    color: var(--purple);
    font-size: .82rem;
    font-weight: 500;
    background: rgba(167,139,250,.10);
    padding: .25rem .85rem;
    border-radius: 20px;
    border: 1px solid rgba(167,139,250,.22);
    white-space: nowrap;
    margin-left: 1rem;
}
.sim-badge {
    color: var(--green-light);
    font-size: .75rem;
    font-weight: 600;
    background: rgba(16,185,129,.10);
    padding: .2rem .65rem;
    border-radius: 20px;
    border: 1px solid rgba(16,185,129,.22);
    margin-left: .6rem;
    white-space: nowrap;
}

/* ── Salary result ── */
.salary-box {
    background: linear-gradient(135deg, #0d1f1a, #101d25);
    border: 1px solid rgba(16,185,129,.30);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.salary-box::before {
    content: '';
    position: absolute;
    inset: -80%;
    background: radial-gradient(circle at 50% 50%, rgba(16,185,129,.08), transparent 60%);
    pointer-events: none;
}
.salary-amt {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #34d399, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
}
.salary-lbl {
    color: #6ee7b7;
    font-size: .85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-top: .4rem;
    position: relative;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117, #111827) !important;
    border-right: 1px solid var(--border);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: rgba(18,23,43,.85);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: .7rem 1.4rem;
    color: var(--text-dim);
    font-weight: 500;
    transition: all .3s ease;
}
.stTabs [data-baseweb="tab"]:hover { border-color: var(--accent); color: var(--text); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent), var(--purple)) !important;
    color: #fff !important;
    border-color: transparent !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--purple)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .7rem 2rem !important;
    font-weight: 600 !important;
    font-size: .92rem !important;
    transition: all .3s ease !important;
    width: 100%;
    letter-spacing: .3px;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px var(--accent-glow) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs ── */
.stTextInput input,
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}
.stTextInput input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent-glow) !important; }

/* ── Slider ── */
.stSlider > div > div > div { background: var(--accent) !important; }

/* ── Plotly charts: transparent bg ── */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* ── Divider helper ── */
.grad-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    margin: 1.5rem 0;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 1.8rem 0 .8rem;
    color: var(--text-dim);
    font-size: .78rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .kpi-grid { grid-template-columns: repeat(2,1fr); }
    .hero h1 { font-size: 1.6rem; }
    .hero { padding: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER — Categorise job titles into groups
# (The raw CSV has no 'job_category' column; this creates one
#  to match the categories the ML model was trained on.)
# ============================================================
CATEGORY_KEYWORDS = {
    "Data Science": [
        "data scientist", "machine learning", "deep learning", "ai ",
        "artificial intelligence", "nlp", "computer vision", "data analy",
    ],
    "Data Engineering": [
        "data engineer", "etl", "data pipeline", "airflow", "spark",
        "hadoop", "data warehouse", "big data",
    ],
    "Web Development": [
        "web develop", "html", "css", "javascript", "react", "angular",
        "vue", "node.js", "django", "flask", "php", "laravel", "ruby on rails",
    ],
    "Frontend Development": [
        "frontend", "front-end", "front end", "ui develop",
    ],
    "Full Stack Development": [
        "full stack", "fullstack", "full-stack", "mern", "mean stack",
    ],
    "Mobile Development": [
        "mobile", "ios", "android", "flutter", "react native", "swift",
        "kotlin", "app develop",
    ],
    "DevOps": [
        "devops", "docker", "kubernetes", "ci/cd", "aws ", "azure",
        "cloud engineer", "sre", "infrastructure", "terraform",
    ],
    "Design": [
        "designer", "ui/ux", "ux ", "graphic design", "figma",
        "adobe", "illustrator", "photoshop", "branding",
    ],
    "Marketing": [
        "marketing", "seo", "social media", "content writer",
        "copywriter", "email market", "digital market", "media buyer",
        "ads manager", "google ads", "facebook ads",
    ],
    "Project Management": [
        "project manager", "scrum", "agile", "product manager",
        "program manager", "jira",
    ],
    "QA / Testing": [
        "qa", "quality assurance", "test engineer", "selenium",
        "automation test", "manual test",
    ],
    "Sales / Business": [
        "sales", "business develop", "account manager", "lead gen",
        "customer success", "business analyst",
    ],
    "WordPress / CMS": [
        "wordpress", "shopify", "woocommerce", "cms", "elementor",
        "webflow", "squarespace",
    ],
}


def _categorize(title: str) -> str:
    """Map a lowercased title string to one of the fixed categories."""
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in title:
                return cat
    return "Other"


# ============================================================
# DATA LOADING — cached to avoid re-reading 50 MB on every run
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    path = os.path.join(BASE_DIR, "data", "raw", "jobs.csv")
    df = pd.read_csv(path)

    # Basic cleaning
    df["title_clean"] = df["title"].fillna("").str.lower().str.strip()
    df = df[df["title_clean"] != ""].copy()
    df["country"] = df["country"].fillna("Unknown").str.strip()

    # Derive job_category (not in the raw CSV)
    df["job_category"] = df["title_clean"].apply(_categorize)

    # Ensure boolean
    df["is_hourly"] = df["is_hourly"].astype(bool)

    # Budget → numeric
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["hourly_low"] = pd.to_numeric(df["hourly_low"], errors="coerce")
    df["hourly_high"] = pd.to_numeric(df["hourly_high"], errors="coerce")

    return df


@st.cache_resource(show_spinner="Loading ML model …")
def load_model():
    model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
    cols_path = os.path.join(BASE_DIR, "models", "model_columns.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(cols_path, "rb") as f:
        model_columns = pickle.load(f)
    return model, model_columns


@st.cache_resource(show_spinner="Building search index …")
def build_tfidf(_df_titles):
    tfidf = TfidfVectorizer(stop_words="english", max_features=8000)
    matrix = tfidf.fit_transform(_df_titles)
    return tfidf, matrix


# ── Load everything ──
df = load_data()
model, model_columns = load_model()
tfidf, tfidf_matrix = build_tfidf(df["title_clean"])


# ============================================================
# CORE FUNCTIONS
# ============================================================
def recommend_jobs(query: str, top_n: int = 8):
    """Return top-N similar jobs for a search query."""
    query = query.lower().strip()
    idx_matches = df[df["title_clean"].str.contains(query, na=False)].index
    if len(idx_matches) == 0:
        return None, None
    anchor = idx_matches[0]
    sim_scores = cosine_similarity(tfidf_matrix[anchor], tfidf_matrix).flatten()
    # Exclude the anchor itself
    sim_scores[anchor] = -1
    top_idx = sim_scores.argsort()[::-1][:top_n]
    results = df.iloc[top_idx][["title", "country", "job_category"]].copy()
    results["similarity"] = [sim_scores[i] for i in top_idx]
    return results, df.iloc[anchor]["title"]


def predict_salary(category: str, country: str, is_hourly: bool) -> float:
    """Use the trained model to predict salary / budget."""
    input_df = pd.DataFrame(
        {"job_category": [category], "country": [country], "is_hourly": [int(is_hourly)]}
    )
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    return model.predict(input_encoded)[0]


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🔮 **JobLens**")
    st.caption("Job Market Intelligence Platform")
    st.markdown('<div class="grad-divider"></div>', unsafe_allow_html=True)

    st.markdown("### About")
    st.markdown(
        "JobLens combines **TF-IDF similarity search** with a trained "
        "**ML salary predictor** to help you explore the job market and "
        "estimate compensation across roles and regions."
    )

    st.markdown('<div class="grad-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Quick Stats")
    st.metric("Total Jobs", f"{len(df):,}")
    st.metric("Countries", f"{df['country'].nunique()}")
    st.metric("Job Categories", f"{df['job_category'].nunique()}")

    st.markdown('<div class="grad-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;color:#64748b;font-size:.75rem;'>"
        "Built with Streamlit & scikit-learn</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# HERO BANNER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🔮 JobLens — Job Market Intelligence</h1>
        <p>Discover similar roles, explore market trends, and predict salaries — all powered by machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TABS
# ============================================================
tab_dash, tab_rec, tab_sal, tab_explore = st.tabs(
    ["📊 Dashboard", "🔍 Recommendations", "💰 Salary Predictor", "🗂️ Data Explorer"]
)

# ────────────────────────────────────────────
# TAB 1 — Dashboard
# ────────────────────────────────────────────
with tab_dash:
    # KPI row
    total_jobs = len(df)
    total_countries = df["country"].nunique()
    total_categories = df["job_category"].nunique()
    avg_budget = df["budget"].dropna().mean()

    kpi_html = f"""
    <div class="kpi-grid">
        <div class="kpi">
            <div class="kpi-val">{total_jobs:,}</div>
            <div class="kpi-label">Total Jobs</div>
        </div>
        <div class="kpi">
            <div class="kpi-val">{total_countries}</div>
            <div class="kpi-label">Countries</div>
        </div>
        <div class="kpi">
            <div class="kpi-val">{total_categories}</div>
            <div class="kpi-label">Categories</div>
        </div>
        <div class="kpi">
            <div class="kpi-val">${avg_budget:,.0f}</div>
            <div class="kpi-label">Avg Budget</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="sec-title">📂 Jobs by Category</div>', unsafe_allow_html=True)
        cat_counts = (
            df["job_category"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Category", "job_category": "Category", "count": "Jobs"})
        )
        if "Jobs" not in cat_counts.columns:
            cat_counts.columns = ["Category", "Jobs"]

        fig_cat = px.bar(
            cat_counts,
            x="Jobs",
            y="Category",
            orientation="h",
            color="Jobs",
            color_continuous_scale=["#312e81", "#6366f1", "#a78bfa", "#c4b5fd"],
        )
        fig_cat.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=420,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_right:
        st.markdown('<div class="sec-title">🌍 Top 15 Countries</div>', unsafe_allow_html=True)
        top_countries = (
            df[df["country"] != "Unknown"]["country"]
            .value_counts()
            .head(15)
            .reset_index()
        )
        top_countries.columns = ["Country", "Jobs"]

        fig_geo = px.bar(
            top_countries,
            x="Jobs",
            y="Country",
            orientation="h",
            color="Jobs",
            color_continuous_scale=["#064e3b", "#10b981", "#34d399", "#6ee7b7"],
        )
        fig_geo.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=420,
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    # Hourly vs Fixed pie
    st.markdown('<div class="sec-title">⏱️ Hourly vs Fixed-Price Distribution</div>', unsafe_allow_html=True)
    hourly_counts = df["is_hourly"].value_counts().reset_index()
    hourly_counts.columns = ["Type", "Count"]
    hourly_counts["Type"] = hourly_counts["Type"].map({True: "Hourly", False: "Fixed Price"})

    col_pie, col_info = st.columns([1, 1], gap="large")
    with col_pie:
        fig_pie = px.pie(
            hourly_counts,
            values="Count",
            names="Type",
            color_discrete_sequence=["#6366f1", "#10b981"],
            hole=0.55,
        )
        fig_pie.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=13)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_info:
        st.markdown("")
        hourly_pct = (df["is_hourly"].sum() / len(df) * 100)
        fixed_pct = 100 - hourly_pct
        avg_hourly_low = df["hourly_low"].dropna().mean()
        avg_hourly_high = df["hourly_high"].dropna().mean()

        st.markdown(
            f"""
            <div class="kpi" style="margin-bottom:1rem;">
                <div class="kpi-val">{hourly_pct:.1f}%</div>
                <div class="kpi-label">Hourly Contracts</div>
            </div>
            <div class="kpi" style="margin-bottom:1rem;">
                <div class="kpi-val">{fixed_pct:.1f}%</div>
                <div class="kpi-label">Fixed-Price Projects</div>
            </div>
            <div class="kpi">
                <div class="kpi-val">${avg_hourly_low:.0f}–${avg_hourly_high:.0f}</div>
                <div class="kpi-label">Avg Hourly Range</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ────────────────────────────────────────────
# TAB 2 — Recommendations
# ────────────────────────────────────────────
with tab_rec:
    st.markdown('<div class="sec-title">🔍 Job Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8;margin-bottom:1.2rem;'>"
        "Enter a job title and we'll find the most similar roles using TF-IDF cosine similarity.</p>",
        unsafe_allow_html=True,
    )

    r_col1, r_col2 = st.columns([3, 1])
    with r_col1:
        user_query = st.text_input(
            "Job title",
            placeholder="e.g. Data Scientist, React Developer, UX Designer …",
            label_visibility="collapsed",
        )
    with r_col2:
        top_n = st.slider("Results", 3, 20, 8, label_visibility="collapsed")

    if st.button("🔍  Find Similar Jobs", key="btn_rec"):
        if not user_query.strip():
            st.warning("⚠️ Please enter a job title to search.")
        else:
            with st.spinner("Searching …"):
                results, anchor_title = recommend_jobs(user_query, top_n)

            if results is None:
                st.error("❌ No matching jobs found. Try a different keyword.")
            else:
                st.markdown(
                    f"<p style='color:#818cf8;font-weight:600;margin-bottom:.5rem;'>"
                    f"Anchor job: <span style='color:#c7d2fe'>{anchor_title}</span></p>",
                    unsafe_allow_html=True,
                )
                for _, row in results.iterrows():
                    sim_pct = row["similarity"] * 100
                    card_html = f"""
                    <div class="job-card">
                        <span class="jt">{row['title']}</span>
                        <span class="sim-badge">{sim_pct:.0f}% match</span>
                        <span class="jc">📍 {row['country']}</span>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

# ────────────────────────────────────────────
# TAB 3 — Salary Predictor
# ────────────────────────────────────────────
with tab_sal:
    st.markdown('<div class="sec-title">💰 ML Salary Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8;margin-bottom:1.2rem;'>"
        "Select a job category, country and contract type to get an estimated salary / budget "
        "from our trained model.</p>",
        unsafe_allow_html=True,
    )

    s_col1, s_col2, s_col3 = st.columns(3, gap="large")

    # Get the unique categories that actually appear in model_columns
    all_cats = sorted(df["job_category"].unique())

    with s_col1:
        sel_category = st.selectbox("Job Category", all_cats, key="sel_cat")
    with s_col2:
        country_list = sorted(df["country"].unique())
        sel_country = st.selectbox("Country", country_list, key="sel_country")
    with s_col3:
        sel_hourly = st.selectbox("Contract Type", ["Fixed Price", "Hourly"], key="sel_hourly")

    is_hourly = sel_hourly == "Hourly"

    if st.button("💰  Predict Salary", key="btn_sal"):
        with st.spinner("Running prediction …"):
            salary = predict_salary(sel_category, sel_country, is_hourly)

        unit = "/hr" if is_hourly else " (Fixed)"
        st.markdown(
            f"""
            <div class="salary-box">
                <div class="salary-amt">${salary:,.2f}{unit}</div>
                <div class="salary-lbl">Estimated Compensation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Context: show average for that category
        cat_avg = df.loc[df["job_category"] == sel_category, "budget"].dropna().mean()
        country_avg = df.loc[df["country"] == sel_country, "budget"].dropna().mean()

        ctx1, ctx2 = st.columns(2)
        with ctx1:
            st.markdown(
                f"""<div class="kpi" style="margin-top:1rem;">
                    <div class="kpi-val">${cat_avg:,.0f}</div>
                    <div class="kpi-label">Avg for {sel_category}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with ctx2:
            st.markdown(
                f"""<div class="kpi" style="margin-top:1rem;">
                    <div class="kpi-val">${country_avg:,.0f}</div>
                    <div class="kpi-label">Avg in {sel_country}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ────────────────────────────────────────────
# TAB 4 — Data Explorer
# ────────────────────────────────────────────
with tab_explore:
    st.markdown('<div class="sec-title">🗂️ Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8;margin-bottom:1.2rem;'>"
        "Filter and explore the raw job listings interactively.</p>",
        unsafe_allow_html=True,
    )

    e_col1, e_col2, e_col3 = st.columns(3, gap="large")
    with e_col1:
        filt_cat = st.multiselect("Category", sorted(df["job_category"].unique()), key="filt_cat")
    with e_col2:
        filt_country = st.multiselect("Country", sorted(df["country"].unique()), key="filt_country")
    with e_col3:
        filt_hourly = st.selectbox("Type", ["All", "Hourly", "Fixed Price"], key="filt_type")

    filtered = df.copy()
    if filt_cat:
        filtered = filtered[filtered["job_category"].isin(filt_cat)]
    if filt_country:
        filtered = filtered[filtered["country"].isin(filt_country)]
    if filt_hourly == "Hourly":
        filtered = filtered[filtered["is_hourly"] == True]
    elif filt_hourly == "Fixed Price":
        filtered = filtered[filtered["is_hourly"] == False]

    st.markdown(
        f"<p style='color:#818cf8;font-weight:600;'>Showing {len(filtered):,} of {len(df):,} jobs</p>",
        unsafe_allow_html=True,
    )

    display_cols = ["title", "job_category", "country", "is_hourly", "budget", "hourly_low", "hourly_high"]
    st.dataframe(
        filtered[display_cols].head(500),
        use_container_width=True,
        height=480,
    )

    # Budget distribution for filtered set
    budget_data = filtered["budget"].dropna()
    if len(budget_data) > 10:
        st.markdown('<div class="sec-title">💵 Budget Distribution (Filtered)</div>', unsafe_allow_html=True)
        # Cap at 99th percentile for cleaner chart
        cap = budget_data.quantile(0.99)
        fig_hist = px.histogram(
            budget_data[budget_data <= cap],
            nbins=60,
            color_discrete_sequence=["#6366f1"],
            labels={"value": "Budget ($)", "count": "Jobs"},
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            showlegend=False,
            xaxis_title="Budget ($)",
            yaxis_title="Number of Jobs",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="app-footer">'
    "🔮 <strong>JobLens</strong> — Job Market Analysis & Recommendation System &nbsp;|&nbsp; "
    "Built by Devesh Kumar Mandal"
    "</div>",
    unsafe_allow_html=True,
)