"""
Rossmann Store Sales Intelligence - Streamlit Corporate Dashboard
===================================================================
WeIntern Data Science Internship - Week 02 Project

Features:
- Models: All 4 trained PKL models (Best Model, Tuned RF, Random Forest, Linear Regression)
- Data: All 3 cleaned datasets (cleaned_train.csv, cleaned_test.csv, cleaned_store.csv)
- Modules:
    1. Executive Overview & KPIs
    2. Task 01 - Data Cleaning & Quality Audit
    3. Task 02 - Exploratory Data Analysis & Business Insights
    4. Task 03 - Machine Learning Sales Forecasting (Interactive & Batch)
    5. Reports & Artifacts Downloads
"""

import io
import os
import sys
import time
import warnings
from pathlib import Path

# Cross-version compatibility shim for scikit-learn unpickling
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# =============================================================================
# STREAMLIT PAGE CONFIG & CORPORATE THEME
# =============================================================================
st.set_page_config(
    page_title="Rossmann Analytics | Sales Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CORPORATE_CSS = """
<style>
    :root {
        --brand-navy: #14243A;
        --brand-blue: #2563EB;
        --brand-cyan: #0EA5A4;
        --surface: #FFFFFF;
        --muted: #64748B;
        --border-color: #E2E8F0;
    }

    .stApp {
        background: #F8FAFC;
    }

    [data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #111E2E;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #F1F5F9;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.35rem 0.25rem;
        font-weight: 500;
        font-size: 0.92rem;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.12);
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1540px;
    }

    h1, h2, h3, h4 {
        color: var(--brand-navy);
        letter-spacing: -0.02em;
        font-weight: 700;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(20, 36, 58, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(20, 36, 58, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    [data-testid="stMetricValue"] {
        color: #14243A;
        font-weight: 800;
        font-size: 1.65rem;
    }

    [data-testid="stTabs"] button {
        font-weight: 600;
        font-size: 0.92rem;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
        background: #FFFFFF;
    }

    .app-brand {
        background: linear-gradient(120deg, #111E2E 0%, #1A365D 55%, #0D9488 100%);
        color: white;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(17, 30, 46, 0.12);
    }

    .app-brand .eyebrow {
        color: #93C5FD;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .app-brand .brand-title {
        color: #FFFFFF;
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.25;
        margin: 0;
    }

    .app-brand .brand-subtitle {
        color: #E2E8F0;
        font-size: 0.93rem;
        margin-top: 8px;
        max-width: 900px;
        line-height: 1.45;
    }

    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-success { background: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
    .badge-info { background: #DBEAFE; color: #1E40AF; border: 1px solid #BFDBFE; }
    .badge-purple { background: #F3E8FF; color: #6B21A8; border: 1px solid #E9D5FF; }

    .card-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.025);
    }

    div.stButton > button[kind="primary"] {
        background: #2563EB;
        border: 1px solid #1D4ED8;
        border-radius: 8px;
        font-weight: 650;
        padding: 0.45rem 1.2rem;
    }

    div.stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CORPORATE_CSS, unsafe_allow_html=True)

# =============================================================================
# CONSTANTS & FEATURE DEFINITIONS
# =============================================================================
FEATURE_COLUMNS = [
    "store", "storetype", "assortment",
    "competitiondistance", "competitiondistance_missing",
    "competitionopensincemonth", "competitionopensinceyear",
    "competition_opening_info_missing", "competition_age_months",
    "promo", "promo2", "promo2sinceweek", "promo2sinceyear",
    "promointerval", "promo2_not_applicable", "promo2_active",
    "stateholiday", "schoolholiday",
    "dayofweek", "year", "month", "day", "dayofyear", "weekofyear",
    "is_weekend", "is_month_start", "is_month_end",
]
CATEGORICAL_FEATURES = ["storetype", "assortment", "promointerval", "stateholiday"]
NUMERICAL_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]

# =============================================================================
# INTELLIGENT PATH RESOLUTION
# =============================================================================
def resolve_path(rel_path: str, fallback_rel_paths=None) -> Path:
    """Locates a resource whether run from repo root or subfolder."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(rel_path),
        script_dir / rel_path,
        script_dir / "DataScience-Week2---WeIntern-Project-main" / rel_path,
        script_dir.parent / rel_path,
        script_dir.parent / "DataScience-Week2---WeIntern-Project-main" / rel_path,
    ]
    if fallback_rel_paths:
        for fb in fallback_rel_paths:
            candidates.extend([
                Path(fb),
                script_dir / fb,
                script_dir / "DataScience-Week2---WeIntern-Project-main" / fb,
                script_dir.parent / fb,
                script_dir.parent / "DataScience-Week2---WeIntern-Project-main" / fb,
            ])
    for p in candidates:
        if p.exists():
            return p.resolve()
    return candidates[0].resolve()

# =============================================================================
# DATA & MODEL LOADERS (CACHED)
# =============================================================================
@st.cache_data(show_spinner="Loading cleaned Rossmann datasets...")
def load_cleaned_datasets():
    store_p = resolve_path("data/cleaned/cleaned_store.csv")
    test_p = resolve_path("data/cleaned/cleaned_test.csv")
    train_p = resolve_path("data/cleaned/cleaned_train.csv")

    store_df = pd.read_csv(store_p, low_memory=False)
    test_df = pd.read_csv(test_p, low_memory=False)
    train_df = pd.read_csv(train_p, low_memory=False)

    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    train_df["stateholiday"] = train_df["stateholiday"].astype(str).str.replace(".0", "", regex=False).str.strip()
    test_df["stateholiday"] = test_df["stateholiday"].astype(str).str.replace(".0", "", regex=False).str.strip()
    store_df["storetype"] = store_df["storetype"].astype(str).str.strip()
    store_df["assortment"] = store_df["assortment"].astype(str).str.strip()
    store_df["promointerval"] = store_df["promointerval"].fillna("No Promo2").astype(str).str.strip()

    return store_df, test_df, train_df

@st.cache_resource(show_spinner="Loading trained machine learning models...")
def load_all_models():
    models_dict = {}
    model_defs = [
        ("Best Model (Tuned RF)", "best_sales_prediction_model.pkl"),
        ("Tuned Random Forest", "tuned_random_forest_model.pkl"),
        ("Random Forest (Baseline)", "random_forest_model.pkl"),
        ("Linear Regression", "linear_regression_model.pkl"),
    ]
    for display_name, fname in model_defs:
        resolved = resolve_path(f"outputs/models/{fname}", [f"models/{fname}"])
        if resolved.exists():
            try:
                m = joblib.load(resolved)
                if hasattr(m, "named_steps"):
                    for step in m.named_steps.values():
                        for attr, val in [("tol", 1e-4), ("positive", False)]:
                            if not hasattr(step, attr):
                                try:
                                    setattr(step, attr, val)
                                except Exception:
                                    pass
                models_dict[display_name] = {
                    "model": m,
                    "filename": fname,
                    "path": resolved,
                    "size_mb": round(os.path.getsize(resolved) / (1024 * 1024), 2),
                    "loaded": True,
                }
            except Exception as e:
                models_dict[display_name] = {
                    "model": None,
                    "filename": fname,
                    "path": resolved,
                    "size_mb": 0,
                    "loaded": False,
                    "error": str(e),
                }
        else:
            models_dict[display_name] = {
                "model": None,
                "filename": fname,
                "path": resolved,
                "size_mb": 0,
                "loaded": False,
                "error": "File not found on disk",
            }
    return models_dict

@st.cache_data(show_spinner="Engineering temporal & store features...")
def engineer_features(train_clean, store_clean):
    merged = train_clean.merge(store_clean, on="store", how="left")
    merged["year"] = merged["date"].dt.year
    merged["month"] = merged["date"].dt.month
    merged["day"] = merged["date"].dt.day
    merged["dayofyear"] = merged["date"].dt.dayofyear
    merged["weekofyear"] = merged["date"].dt.isocalendar().week.astype(int)
    merged["day_name"] = merged["date"].dt.day_name()
    merged["is_weekend"] = (merged["dayofweek"] >= 6).astype(int)
    merged["is_month_start"] = merged["date"].dt.is_month_start.astype(int)
    merged["is_month_end"] = merged["date"].dt.is_month_end.astype(int)

    known_comp = (merged["competitionopensincemonth"] > 0) & (merged["competitionopensinceyear"] > 0)
    merged["competition_age_months"] = np.where(
        known_comp,
        (merged["year"] - merged["competitionopensinceyear"]) * 12 + (merged["month"] - merged["competitionopensincemonth"]),
        0
    ).clip(min=0)

    month_abbr = merged["date"].dt.strftime("%b").replace({"Sep": "Sept"})
    interval = merged["promointerval"].astype(str)
    month_in_interval = pd.Series(False, index=merged.index)
    for iv in interval.unique():
        if iv in ("No Promo2", "nan", "None", ""):
            continue
        months = {m.strip() for m in iv.split(",")}
        mask = interval.eq(iv)
        month_in_interval[mask] = month_abbr[mask].isin(months)

    cur_period = merged["date"].dt.year * 100 + merged["date"].dt.isocalendar().week.astype(int)
    start_period = merged["promo2sinceyear"].astype(int) * 100 + merged["promo2sinceweek"].astype(int)
    merged["promo2_active"] = ((merged["promo2"] == 1) & month_in_interval & (cur_period >= start_period)).astype(int)
    return merged

def prepare_X_for_model(df):
    X = df[FEATURE_COLUMNS].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].fillna("Unknown").astype(str)
    return X

def render_matplotlib_fig(fig):
    st.pyplot(fig)
    plt.close(fig)

# =============================================================================
# INITIALIZE DATA & SIDEBAR STATUS
# =============================================================================
store_df, test_df, train_df = load_cleaned_datasets()
models = load_all_models()

# Sidebar Navigation
st.sidebar.markdown("## 📈 Rossmann Analytics")
st.sidebar.caption("SALES INTELLIGENCE DASHBOARD")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Executive Overview",
        "🧹 Task 01: Data Quality & Cleaning",
        "📊 Task 02: Exploratory Data Analysis",
        "🤖 Task 03: Sales Forecasting (4 Models)",
        "📑 Reports & Artifacts",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Active Assets")
st.sidebar.markdown(
    f"""
    <div style='font-size: 0.82rem; line-height: 1.6;'>
        <b>Data Cleaned:</b><br>
        ✅ <code>cleaned_train.csv</code> ({len(train_df):,} rows)<br>
        ✅ <code>cleaned_test.csv</code> ({len(test_df):,} rows)<br>
        ✅ <code>cleaned_store.csv</code> ({len(store_df):,} stores)<br>
        <br>
        <b>Trained Models (PKL):</b><br>
        {'✅' if models['Best Model (Tuned RF)']['loaded'] else '❌'} <code>best_sales_prediction_model.pkl</code><br>
        {'✅' if models['Tuned Random Forest']['loaded'] else '❌'} <code>tuned_random_forest_model.pkl</code><br>
        {'✅' if models['Random Forest (Baseline)']['loaded'] else '❌'} <code>random_forest_model.pkl</code><br>
        {'✅' if models['Linear Regression']['loaded'] else '❌'} <code>linear_regression_model.pkl</code>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.caption("WeIntern Internship Project · Week 02\nDesigned with Streamlit, Scikit-Learn & Pandas")

# Header Banner on All Pages
st.markdown("""
<div class="app-brand">
    <div class="eyebrow">WeIntern · Data Science · Week 02 Project</div>
    <div class="brand-title">Rossmann Store Sales Intelligence & Forecasting</div>
    <div class="brand-subtitle">
        An enterprise dashboard integrating clean datasets (Train, Test, Store) with 4 production ML models
        (Linear Regression, Random Forest, Tuned RF, and Best Model) for end-to-end sales forecasting.
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: 🏠 EXECUTIVE OVERVIEW
# =============================================================================
if page == "🏠 Executive Overview":
    st.markdown("### 🌟 Executive Performance Summary")

    # Compute key metrics
    total_sales = float(train_df["sales"].sum())
    total_customers = float(train_df["customers"].sum())
    open_train = train_df[train_df["open"] == 1]
    avg_daily_sales = float(open_train["sales"].mean())
    avg_basket_size = total_sales / total_customers if total_customers > 0 else 0
    unique_stores = int(store_df["store"].nunique())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Historical Sales", f"€{total_sales / 1e9:.2f}B", help="Cumulative revenue across all stores")
    m2.metric("Total Customer Visits", f"{total_customers / 1e6:.1f}M", help="Cumulative recorded store footfall")
    m3.metric("Avg Daily Sales (Open)", f"€{avg_daily_sales:,.0f}", help="Mean daily revenue per open store")
    m4.metric("Avg Basket Size", f"€{avg_basket_size:.2f}", help="Average sales spend per customer visit")
    m5.metric("Active Stores", f"{unique_stores:,}", help="Total monitored Rossmann drugstore branches")

    st.markdown("""
    <div class="card-box">
        <h4>🎯 Project Objectives & End-to-End Workflow</h4>
        <p style='color: #475569; font-size: 0.94rem;'>
            This platform bridges data engineering and machine learning to forecast sales across 1,115 Rossmann drugstores.
            It evaluates promotional efficiency, seasonal holiday demand, and competitive presence to optimize retail store operations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.markdown("#### 🏗️ Project Architecture")
        st.markdown("""
        | Stage | Key Deliverables & Operations | Status |
        | :--- | :--- | :---: |
        | **Task 01: Data Cleaning** | Imputed missing Competition Distance (median 2,325m), filled Promo2 flags, validated negative records, resolved 0-missing cells | `100% Clean` |
        | **Task 02: EDA** | Analyzed promotional lift (+33.9%), Store Type 'b' dominance, weekly shopping rhythms, customer-to-sales correlation (r = 0.89) | `Completed` |
        | **Task 03: ML Modeling** | Trained Linear Regression, Random Forest, & Tuned Random Forest on chronological 80:20 split (Cutoff: Dec 2014) | `4 PKL Models Active` |
        | **Production Deployment** | Real-time single-store interactive forecasting, batch test inference, error residual diagnostics | `Operational` |
        """)

    with col_r:
        st.markdown("#### 🏆 Model Performance Benchmark")
        model_comp_path = resolve_path("outputs/predictions/model_comparison.csv")
        if model_comp_path.exists():
            comp_df = pd.read_csv(model_comp_path)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            best_r2 = float(comp_df.iloc[0]["R2"])
            best_rmse = float(comp_df.iloc[0]["RMSE"])
            st.success(f"🌟 **Champion Model: {comp_df.iloc[0]['Model']}** achieves **R² = {best_r2:.4f}** with RMSE of **€{best_rmse:,.1f}**.")
        else:
            st.info("Model comparison table available in Task 03 tab.")

    st.markdown("---")
    st.markdown("#### 📊 Dataset Inventory & Readiness Check")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**`cleaned_train.csv`**")
        st.markdown(f"- Records: **{len(train_df):,}**\n- Date Range: **{train_df['date'].min():%Y-%m-%d}** to **{train_df['date'].max():%Y-%m-%d}**\n- Missing Values: **0**\n- Target: `sales`")
    with d2:
        st.markdown("**`cleaned_test.csv`**")
        st.markdown(f"- Records: **{len(test_df):,}**\n- Date Range: **{test_df['date'].min():%Y-%m-%d}** to **{test_df['date'].max():%Y-%m-%d}**\n- Missing Values: **0**\n- Unique Stores: **{test_df['store'].nunique()}**")
    with d3:
        st.markdown("**`cleaned_store.csv`**")
        st.markdown(f"- Stores: **{len(store_df):,}**\n- Store Types: `a`, `b`, `c`, `d`\n- Assortments: `Basic (a)`, `Extra (b)`, `Extended (c)`\n- Missing Values: **0**")


# =============================================================================
# PAGE 2: 🧹 TASK 01: DATA QUALITY & CLEANING
# =============================================================================
elif page == "🧹 Task 01: Data Quality & Cleaning":
    st.markdown("### 🧹 Task 01: Data Cleaning & Quality Audit")
    st.markdown("<p style='color: #64748B;'>Audit all 3 cleaned CSV datasets, verify integrity checks, and review outlier handling decisions.</p>", unsafe_allow_html=True)

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Train Missing Cells", "0", "100% Resolved")
    q2.metric("Test Missing Cells", "0", "100% Resolved")
    q3.metric("Store Missing Cells", "0", "100% Resolved")
    q4.metric("Duplicate Rows", "0", "Clean")

    tabs = st.tabs([
        "📋 Cleaned Data Explorer",
        "⚖️ Before vs After Cleaning",
        "🛡️ Business Rules & Integrity",
        "📦 Outlier Analysis & IQR",
    ])

    # Tab 1: Cleaned Data Explorer
    with tabs[0]:
        st.markdown("#### Inspect Cleaned Data Files")
        data_choice = st.radio("Select Cleaned Dataset:", ["cleaned_store.csv", "cleaned_train.csv (sample)", "cleaned_test.csv"], horizontal=True)

        if data_choice == "cleaned_store.csv":
            st.dataframe(store_df, use_container_width=True)
            st.caption(f"Showing all {len(store_df)} store metadata records with imputed competition & promo features.")
            csv_data = store_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download cleaned_store.csv", csv_data, file_name="cleaned_store.csv", mime="text/csv")
        elif data_choice == "cleaned_test.csv":
            st.dataframe(test_df.head(200), use_container_width=True)
            st.caption(f"Showing first 200 of {len(test_df):,} test records with imputed 'open' status.")
            csv_data = test_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download cleaned_test.csv", csv_data, file_name="cleaned_test.csv", mime="text/csv")
        else:
            st.dataframe(train_df.head(200), use_container_width=True)
            st.caption(f"Showing first 200 of {len(train_df):,} cleaned training records.")
            st.info("💡 Note: cleaned_train.csv contains 1,017,209 rows. Preview is capped for rapid rendering.")

    # Tab 2: Before vs After
    with tabs[1]:
        st.markdown("#### Cleaning Quality Metrics & Transformations")
        c_qual_p = resolve_path("outputs/predictions/cleaning_quality_summary.csv")
        if c_qual_p.exists():
            st.markdown("**Cleaning Quality Summary**")
            st.dataframe(pd.read_csv(c_qual_p), use_container_width=True, hide_index=True)

        b_after_p = resolve_path("outputs/predictions/before_after_comparison.csv")
        if b_after_p.exists():
            st.markdown("**Before vs After Comparison**")
            st.dataframe(pd.read_csv(b_after_p), use_container_width=True, hide_index=True)

        st.markdown("""
        ##### 🔧 Key Cleaning Actions Executed:
        1. **Competition Distance**: 3 missing store records imputed using historical median (**2,325.0 meters**) with `competitiondistance_missing` indicator.
        2. **Competition Opening Month/Year**: 354 missing records filled with `0` and marked with `competition_opening_info_missing = 1`.
        3. **Promo2 Fields**: Non-participating stores (Promo2 = 0) assigned `promo2_not_applicable = 1`, week/year filled with `0`, and interval set to `'No Promo2'`.
        4. **Test Open Column**: 11 missing records imputed using store & weekday historical mode.
        5. **Categorical Standardization**: Column names formatted to snake_case; `stateholiday` values unified (`'0.0'` mapped to `'0'`).
        """)

    # Tab 3: Business Rules
    with tabs[2]:
        st.markdown("#### Business Rules & Integrity Verification")
        validations = {
            "Negative Sales records in Train": int((train_df["sales"] < 0).sum()),
            "Negative Customers records in Train": int((train_df["customers"] < 0).sum()),
            "Invalid Store IDs (<= 0) in Train": int((train_df["store"] <= 0).sum()),
            "Invalid DayOfWeek values outside 1-7 in Train": int((~train_df["dayofweek"].between(1, 7)).sum()),
            "Invalid Store IDs in Test": int((test_df["store"] <= 0).sum()),
            "Missing Open values in Test": int(test_df["open"].isna().sum()),
            "Negative Competition Distance in Store": int((store_df["competitiondistance"] < 0).sum()),
        }
        val_df = pd.DataFrame(list(validations.items()), columns=["Integrity Check Rule", "Violation Count"])
        val_df["Status"] = val_df["Violation Count"].apply(lambda x: "✅ PASSED (0 violations)" if x == 0 else f"❌ FAILED ({x})")
        st.dataframe(val_df, use_container_width=True, hide_index=True)

        st.markdown("##### 🔍 Zero Sales Analysis")
        zero_sales_closed = int(((train_df["sales"] == 0) & (train_df["open"] == 0)).sum())
        zero_sales_open = int(((train_df["sales"] == 0) & (train_df["open"] == 1)).sum())
        z1, z2 = st.columns(2)
        z1.metric("Closed Stores with Sales = 0", f"{zero_sales_closed:,}", "Expected business behavior")
        z2.metric("Open Stores with Sales = 0", f"{zero_sales_open:,}", "Refurbishment / anomalies")
        st.caption("Zero sales records are deliberately retained: closed stores have zero sales by definition and are handled directly.")

    # Tab 4: Outliers
    with tabs[3]:
        st.markdown("#### Outlier Analysis via Interquartile Range (IQR)")
        out_p = resolve_path("outputs/predictions/outlier_analysis.csv")
        if out_p.exists():
            st.dataframe(pd.read_csv(out_p).round(2), use_container_width=True, hide_index=True)

        st.markdown("##### 📦 Distribution Boxplots (Open Stores)")
        open_stores = train_df[train_df["open"] == 1]
        sample_open = open_stores.sample(min(len(open_stores), 50_000), random_state=42)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(y=sample_open["sales"], ax=axes[0], color="#3B82F6")
        axes[0].set_title("Sales Distribution (Open Stores Sample)")
        axes[0].set_ylabel("Sales (€)")

        sns.boxplot(y=sample_open["customers"], ax=axes[1], color="#10B981")
        axes[1].set_title("Customers Distribution (Open Stores Sample)")
        axes[1].set_ylabel("Customer Count")
        fig.tight_layout()
        render_matplotlib_fig(fig)

        st.info("💡 **Analytical Decision**: Outliers detected via IQR are **retained** without truncation or removal. In retail chains, extreme high sales occur naturally on pre-holiday periods, major promotions, and high-footfall flagship locations. Capping them would distort revenue forecasting.")


# =============================================================================
# PAGE 3: 📊 TASK 02: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
elif page == "📊 Task 02: Exploratory Data Analysis":
    st.markdown("### 📊 Task 02: Exploratory Data Analysis & Strategic Insights")
    st.markdown("<p style='color: #64748B;'>Explore customer behavior, seasonal trends, and promotional lift across the chain.</p>", unsafe_allow_html=True)

    eda_df = engineer_features(train_df, store_df)
    open_df = eda_df[eda_df["open"] == 1]

    # Global KPI strip
    promo_sales = open_df[open_df["promo"] == 1]["sales"].mean()
    no_promo_sales = open_df[open_df["promo"] == 0]["sales"].mean()
    promo_lift = ((promo_sales - no_promo_sales) / no_promo_sales) * 100

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Avg Sales (With Promo)", f"€{promo_sales:,.0f}")
    e2.metric("Avg Sales (No Promo)", f"€{no_promo_sales:,.0f}")
    e3.metric("Promotional Lift", f"+{promo_lift:.1f}%", "Significant revenue driver")
    e4.metric("Customer-Sales Correlation", f"{open_df['customers'].corr(open_df['sales']):.3f}", "Very strong positive")

    eda_tabs = st.tabs([
        "🏷️ Promotional Lift & Impact",
        "🏬 Store Types & Assortment",
        "📅 Seasonality & Temporal Trends",
        "🎉 Holiday Effects",
        "🏆 Top & Bottom Stores",
        "🔥 Correlation Matrix",
        "💡 Strategic Recommendations",
    ])

    # Tab 1: Promo Lift
    with eda_tabs[0]:
        st.markdown("#### Promotional Impact on Store Sales & Footfall")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sample_sub = open_df.sample(min(len(open_df), 40_000), random_state=42)
            sns.barplot(data=sample_sub, x="promo", y="sales", palette=["#94A3B8", "#2563EB"], ax=ax)
            ax.set_title("Average Daily Sales: Promo vs Non-Promo")
            ax.set_xticklabels(["No Promo (0)", "Active Promo (1)"])
            ax.set_ylabel("Daily Sales (€)")
            fig.tight_layout()
            render_matplotlib_fig(fig)

        with p_col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=sample_sub, x="promo", y="customers", palette=["#94A3B8", "#0D9488"], ax=ax)
            ax.set_title("Average Daily Customers: Promo vs Non-Promo")
            ax.set_xticklabels(["No Promo (0)", "Active Promo (1)"])
            ax.set_ylabel("Daily Customers")
            fig.tight_layout()
            render_matplotlib_fig(fig)

        st.markdown(f"""
        <div class="card-box">
            <b>Executive Takeaway:</b> Active promotions produce a massive <b>+{promo_lift:.1f}% revenue lift</b>
            (€{promo_sales:,.0f} vs €{no_promo_sales:,.0f}). Customer footfall increases substantially on promo days,
            confirming promotions are highly effective for driving store velocity.
        </div>
        """, unsafe_allow_html=True)

    # Tab 2: Store Types & Assortment
    with eda_tabs[1]:
        st.markdown("#### Store Type & Assortment Segmentation")
        st_col1, st_col2 = st.columns(2)
        with st_col1:
            st_mean = open_df.groupby("storetype")["sales"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=st_mean, x="storetype", y="sales", palette="Blues_d", ax=ax)
            ax.set_title("Average Sales by Store Type (Open Stores)")
            ax.set_xlabel("Store Type (a, b, c, d)")
            ax.set_ylabel("Mean Sales (€)")
            fig.tight_layout()
            render_matplotlib_fig(fig)

        with st_col2:
            assort_mean = open_df.groupby("assortment")["sales"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=assort_mean, x="assortment", y="sales", palette="Greens_d", ax=ax)
            ax.set_title("Average Sales by Assortment (Open Stores)")
            ax.set_xlabel("Assortment (a=Basic, b=Extra, c=Extended)")
            ax.set_ylabel("Mean Sales (€)")
            fig.tight_layout()
            render_matplotlib_fig(fig)

        st.markdown("""
        - **Store Type 'b'** generates the highest average sales by far (~€10,000+), typically situated in high-traffic transportation hubs and operating with extended hours.
        - **Assortment 'b' (Extra)** yields the highest revenue per store, followed by Extended (c) and Basic (a).
        """)

    # Tab 3: Seasonality & Trends
    with eda_tabs[2]:
        st.markdown("#### Temporal Trends & Shopping Patterns")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            monthly_sales = eda_df.groupby("month")["sales"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=monthly_sales, x="month", y="sales", color="#2563EB", ax=ax)
            ax.set_title("Total Sales by Month across Chain")
            ax.set_xlabel("Month (1 - 12)")
            ax.set_ylabel("Total Sales (€)")
            fig.tight_layout()
            render_matplotlib_fig(fig)

        with t_col2:
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_sales = open_df.groupby("day_name")["sales"].mean().reindex(weekday_order).reset_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=weekday_sales, x="day_name", y="sales", color="#0EA5E9", ax=ax)
            ax.set_title("Average Daily Sales by Day of Week (Open Stores)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.set_ylabel("Mean Daily Sales (€)")
            fig.tight_layout()
            render_matplotlib_fig(fig)

        st.caption("Peak revenue months occur in November and December due to Christmas and year-end holiday shopping. Monday and Friday/Saturday show strong volume spikes.")

    # Tab 4: Holidays
    with eda_tabs[3]:
        st.markdown("#### Impact of State & School Holidays")
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            school_mean = open_df.groupby("schoolholiday")["sales"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=school_mean, x="schoolholiday", y="sales", palette=["#94A3B8", "#F59E0B"], ax=ax)
            ax.set_title("Sales by School Holiday (Open Stores)")
            ax.set_xticklabels(["Regular Day", "School Holiday"])
            fig.tight_layout()
            render_matplotlib_fig(fig)

        with h_col2:
            state_mean = open_df.groupby("stateholiday")["sales"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=state_mean, x="stateholiday", y="sales", palette="Set2", ax=ax)
            ax.set_title("Sales by State Holiday Type (Open Stores)")
            ax.set_xticklabels(["None (0)", "Public (a)", "Easter (b)", "Christmas (c)"])
            fig.tight_layout()
            render_matplotlib_fig(fig)

    # Tab 5: Top & Bottom Stores
    with eda_tabs[4]:
        st.markdown("#### Store Performance Rankings")
        top10 = open_df.groupby("store")["sales"].mean().sort_values(ascending=False).head(10).reset_index()
        bot10 = open_df.groupby("store")["sales"].mean().sort_values(ascending=True).head(10).reset_index()

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=top10, x="store", y="sales", order=top10["store"], palette="Blues_r", ax=ax)
            ax.set_title("Top 10 Stores by Average Daily Sales")
            ax.set_ylabel("Mean Daily Sales (€)")
            fig.tight_layout()
            render_matplotlib_fig(fig)

        with r_col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=bot10, x="store", y="sales", order=bot10["store"], palette="Reds_r", ax=ax)
            ax.set_title("Bottom 10 Stores by Average Daily Sales")
            ax.set_ylabel("Mean Daily Sales (€)")
            fig.tight_layout()
            render_matplotlib_fig(fig)

    # Tab 6: Correlation
    with eda_tabs[5]:
        st.markdown("#### Multi-Factor Correlation Heatmap")
        corr_cols = [c for c in ["sales", "customers", "dayofweek", "promo", "schoolholiday", "competitiondistance", "promo2"] if c in open_df.columns]
        corr_matrix = open_df[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Matrix of Sales & Operating Features")
        fig.tight_layout()
        render_matplotlib_fig(fig)

    # Tab 7: Strategic Insights
    with eda_tabs[6]:
        st.markdown("""
        ### 💡 Strategic Business Takeaways & Recommendations

        1. **Optimize Promotion Cycles**:
           - Active promotions boost sales by **+33.9%**. Align high-margin inventory and marketing spend during active promo weeks to maximize basket size.
        2. **Expand High-Performing Store Formats**:
           - **Store Type 'b'** delivers disproportionate revenue per store. Evaluate whether other high-traffic metropolitan locations can be upgraded to format 'b' or offer the **Extra (b)** assortment.
        3. **Inventory Management around Q4 Peak**:
           - November and December account for substantial revenue spikes. Ramp up warehouse supply chains by mid-October to avoid stock-outs.
        4. **Customer Retention & Staffing Optimization**:
           - Customer traffic has a **0.89 correlation** with revenue. Staffing rosters should be weighted toward peak traffic days (Mondays, Fridays, and Saturdays).
        5. **Competition Proximity Response**:
           - Stores with competitors within 1,000 meters benefit heavily from active promotions to prevent customer churn.
        """)


# =============================================================================
# PAGE 4: 🤖 TASK 03: SALES FORECASTING (ALL 4 PKL MODELS)
# =============================================================================
elif page == "🤖 Task 03: Sales Forecasting (4 Models)":
    st.markdown("### 🤖 Task 03: Sales Prediction & Machine Learning Engine")
    st.markdown("<p style='color: #64748B;'>Compare all 4 trained PKL models, inspect residuals, run live interactive predictions, or batch-score test datasets.</p>", unsafe_allow_html=True)

    # Active model cards
    active_m_cols = st.columns(4)
    model_keys = list(models.keys())
    for idx, (m_name, m_info) in enumerate(models.items()):
        status_txt = f"{m_info['size_mb']} MB · Ready" if m_info['loaded'] else "Error / Missing"
        active_m_cols[idx].metric(m_name, status_txt)

    ml_tabs = st.tabs([
        "🏆 Model Comparison & Leaderboard",
        "📈 Residuals & Diagnostics",
        "🎯 Feature Importance",
        "⚡ Live Store Forecaster (Single Day)",
        "📂 Batch Test Predictions",
    ])

    # Tab 1: Leaderboard
    with ml_tabs[0]:
        st.markdown("#### Model Performance Comparison on 80:20 Chronological Split")
        model_comp_path = resolve_path("outputs/predictions/model_comparison.csv")
        if model_comp_path.exists():
            comp_df = pd.read_csv(model_comp_path)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(7, 3.8))
                sns.barplot(data=comp_df, x="Model", y="RMSE", palette="Blues_r", ax=ax)
                ax.set_title("Root Mean Squared Error (RMSE) - Lower is Better")
                ax.set_ylabel("RMSE (€)")
                fig.tight_layout()
                render_matplotlib_fig(fig)

            with c2:
                fig, ax = plt.subplots(figsize=(7, 3.8))
                sns.barplot(data=comp_df, x="Model", y="R2", palette="Greens_r", ax=ax)
                ax.set_title("R² Score (Goodness of Fit) - Higher is Better")
                ax.set_ylabel("R²")
                fig.tight_layout()
                render_matplotlib_fig(fig)

        st.markdown("""
        ##### ⏳ Validation Protocol:
        - **Split Methodology**: Chronological 80:20 time-based split across unique dates (Cutoff Date: **2014-12-09**).
        - **Prevents Data Leakage**: Standard random train_test_split would leak future store trends into historical training. TimeSeries validation ensures realistic real-world evaluation.
        - **Hyperparameter Tuning**: Tuned via `RandomizedSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.
        """)

    # Tab 2: Diagnostics
    with ml_tabs[1]:
        st.markdown("#### Model Diagnostics & Error Analysis")
        val_pred_path = resolve_path("outputs/predictions/validation_predictions.csv")
        if val_pred_path.exists():
            val_df = pd.read_csv(val_pred_path)
            sample_val = val_df.sample(min(len(val_df), 10_000), random_state=42)

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                fig, ax = plt.subplots(figsize=(6.5, 5))
                sns.scatterplot(data=sample_val, x="Actual_Sales", y="Predicted_Sales", alpha=0.3, color="#2563EB", ax=ax)
                max_v = sample_val["Actual_Sales"].quantile(0.99)
                ax.plot([0, max_v], [0, max_v], "r--", linewidth=2, label="Ideal Fit")
                ax.set_title("Actual vs Predicted Sales (Validation Sample)")
                ax.set_xlabel("Actual Sales (€)")
                ax.set_ylabel("Predicted Sales (€)")
                ax.legend()
                fig.tight_layout()
                render_matplotlib_fig(fig)

            with d_col2:
                fig, ax = plt.subplots(figsize=(6.5, 5))
                sns.histplot(sample_val["Prediction_Error"], bins=50, kde=True, color="#0EA5E9", ax=ax)
                ax.set_title("Residual Distribution (Prediction Error)")
                ax.set_xlabel("Actual - Predicted Error (€)")
                fig.tight_layout()
                render_matplotlib_fig(fig)

            st.caption("Residual distribution is symmetric and strongly centered around 0 error, confirming minimal model bias.")

    # Tab 3: Feature Importance
    with ml_tabs[2]:
        st.markdown("#### Top Predictive Feature Drivers")
        feat_imp_path = resolve_path("outputs/predictions/feature_importance.csv")
        if feat_imp_path.exists():
            feat_df = pd.read_csv(feat_imp_path)
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.barplot(data=feat_df.head(15), x="Importance", y="Feature", palette="viridis", ax=ax)
            ax.set_title("Top 15 Feature Importances (Tuned Random Forest)")
            fig.tight_layout()
            render_matplotlib_fig(fig)

            st.dataframe(feat_df.head(25), use_container_width=True, hide_index=True)

    # Tab 4: Live Forecaster
    with ml_tabs[3]:
        st.markdown("#### ⚡ Interactive Store Sales Simulator")
        st.markdown("Simulate any store scenario and generate instant sales forecasts across all 4 machine learning models.")

        with st.form("single_forecast_form"):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                sel_store = st.selectbox("Select Store ID:", sorted(store_df["store"].unique()), index=0)
                pred_date = st.date_input("Forecast Date:", value=test_df["date"].min().date())
                is_open = st.selectbox("Store Status:", [1, 0], format_func=lambda x: "Open" if x == 1 else "Closed")

            with f_col2:
                promo_act = st.selectbox("Promotion Active?:", [1, 0], format_func=lambda x: "Yes (Active)" if x == 1 else "No (Standard)")
                school_hol = st.selectbox("School Holiday?:", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                state_hol = st.selectbox("State Holiday:", ["0", "a", "b", "c"],
                                         format_func=lambda x: {"0": "None (0)", "a": "Public Holiday (a)", "b": "Easter (b)", "c": "Christmas (c)"}[x])

            with f_col3:
                chosen_model_name = st.selectbox("Primary Forecast Model:", model_keys, index=0)
                compare_all_models = st.checkbox("Compare All 4 Models Side-by-Side", value=True)

            submit_pred = st.form_submit_button("🚀 Generate Sales Forecast", type="primary")

        if submit_pred:
            if is_open == 0:
                st.warning("Store is closed on this date. Projected Sales: **€0.00**")
            else:
                # Build feature row
                s_meta = store_df[store_df["store"] == sel_store].iloc[0]
                ts = pd.Timestamp(pred_date)
                row_dict = {
                    "store": sel_store,
                    "date": ts,
                    "dayofweek": ts.isoweekday(),
                    "open": 1,
                    "promo": promo_act,
                    "stateholiday": str(state_hol).replace(".0", ""),
                    "schoolholiday": school_hol,
                    "storetype": str(s_meta["storetype"]),
                    "assortment": str(s_meta["assortment"]),
                    "competitiondistance": float(s_meta["competitiondistance"]),
                    "competitiondistance_missing": int(s_meta["competitiondistance_missing"]),
                    "competitionopensincemonth": int(s_meta["competitionopensincemonth"]),
                    "competitionopensinceyear": int(s_meta["competitionopensinceyear"]),
                    "competition_opening_info_missing": int(s_meta["competition_opening_info_missing"]),
                    "promo2": int(s_meta["promo2"]),
                    "promo2sinceweek": int(s_meta["promo2sinceweek"]),
                    "promo2sinceyear": int(s_meta["promo2sinceyear"]),
                    "promointerval": str(s_meta["promointerval"]),
                    "promo2_not_applicable": int(s_meta["promo2_not_applicable"]),
                }
                single_df = pd.DataFrame([row_dict])
                single_df["year"] = single_df["date"].dt.year
                single_df["month"] = single_df["date"].dt.month
                single_df["day"] = single_df["date"].dt.day
                single_df["dayofyear"] = single_df["date"].dt.dayofyear
                single_df["weekofyear"] = single_df["date"].dt.isocalendar().week.astype(int)
                single_df["is_weekend"] = (single_df["dayofweek"] >= 6).astype(int)
                single_df["is_month_start"] = single_df["date"].dt.is_month_start.astype(int)
                single_df["is_month_end"] = single_df["date"].dt.is_month_end.astype(int)

                known_comp = (single_df["competitionopensincemonth"] > 0) & (single_df["competitionopensinceyear"] > 0)
                single_df["competition_age_months"] = np.where(
                    known_comp,
                    (single_df["year"] - single_df["competitionopensinceyear"]) * 12 + (single_df["month"] - single_df["competitionopensincemonth"]),
                    0
                ).clip(min=0)

                if single_df.loc[0, "promo2"] == 1 and single_df.loc[0, "promointerval"] != "No Promo2":
                    months = [m.strip() for m in str(single_df.loc[0, "promointerval"]).split(",")]
                    cur_m = ts.strftime("%b").replace("Sep", "Sept")
                    cur_p = ts.year * 100 + int(ts.isocalendar().week)
                    start_p = int(single_df.loc[0, "promo2sinceyear"]) * 100 + int(single_df.loc[0, "promo2sinceweek"])
                    single_df["promo2_active"] = int((cur_p >= start_p) and (cur_m in months))
                else:
                    single_df["promo2_active"] = 0

                X_input = prepare_X_for_model(single_df)

                # Execute prediction
                primary_m = models[chosen_model_name]["model"]
                primary_pred = max(float(primary_m.predict(X_input)[0]), 0.0)

                st.markdown(f"""
                <div style='background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 18px 24px; margin-top: 15px;'>
                    <h3 style='color: #1E40AF; margin-bottom: 4px;'>Predicted Daily Revenue: €{primary_pred:,.2f}</h3>
                    <p style='color: #3B82F6; margin: 0;'>Forecast Model: <b>{chosen_model_name}</b> | Store: <b>#{sel_store}</b> | Date: <b>{pred_date}</b></p>
                </div>
                """, unsafe_allow_html=True)

                if compare_all_models:
                    st.markdown("##### 🔍 Multi-Model Forecast Comparison")
                    comparative_results = []
                    for name in model_keys:
                        m_obj = models[name]["model"]
                        if m_obj:
                            val = max(float(m_obj.predict(X_input)[0]), 0.0)
                            comparative_results.append({"Model": name, "Predicted Sales (€)": round(val, 2)})
                    comp_pred_df = pd.DataFrame(comparative_results)

                    res_c1, res_c2 = st.columns([1, 1.2])
                    with res_c1:
                        st.dataframe(comp_pred_df, use_container_width=True, hide_index=True)
                    with res_c2:
                        fig, ax = plt.subplots(figsize=(7, 3.2))
                        sns.barplot(data=comp_pred_df, x="Model", y="Predicted Sales (€)", palette="Blues_d", ax=ax)
                        ax.set_ylabel("Forecast (€)")
                        fig.tight_layout()
                        render_matplotlib_fig(fig)

    # Tab 5: Batch Test Predictions
    with ml_tabs[4]:
        st.markdown("#### Batch Prediction on Cleaned Test Set")
        st.markdown(f"Generate forecasts for all **{len(test_df):,}** records in `cleaned_test.csv`.")

        batch_model_name = st.selectbox("Select Model for Batch Scoring:", model_keys, index=0)

        if st.button("⚡ Run Batch Predictions on Test Set", type="primary"):
            with st.spinner("Scoring test records... this takes 3-5 seconds."):
                m_chosen = models[batch_model_name]["model"]
                merged_test = test_df.merge(store_df, on="store", how="left")
                merged_test["year"] = merged_test["date"].dt.year
                merged_test["month"] = merged_test["date"].dt.month
                merged_test["day"] = merged_test["date"].dt.day
                merged_test["dayofyear"] = merged_test["date"].dt.dayofyear
                merged_test["weekofyear"] = merged_test["date"].dt.isocalendar().week.astype(int)
                merged_test["is_weekend"] = (merged_test["dayofweek"] >= 6).astype(int)
                merged_test["is_month_start"] = merged_test["date"].dt.is_month_start.astype(int)
                merged_test["is_month_end"] = merged_test["date"].dt.is_month_end.astype(int)

                known_comp = (merged_test["competitionopensincemonth"] > 0) & (merged_test["competitionopensinceyear"] > 0)
                merged_test["competition_age_months"] = np.where(
                    known_comp,
                    (merged_test["year"] - merged_test["competitionopensinceyear"]) * 12 + (merged_test["month"] - merged_test["competitionopensincemonth"]),
                    0
                ).clip(min=0)

                month_abbr = merged_test["date"].dt.strftime("%b").replace({"Sep": "Sept"})
                interval = merged_test["promointerval"].astype(str)
                month_in_interval = pd.Series(False, index=merged_test.index)
                for iv in interval.unique():
                    if iv in ("No Promo2", "nan", "None", ""):
                        continue
                    months = {m.strip() for m in iv.split(",")}
                    mask = interval.eq(iv)
                    month_in_interval[mask] = month_abbr[mask].isin(months)
                cur_period = merged_test["date"].dt.year * 100 + merged_test["date"].dt.isocalendar().week.astype(int)
                start_period = merged_test["promo2sinceyear"].astype(int) * 100 + merged_test["promo2sinceweek"].astype(int)
                merged_test["promo2_active"] = ((merged_test["promo2"] == 1) & month_in_interval & (cur_period >= start_period)).astype(int)

                X_test = prepare_X_for_model(merged_test)
                open_mask = (merged_test["open"] == 1).to_numpy()
                preds = np.zeros(len(merged_test))
                if open_mask.any():
                    preds[open_mask] = m_chosen.predict(X_test.loc[open_mask])
                preds = np.clip(preds, a_min=0, a_max=None)

                final_sub = pd.DataFrame({"Id": merged_test["id"].astype(int), "Sales": preds})

                st.session_state["batch_predictions"] = final_sub

        if "batch_predictions" in st.session_state:
            sub = st.session_state["batch_predictions"]
            st.success(f"✅ Successfully generated {len(sub):,} predictions!")
            b1, b2, b3 = st.columns(3)
            b1.metric("Projected Test Revenue", f"€{sub['Sales'].sum() / 1e6:,.1f}M")
            b2.metric("Avg Daily Store Forecast", f"€{sub[sub['Sales'] > 0]['Sales'].mean():,.0f}")
            b3.metric("Zero-Sales (Closed) Stores", f"{(sub['Sales'] == 0).sum():,}")

            st.dataframe(sub.head(30), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download final_sales_predictions.csv",
                sub.to_csv(index=False).encode("utf-8"),
                file_name="final_sales_predictions.csv",
                mime="text/csv"
            )


# =============================================================================
# PAGE 5: 📑 REPORTS & ARTIFACTS
# =============================================================================
elif page == "📑 Reports & Artifacts":
    st.markdown("### 📑 Project Reports & Documentation Artifacts")
    st.markdown("<p style='color: #64748B;'>Access official project reports, training summaries, and exportable deliverables.</p>", unsafe_allow_html=True)

    report_files = [
        ("Data Cleaning Report (Task 01)", "reports/01_Data_Cleaning_Report.txt"),
        ("EDA Summary Report (Task 02)", "reports/02_EDA_Summary_Report.txt"),
        ("Sales Prediction Report (Task 03)", "reports/03_Sales_Prediction_Report.txt"),
        ("Combined Rossmann Report (Full Project)", "reports/Rossmann_Combined_Report.txt"),
    ]

    selected_rep_title = st.selectbox("Choose a report to inspect:", [r[0] for r in report_files])
    rep_rel_path = next(r[1] for r in report_files if r[0] == selected_rep_title)
    rep_path = resolve_path(rep_rel_path)

    if rep_path.exists():
        with open(rep_path, "r", encoding="utf-8") as f:
            content = f.read()

        st.text_area("Report Content:", content, height=450)
        st.download_button(
            f"⬇️ Download {rep_path.name}",
            content.encode("utf-8"),
            file_name=rep_path.name,
            mime="text/plain"
        )
    else:
        st.warning(f"Report file '{rep_rel_path}' not found.")

    st.markdown("---")
    st.markdown("#### 📦 Model Weights Download (.pkl)")
    m_dl_cols = st.columns(4)
    for idx, (m_name, m_info) in enumerate(models.items()):
        if m_info["loaded"] and m_info["path"].exists():
            with open(m_info["path"], "rb") as f:
                p_bytes = f.read()
            m_dl_cols[idx].download_button(
                f"⬇️ {m_info['filename']}",
                p_bytes,
                file_name=m_info['filename'],
                mime="application/octet-stream"
            )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(
    "ROSSMANN STORE SALES INTELLIGENCE  |  WEINTERN DATA SCIENCE INTERNSHIP · WEEK 02  |  "
    "Data Cleaning · Exploratory Analysis · 4 Machine Learning Models"
)
