# streamlit_app.py
"""
Budget Explorer — Streamlit app
Features:
 - Upload CSV or use bundled 'Budget 2014-2025.csv'
 - Automatic numeric cleaning (commas, parentheses, percent, dashes)
 - Detects year columns (2010-2030) and melts wide->long
 - Summary metrics: totals, YoY, CAGR, top categories
 - Interactive charts: time series, stacked area, treemap, bar, correlation heatmap
 - Data table with search & download cleaned CSV / long CSV
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import io
import re
from typing import List, Optional

st.set_page_config(page_title="Budget Explorer (2014–2025)", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Helpers: cleaning & transforms
# ---------------------------
@st.cache_data
def read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b), dtype=str)

def safe_read_path(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        return None

def clean_string_number_series(s: pd.Series) -> pd.Series:
    """
    Convert formatted strings into numeric floats:
    - (1,234) -> -1234
    - "1,234" -> 1234
    - "12.3%" -> 0.123
    - empties -> NaN
    """
    x = s.astype(str).fillna("").str.strip()
    # treat blanks
    x = x.replace({"": None, "nan": None, "NaN": None, "None": None})
    # parentheses negative
    x = x.astype(str).str.replace(r'^\((.*)\)$', r'-\1', regex=True)
    # remove non-breaking spaces, normal spaces, commas
    x = x.str.replace(r'[,\s\u00A0]', '', regex=True)
    # percent
    is_pct = x.str.endswith('%')
    x = x.str.replace('%','', regex=False)
    # remove currency symbols or stray characters
    x = x.str.replace(r'[^\d\.\-]', '', regex=True)
    out = pd.to_numeric(x, errors='coerce')
    if is_pct.any():
        out[is_pct.fillna(False)] = out[is_pct.fillna(False)].div(100)
    return out

def auto_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        # attempt conversion when column contains digits in sample
        sample = df2[c].dropna().astype(str).head(50).tolist()
        if any(re.search(r'\d', str(v)) for v in sample):
            try:
                numeric = clean_string_number_series(df2[c])
                # only replace if conversion produced many numbers (heuristic)
                if numeric.notna().sum() >= max(1, int(0.15 * len(df2))):
                    df2[c] = numeric
            except Exception:
                pass
    return df2

def detect_year_columns(cols: List[str]) -> List[str]:
    years = []
    for c in cols:
        cs = str(c).strip()
        if cs.isdigit():
            y = int(cs)
            if 2010 <= y <= 2030:
                years.append(c)
    # sort numerically
    years_sorted = sorted(years, key=lambda x: int(str(x)))
    return years_sorted

def melt_to_long_if_years(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    years = detect_year_columns(df.columns.tolist())
    if len(years) >= 2:
        id_vars = [c for c in df.columns if c not in years]
        long = df.melt(id_vars=id_vars, value_vars=years, var_name="Year", value_name="Value")
        # clean value and year
        long['Value'] = clean_string_number_series(long['Value'])
        long['Year'] = pd.to_numeric(long['Year'], errors='coerce').astype('Int64')
        return long
    return None

# ---------------------------
# Load data (upload or bundled)
# ---------------------------
st.title("📊 Budget Explorer (2014–2025)")
st.markdown("Upload your CSV or use the bundled `Budget 2014-2025.csv`. The app auto-cleans numbers and builds interactive visualizations.")

col1, col2 = st.columns([3,1])
with col2:
    st.sidebar.header("Data source")
    use_bundled = st.sidebar.checkbox("Use bundled `Budget 2014-2025.csv` (repo)", value=True)
    uploaded = st.sidebar.file_uploader("Or upload CSV", type=["csv"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("App options")
    sample_rows = st.sidebar.slider("Preview rows", 5, 500, 100, step=5)
    enable_demo_charts = st.sidebar.checkbox("Show extra demo charts", value=True)

df_raw = None
if uploaded is not None:
    try:
        df_raw = read_csv_bytes(uploaded.read())
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()
else:
    if use_bundled:
        df_raw = safe_read_path("Budget 2014-2025.csv")
        if df_raw is None:
            st.warning("Bundled CSV not found. Please upload the CSV or put `Budget 2014-2025.csv` in the folder.")
            if uploaded is None:
                st.stop()
    else:
        st.info("Upload a CSV file to begin.")
        st.stop()

# Show raw preview and schema
st.subheader("Data preview & schema")
st.markdown(f"Rows: **{df_raw.shape[0]}** — Columns: **{df_raw.shape[1]}**")
st.dataframe(df_raw.head(sample_rows))

with st.expander("Column names and sample values"):
    info = []
    for c in df_raw.columns:
        sample_vals = ", ".join(map(str, df_raw[c].dropna().astype(str).head(3).tolist()))
        info.append(f"**{c}** — sample: {sample_vals}")
    st.write("\n\n".join(info))

# ---------------------------
# Clean data
# ---------------------------
st.subheader("Cleaning & type inference")
st.markdown("Automatically converting formatted numeric columns to numeric types (commas, parentheses, percent, spaces).")

df_clean = auto_clean_dataframe(df_raw)
st.write("Preview (cleaned):")
st.dataframe(df_clean.head(sample_rows))

# Detect year columns and produce long form
years = detect_year_columns(df_clean.columns.tolist())
long_df = melt_to_long_if_years(df_raw)  # use raw for melting so we don't lose ids

# ---------------------------
# Analysis panel (main)
# ---------------------------
st.header("Analysis")

# Column selectors and filters
left, right = st.columns([2,1])
with right:
    st.markdown("### Filters")
    # auto find potential category columns: object type and low cardinality
    cat_candidates = [c for c in df_clean.columns if df_clean[c].dtype == object or df_clean[c].nunique() < (0.5 * len(df_clean))]
    # remove year columns from category candidates
    cat_candidates = [c for c in cat_candidates if c not in years and not str(c).isdigit()]
    chosen_cat = st.selectbox("Category column (optional)", options=["(none)"] + cat_candidates)
    if chosen_cat and chosen_cat != "(none)":
        cat_values = sorted(df_clean[chosen_cat].dropna().astype(str).unique().tolist())
        chosen_values = st.multiselect(f"Select {chosen_cat} values (empty = all)", options=cat_values, default=cat_values[:5])
    else:
        chosen_values = None

with left:
    st.markdown("### Quick metrics")
    # compute overall numeric totals for top numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        # choose a default numeric for metrics: largest sum
        sums = df_clean[numeric_cols].sum(numeric_only=True, skipna=True).sort_values(ascending=False)
        top_numeric = sums.index[0]
        total_val = sums.iloc[0]
        st.metric(label=f"Top numeric: {top_numeric} (sum)", value=f"{total_val:,.0f}")
        st.write("Other numeric columns (sum):")
        st.dataframe(sums.reset_index().rename(columns={"index":"col", 0:"sum"}).head(10))
    else:
        st.info("No numeric columns detected after cleaning.")

# ---------------------------
# Year-based analysis (if available)
# ---------------------------
if long_df is not None:
    st.subheader("Time-series (year-based analysis)")
    st.markdown(f"Detected year columns: **{years}**")
    # apply filters
    working_long = long_df.copy()
    if chosen_cat and chosen_cat != "(none)" and chosen_values:
        working_long = working_long[working_long[chosen_cat].astype(str).isin(chosen_values)]
    # group by Year and optional category
    group_by = st.selectbox("Group time series by", options=["(none)"] + [c for c in working_long.columns if c not in ["Year","Value"]])
    agg_func = st.selectbox("Aggregation", options=["sum","mean","median"], index=0)
    if group_by == "(none)":
        ts = getattr(working_long.groupby("Year")["Value"], agg_func)().reset_index().dropna()
        ts = ts.sort_values("Year")
        st.markdown("### Overall time-series")
        chart = alt.Chart(ts).mark_line(point=True).encode(x=alt.X("Year:O"), y=alt.Y("Value:Q", title=f"Value ({agg_func})")).interactive()
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(ts)
    else:
        st.markdown(f"### Time series grouped by **{group_by}**")
        options = sorted(working_long[group_by].dropna().astype(str).unique().tolist())
        sel = st.multiselect(f"Select {group_by} values (empty=top 6)", options=options, default=options[:6])
        df_plot = working_long.copy()
        if sel:
            df_plot = df_plot[df_plot[group_by].astype(str).isin(sel)]
        series = getattr(df_plot.groupby(["Year", group_by])["Value"], agg_func)().reset_index().dropna().sort_values("Year")
        chart2 = alt.Chart(series).mark_line(point=True).encode(
            x=alt.X("Year:O"),
            y=alt.Y("Value:Q", title=f"Value ({agg_func})"),
            color=alt.Color(f"{group_by}:N", title=group_by)
        ).interactive()
        st.altair_chart(chart2, use_container_width=True)
        st.dataframe(series.head(500))
    # YoY and CAGR
    st.markdown("### Growth metrics")
    aggregated = working_long.groupby("Year")["Value"].sum().sort_index()
    aggregated = aggregated.dropna()
    if len(aggregated) >= 2:
        yoy = aggregated.pct_change().dropna()
        cagr = (aggregated.iloc[-1] / aggregated.iloc[0]) ** (1.0 / (aggregated.index.astype(int).max() - aggregated.index.astype(int).min())) - 1
        st.metric("Latest Year total", f"{aggregated.iloc[-1]:,.0f}")
        st.metric("Latest YoY", f"{yoy.iloc[-1]:.2%}")
        st.metric("CAGR (first->last)", f"{cagr:.2%}")
        st.line_chart(aggregated)
    else:
        st.info("Not enough year data to compute growth metrics.")

# ---------------------------
# Non-year / general analysis
# ---------------------------
st.subheader("Category / Breakdown analysis")
if chosen_cat and chosen_cat != "(none)":
    df_break = df_clean.copy()
    # choose numeric for breakdown
    numeric_cols = df_break.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available for breakdown.")
    else:
        num = st.selectbox("Choose numeric value for breakdown", numeric_cols, index=0)
        # filter values
        df_plot = df_break.copy()
        if chosen_values:
            df_plot = df_plot[df_plot[chosen_cat].astype(str).isin(chosen_values)]
        # top categories
        agg = df_plot.groupby(chosen_cat)[num].sum(numeric_only=True).sort_values(ascending=False)
        st.markdown("#### Top categories (by sum)")
        st.dataframe(agg.reset_index().rename(columns={num:"sum"}).head(20))
        # treemap
        topn = st.slider("Top N categories for treemap", 3, 50, 12)
        top_idx = agg.head(topn).reset_index()
        if not top_idx.empty:
            fig = px.treemap(top_idx, path=[chosen_cat], values=num, title=f"Top {topn} by {num}")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Choose a category column in the sidebar to see breakdowns.")

# ---------------------------
# Correlation heatmap for numerics
# ---------------------------
st.subheader("Correlation heatmap (numeric columns)")
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) >= 2:
    corr = df_clean[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix")
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Not enough numeric columns for correlation heatmap.")

# ---------------------------
# Data export & download
# ---------------------------
st.header("Export & Download")
col1, col2 = st.columns(2)
with col1:
    st.download_button("Download cleaned CSV", df_clean.to_csv(index=False).encode('utf-8'), file_name="budget_cleaned.csv", mime="text/csv")
with col2:
    if long_df is not None:
        st.download_button("Download long-form CSV", long_df.to_csv(index=False).encode('utf-8'), file_name="budget_long.csv", mime="text/csv")

st.markdown("---")
st.markdown("Built with ❤️ — modify `streamlit_app.py` to add custom charts or KPIs.")
