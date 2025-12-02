# streamlit_app.py
pip install pandas streamlit altair



import streamlit as st
import pandas as pd
import altair as alt
import io
import re

st.set_page_config(page_title="Budget Explorer (2014-2025)", layout="wide")

@st.cache_data
def read_csv_bytes(uploaded_bytes):
    # read bytes into DataFrame (handles small/medium files)
    return pd.read_csv(io.BytesIO(uploaded_bytes), dtype=str)

def clean_numeric_series(s: pd.Series) -> pd.Series:
    # Convert common formatted numeric strings to float (commas, parentheses, %)
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        x = s.astype(str).str.strip()
        x = x.replace({'': None, 'nan': None, 'NaN': None, 'None': None})
        # parentheses -> negative
        x = x.str.replace(r'^\((.*)\)$', r'-\1', regex=True)
        # remove commas and spaces
        x = x.str.replace(r'[,\s\u00A0]', '', regex=True)
        # remove non-numeric trailing characters like % then scale
        pct = x.str.endswith('%')
        # remove % and other non numeric chars except - and .
        x = x.str.replace(r'[^\d\.\-]', '', regex=True)
        num = pd.to_numeric(x, errors='coerce')
        # apply percent scaling
        if pct.any():
            num.loc[pct] = num.loc[pct] / 100.0
        return num
    else:
        return pd.to_numeric(s, errors='coerce')

def auto_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        # heuristic: if column is object and contains digits in many rows, try cleaning
        if df2[col].dtype == object or pd.api.types.is_string_dtype(df2[col]):
            sample = df2[col].dropna().astype(str).head(50).tolist()
            if any(re.search(r'\d', str(val)) for val in sample):
                df2[col] = clean_numeric_series(df2[col])
    return df2

def detect_year_columns(df: pd.DataFrame):
    years = []
    for c in df.columns:
        try:
            ci = int(str(c).strip())
            if 1900 < ci < 2100:
                years.append(c)
        except Exception:
            continue
    return sorted(years, key=lambda x: int(str(x)))

def wide_to_long_if_possible(df: pd.DataFrame):
    years = detect_year_columns(df)
    if len(years) >= 2:
        id_vars = [c for c in df.columns if c not in years]
        long = df.melt(id_vars=id_vars, value_vars=years, var_name="Year", value_name="Value")
        # clean Year and Value
        long['Year'] = pd.to_numeric(long['Year'], errors='coerce').astype('Int64')
        long['Value'] = clean_numeric_series(long['Value'])
        return long
    else:
        return None

def main():
    st.title("Budget Explorer (2014-2025)")
    st.markdown("Upload a CSV or use the sample CSV bundled with the repo. The app will try to auto-clean numeric columns and detect year-like columns.")

    col1, col2 = st.columns([3,1])
    with col2:
        st.markdown("**Options**")
        use_sample = st.checkbox("Use uploaded CSV from repo (`Budget 2014-2025.csv`)", value=True)
        upload = st.file_uploader("Or upload a CSV", type=["csv"])

    # Load data
    df = None
    if upload is not None:
        try:
            df = read_csv_bytes(upload.read())
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            st.stop()
    else:
        if use_sample:
            try:
                # streamlit runs from project folder; expect file there
                df = pd.read_csv("Budget 2014-2025.csv", dtype=str)
            except FileNotFoundError:
                st.warning("Bundled CSV not found in the app folder. Please upload a CSV or place `Budget 2014-2025.csv` alongside this app.")
            except Exception as e:
                st.error(f"Error reading bundled CSV: {e}")

    if df is None:
        st.stop()

    st.sidebar.header("Data preview & controls")
    st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    if st.sidebar.button("Show raw preview"):
        st.write(df.head(50))

    # Attempt intelligent cleaning
    st.info("Automatically attempting to parse/clean numeric columns (commas, parentheses, %).")
    cleaned = auto_clean_dataframe(df)

    # Detect year-like columns and optionally reshape
    long = wide_to_long_if_possible(cleaned)

    view = st.sidebar.selectbox("Main view", ["Table", "Long-form (if detected)", "Analyze & Plot"], index=2)

    if view == "Table":
        st.subheader("Cleaned table (first 500 rows)")
        st.dataframe(cleaned.head(500))
        if st.button("Download cleaned CSV"):
            st.download_button("Download cleaned CSV", data=cleaned.to_csv(index=False).encode('utf-8'), file_name="budget_cleaned.csv", mime='text/csv')
    elif view == "Long-form (if detected)":
        if long is not None:
            st.subheader("Long-form (Year, Value) view")
            st.dataframe(long.head(500))
            st.markdown("You can group and plot the `Value` over `Year` (aggregations available).")
            st.download_button("Download long-form CSV", data=long.to_csv(index=False).encode('utf-8'), file_name="budget_long.csv", mime='text/csv')
        else:
            st.warning("No obvious year columns detected to melt into long form. See the table view.")
    else:
        st.subheader("Analyze & Plot")
        numeric_cols = [c for c in cleaned.columns if pd.api.types.is_numeric_dtype(cleaned[c])]
        st.write(f"Detected numeric columns: `{numeric_cols}`")
        if not numeric_cols:
            st.warning("No numeric columns detected after cleaning. Try the raw preview to inspect columns and formats.")
        else:
            col = st.selectbox("Choose numeric column", numeric_cols, index=0)
            agg = st.selectbox("Aggregation for summary / time series", ["sum", "mean", "median", "max", "min"])
            if long is not None and "Year" in long.columns and "Value" in long.columns:
                st.markdown("Using detected Year/Value long-form for time series charts.")
                id_candidates = [c for c in long.columns if c not in ["Year", "Value"]]
                group_choice = st.selectbox("Group by (optional)", ["(none)"] + id_candidates)
                if group_choice == "(none)":
                    series = (long.groupby("Year")["Value"]
                              .agg(agg)
                              .reset_index()
                              .dropna(subset=["Year"]))
                    chart = alt.Chart(series).mark_line(point=True).encode(
                        x=alt.X("Year:O", title="Year"),
                        y=alt.Y("Value:Q", title=f"Value ({agg})")
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
                    st.dataframe(series)
                else:
                    # allow selecting multiple groups
                    available = sorted(long[group_choice].dropna().unique(), key=lambda x: str(x))
                    chosen = st.multiselect(f"Select {group_choice} values (empty=all)", options=available, default=available[:5])
                    df_plot = long.copy()
                    if chosen:
                        df_plot = df_plot[df_plot[group_choice].isin(chosen)]
                    series = (df_plot.groupby(["Year", group_choice])["Value"]
                              .agg(agg)
                              .reset_index()
                              .dropna(subset=["Year"]))
                    chart = alt.Chart(series).mark_line(point=True).encode(
                        x=alt.X("Year:O", title="Year"),
                        y=alt.Y("Value:Q", title=f"Value ({agg})"),
                        color=alt.Color(f"{group_choice}:N", title=group_choice)
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
                    st.dataframe(series.head(500))
            else:
                # No year-based time series available; show distribution plots
                st.write(cleaned[col].describe())
                hist = alt.Chart(cleaned).mark_bar().encode(
                    alt.X(f"{col}:Q", bin=alt.Bin(maxbins=60)),
                    y='count()'
                ).interactive()
                st.altair_chart(hist, use_container_width=True)
                box = alt.Chart(cleaned).mark_boxplot().encode(
                    y=alt.Y(f"{col}:Q")
                )
                st.altair_chart(box, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Export:")
    st.sidebar.download_button("Download original CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="budget_original.csv", mime='text/csv')
    st.sidebar.download_button("Download cleaned CSV", data=cleaned.to_csv(index=False).encode('utf-8'), file_name="budget_cleaned.csv", mime='text/csv')

if __name__ == "__main__":
    main()
