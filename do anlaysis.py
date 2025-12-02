pip install streamlit pandas altair

app.py
Budget 2014-2025.csv
streamlit run app.py
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Budget 2014–2025 Analyzer", layout="wide")

st.title("📊 Budget Data Analyzer (2014–2025)")
st.write("Fully automated analysis for your CSV file.")

# Load your CSV safely
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Budget 2014-2025.csv", dtype=str)
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

df = load_data()

st.success("✅ CSV Loaded Successfully!")

# Show preview
st.subheader("📄 Data Preview (first 100 rows)")
st.dataframe(df.head(100))

# Clean numeric columns
def clean_numeric(col):
    col = col.astype(str).str.replace(",", "").str.replace(" ", "").str.replace("-", "")
    return pd.to_numeric(col, errors="coerce")

clean_df = df.copy()

for c in clean_df.columns:
    clean_df[c] = clean_numeric(clean_df[c])

st.subheader("📘 Dataset Summary (Auto Cleaned)")
st.write(clean_df.describe(include="all"))

# Detect numeric columns
numeric_cols = clean_df.select_dtypes(include=["float64", "int64"]).columns.tolist()

if len(numeric_cols) == 0:
    st.warning("⚠️ No numeric columns found after cleaning.")
else:
    st.subheader("📈 Visualize Numeric Columns")
    col = st.selectbox("Choose a numeric column to plot", numeric_cols)

    chart = (
        alt.Chart(clean_df.reset_index())
        .mark_line(point=True)
        .encode(
            x="index:O",
            y=alt.Y(col, title=col),
            tooltip=[col]
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# Year columns detection
year_cols = [c for c in df.columns if str(c).isdigit() and 2010 <= int(c) <= 2030]

if len(year_cols) >= 2:
    st.subheader("📅 Year-wise Data Detected")
    st.write("Detected columns:", year_cols)

    long_df = df.melt(id_vars=[c for c in df.columns if c not in year_cols],
                      value_vars=year_cols,
                      var_name="Year",
                      value_name="Value")

    # Clean Value column
    long_df["Value"] = clean_numeric(long_df["Value"])

    st.dataframe(long_df.head(100))

    # Plot year-wise
    st.subheader("📈 Year-wise Trend")
    chart2 = (
        alt.Chart(long_df)
        .mark_line(point=True)
        .encode(
            x="Year:O",
            y="Value:Q"
        )
        .interactive()
    )
    st.altair_chart(chart2, use_container_width=True)

# Download cleaned CSV
st.subheader("⬇️ Download Cleaned CSV")
st.download_button(
    label="Download cleaned_budget.csv",
    data=clean_df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_budget.csv",
    mime="text/csv"
)
