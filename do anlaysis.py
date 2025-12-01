import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="CSV Data Analyzer", layout="wide")

st.title("📊 Simple CSV Data Analyzer")

# --------- FILE UPLOADER ----------
st.sidebar.header("Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("CSV Loaded Successfully!")
    st.write("### 🔍 Preview of Data")
    st.dataframe(df, use_container_width=True)

    # --------- BASIC INFO ----------
    st.write("### 📌 Basic Information")
    st.write(df.describe(include="all"))

    # --------- CHARTS ----------
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    if len(numeric_columns) > 0:
        st.write("### 📈 Line Chart")
        column = st.selectbox("Select column for line chart", numeric_columns)
        fig = px.line(df, y=column, title=f"Line Chart of {column}")
        st.plotly_chart(fig, use_container_width=True)

        st.write("### 📊 Bar Chart")
        column = st.selectbox("Select column for bar chart", numeric_columns, key="bar")
        fig = px.bar(df, y=column, title=f"Bar Chart of {column}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns found for charting.")

else:
    st.info("Please upload a CSV file to get started.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
