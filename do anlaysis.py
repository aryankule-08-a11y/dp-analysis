import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="CSV Analyzer", page_icon="📊", layout="wide")

st.title("📊 CSV Data Analyzer")
st.markdown("Upload your CSV file for instant analysis")

uploaded_file = st.file_uploader("Budget 2014-2025.csv", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Data", "📊 Stats", "📈 Charts", "⬇️ Download"])
        
        with tab1:
            st.subheader("Data Preview")
            st.dataframe(df.head(100), use_container_width=True)
            col1, col2 = st.columns(2)
            col1.metric("Rows", len(df))
            col2.metric("Columns", len(df.columns))
        
        with tab2:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.subheader("Column Info")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null': df.count(),
                'Null': df.isnull().sum()
            })
            st.dataframe(info_df, use_container_width=True)
        
        with tab3:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                st.subheader("Histogram")
                col = st.selectbox("Select column", numeric_cols)
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)
                
                if len(numeric_cols) > 1:
                    st.subheader("Correlation Heatmap")
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
                    st.plotly_chart(fig, use_container_width=True)
                
                if len(numeric_cols) >= 2:
                    st.subheader("Scatter Plot")
                    col1, col2 = st.columns(2)
                    x = col1.selectbox("X-axis", numeric_cols)
                    y = col2.selectbox("Y-axis", [c for c in numeric_cols if c != x])
                    fig = px.scatter(df, x=x, y=y, title=f"{x} vs {y}")
                    st.plotly_chart(fig, use_container_width=True)
            
            if categorical_cols:
                st.subheader("Bar Chart")
                col = st.selectbox("Select categorical column", categorical_cols)
                counts = df[col].value_counts().head(20)
                fig = px.bar(x=counts.index, y=counts.values, labels={'x': col, 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Original Data", csv, "data.csv", "text/csv")
            
            with col2:
                cleaned = df.drop_duplicates().to_csv(index=False).encode('utf-8')
                st.download_button("📥 Cleaned Data", cleaned, "cleaned.csv", "text/csv")
            
            with col3:
                stats = df.describe().to_csv().encode('utf-8')
                st.download_button("📥 Statistics", stats, "stats.csv", "text/csv")
            
            st.subheader("Data Quality")
            duplicates = df.duplicated().sum()
            missing = df.isnull().sum().sum()
            
            col1, col2 = st.columns(2)
            col1.metric("Duplicate Rows", duplicates)
            col2.metric("Missing Values", missing)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("👆 Upload a CSV file to start")
