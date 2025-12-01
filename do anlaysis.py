import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="CSV Data Analyzer",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 CSV Data Analysis App")
st.markdown("Upload your CSV file and get instant insights with interactive visualizations")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display basic information
        st.success(f"✅ File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Data Preview", 
            "📊 Statistics", 
            "📈 Visualizations", 
            "🔍 Data Quality",
            "⬇️ Download"
        ])
        
        with tab1:
            st.subheader("Data Preview")
            st.dataframe(df.head(100), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
        
        with tab2:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with tab3:
            st.subheader("Interactive Visualizations")
            
            # Get numeric and categorical columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                st.markdown("### Numeric Column Analysis")
                
                # Histogram
                col1, col2 = st.columns([1, 3])
                with col1:
                    hist_col = st.selectbox("Select column for histogram", numeric_cols)
                with col2:
                    fig_hist = px.histogram(df, x=hist_col, title=f"Distribution of {hist_col}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Correlation heatmap
                if len(numeric_cols) > 1:
                    st.markdown("### Correlation Heatmap")
                    corr_matrix = df[numeric_cols].corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        labels=dict(color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale="RdBu_r",
                        aspect="auto"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Scatter plot
                if len(numeric_cols) >= 2:
                    st.markdown("### Scatter Plot")
                    col1, col2 = st.columns(2)
                    with col1:
                        x_axis = st.selectbox("Select X-axis", numeric_cols, key='scatter_x')
                    with col2:
                        y_axis = st.selectbox("Select Y-axis", [col for col in numeric_cols if col != x_axis], key='scatter_y')
                    
                    fig_scatter = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            if categorical_cols:
                st.markdown("### Categorical Column Analysis")
                cat_col = st.selectbox("Select categorical column", categorical_cols)
                
                # Value counts
                value_counts = df[cat_col].value_counts().head(20)
                fig_bar = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    labels={'x': cat_col, 'y': 'Count'},
                    title=f"Top 20 values in {cat_col}"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab4:
            st.subheader("Data Quality Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Missing Values")
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing Count': df.isnull().sum().values,
                    'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                
                if len(missing_df) > 0:
                    st.dataframe(missing_df, use_container_width=True)
                    
                    fig_missing = px.bar(
                        missing_df,
                        x='Column',
                        y='Missing %',
                        title="Missing Values by Column"
                    )
                    st.plotly_chart(fig_missing, use_container_width=True)
                else:
                    st.success("✅ No missing values found!")
            
            with col2:
                st.markdown("### Duplicate Rows")
                duplicates = df.duplicated().sum()
                st.metric("Duplicate Rows", duplicates)
                
                if duplicates > 0:
                    st.warning(f"Found {duplicates} duplicate rows ({(duplicates/len(df)*100):.2f}%)")
                else:
                    st.success("✅ No duplicate rows found!")
        
        with tab5:
            st.subheader("Download Processed Data")
            
            st.markdown("### Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download original data
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Original Data",
                    data=csv,
                    file_name="original_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download cleaned data (without duplicates)
                cleaned_df = df.drop_duplicates()
                cleaned_csv = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned Data",
                    data=cleaned_csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
            
            # Download statistical summary
            summary_csv = df.describe().to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Statistical Summary",
                data=summary_csv,
                file_name="statistical_summary.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please make sure your file is a valid CSV format")

else:
    st.info("👆 Please upload a CSV file to begin analysis")
    
    # Show example
    st.markdown("### Example CSV Format")
    st.code("""
Name,Age,City,Salary
John,30,New York,50000
Jane,25,Los Angeles,60000
Bob,35,Chicago,55000
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
