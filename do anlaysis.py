st.success("CSV Loaded Successfully!")
st.write("### 🔍 Preview of Data")
st.dataframe(df, use_container_width=True)

# --------- BASIC INFO ----------
st.write("### 📌 Basic Information")
st.write(df.describe(include="all"))

# --------- CHARTS ----------
# include all numeric types reliably
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
all_columns = df.columns.tolist()

if len(numeric_columns) > 0:
    st.write("### 📈 Line Chart")
    # X-axis choice: index or any column
    x_axis_options = ["Index"] + all_columns
    x_choice = st.selectbox("Select X-axis", x_axis_options, index=0, key="line_x")
    y_choice = st.selectbox("Select Y (numeric) for line chart", numeric_columns, key="line_y")

    if x_choice == "Index":
        fig = px.line(df, y=y_choice, title=f"Line Chart of {y_choice} (Index as X)")
    else:
        fig = px.line(df, x=x_choice, y=y_choice, title=f"Line Chart of {y_choice} vs {x_choice}")
    st.plotly_chart(fig, use_container_width=True)

    st.write("### 📊 Bar Chart")
    x_choice_bar = st.selectbox("Select X-axis for bar chart", x_axis_options, index=0, key="bar_x")
    y_choice_bar = st.selectbox("Select Y (numeric) for bar chart", numeric_columns, key="bar_y")

    if x_choice_bar == "Index":
        fig_bar = px.bar(df, y=y_choice_bar, title=f"Bar Chart of {y_choice_bar} (Index as X)")
    else:
        fig_bar = px.bar(df, x=x_choice_bar, y=y_choice_bar, title=f"Bar Chart of {y_choice_bar} vs {x_choice_bar}")
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.warning("No numeric columns found for charting. Try uploading a file with numeric data (integers/floats).")
