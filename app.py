import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import ttest_ind
import pickle

# New imports for Random Forest and Gradient Boosting
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder # For encoding categorical targets if needed

st.set_page_config(page_title="Basic Stats Explorer", layout="wide")

thick_line = "<hr style='height:4px;border:none;background-color:#333;'>"

# ---- CACHED DATA LOADER ----
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        # parse all possible dates
        return pd.read_excel(uploaded_file, parse_dates=True)

# ---- TYPE DETECTION ----
def detect_columns(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    # detect datetime more robustly
    dt_cols = [c for c in df.columns
               if np.issubdtype(df[c].dtype, np.datetime64)]
    return num_cols, cat_cols, dt_cols

# ---- LAYMAN SUMMARY ----
def render_layman_summary(df, num_cols, cat_cols, dt_cols):
    st.markdown(thick_line, unsafe_allow_html=True)
    st.header("📝 Layman-Friendly Summary")
    epsilon = 1e-6

    for col in num_cols:
        data = df[col].dropna()
        if data.empty:
            st.subheader(f"Numeric Column — {col}")
            st.write(f"No data available for {col}.")
            continue

        st.subheader(f"Numeric Column — {col}")
        st.write(f"**Count:** {len(data)}")
        st.write(f"**Mean (Average):** {data.mean():.2f}")
        st.write(f"**Median (Middle Value):** {data.median():.2f}")
        st.write(f"**Range (Min to Max):** {data.min():.2f} to {data.max():.2f}")

        std_dev = data.std()
        mean_val = data.mean()

        if mean_val != 0:
            cv = std_dev / abs(mean_val) # Coefficient of Variation
            if cv < 0.1:
                st.write("The values in this column are **tightly clustered** around the average.")
            elif cv < 0.5:
                st.write("The values in this column are **moderately spread out**.")
            else:
                st.write("The values in this column are **quite spread out**.")
        else:
            st.write("The values in this column are spread out, but the mean is zero, so a direct comparison isn't straightforward.")

        st.write("") # Add a newline for spacing

    for col in cat_cols:
        data = df[col].dropna()
        if data.empty:
            st.subheader(f"Categorical Column — {col}")
            st.write(f"No data available for {col}.")
            continue

        st.subheader(f"Categorical Column — {col}")
        top_category = data.mode()[0]
        top_category_count = data.value_counts()[top_category]
        top_category_percent = (top_category_count / len(data)) * 100
        unique_count = data.nunique()

        st.write(f"**Unique Categories:** {unique_count}")
        st.write(f"The most common category is **'{top_category}'**, appearing in {top_category_percent:.1f}% of the records.")
        if unique_count > 10:
            st.write(f"This column has many unique categories ({unique_count}), which might indicate a wide variety of types.")
        elif unique_count > 2:
            st.write(f"This column has a few distinct categories.")
        else:
            st.write(f"This column has only {unique_count} categories, making it a simple distinction.")
        st.write("") # Add a newline for spacing

    for col in dt_cols:
        data = df[col].dropna()
        if data.empty:
            st.subheader(f"Datetime Column — {col}")
            st.write(f"No data available for {col}.")
            continue

        st.subheader(f"Datetime Column — {col}")
        min_date = data.min()
        max_date = data.max()
        time_span = max_date - min_date
        st.write(f"This column contains dates ranging from **{min_date.strftime('%Y-%m-%d')}** to **{max_date.strftime('%Y-%m-%d')}**.")
        st.write(f"This covers a period of approximately **{time_span.days} days**.")
        st.write("") # Add a newline for spacing

# ---- T-TEST ----
def render_ttest(df, num_cols, cat_cols):
    st.markdown(thick_line, unsafe_allow_html=True)
    st.header("🔬 Two-Group Comparison (T-Test)")
    st.write("This analysis helps you understand if there's a significant difference in the average of a numeric variable between two groups defined by a categorical variable.")

    binary_cat_cols = [c for c in cat_cols if df[c].nunique() == 2]

    if not binary_cat_cols:
        st.warning("No binary (two-group) categorical columns found in your data to perform a T-Test.")
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_cat_col = st.selectbox("Select a binary categorical column (groups)", binary_cat_cols, key="ttest_cat")
    with col2:
        selected_num_col = st.selectbox("Select a numeric column (values to compare)", num_cols, key="ttest_num")

    if selected_cat_col and selected_num_col:
        # Drop rows where either selected column has NaN
        clean_df = df[[selected_cat_col, selected_num_col]].dropna()

        if clean_df.empty:
            st.warning("No complete data points for the selected columns after dropping missing values.")
            return

        groups = clean_df[selected_cat_col].unique()
        if len(groups) != 2:
            st.warning(f"Selected categorical column '{selected_cat_col}' does not have exactly two unique groups after cleaning. Please choose a binary column.")
            return

        group1_name, group2_name = groups[0], groups[1]
        group1_data = clean_df[clean_df[selected_cat_col] == group1_name][selected_num_col]
        group2_data = clean_df[clean_df[selected_cat_col] == group2_name][selected_num_col]

        if len(group1_data) < 2 or len(group2_data) < 2:
            st.warning(f"Not enough data points in one or both groups ({group1_name}: {len(group1_data)} samples, {group2_name}: {len(group2_data)} samples) to perform a T-test. Need at least 2 samples per group.")
            return

        # Perform Welch's t-test (handles unequal variances)
        t_stat, p_value = ttest_ind(group1_data, group2_data, equal_var=False)

        st.subheader(f"Results for {selected_num_col} by {selected_cat_col}")
        st.write(f"**Average for '{group1_name}':** {group1_data.mean():.2f}")
        st.write(f"**Average for '{group2_name}':** {group2_data.mean():.2f}")
        st.write(f"**T-statistic:** {t_stat:.3f}")
        st.write(f"**P-value:** {p_value:.3f}")

        st.subheader("Layman's Interpretation:")
        if p_value < 0.05:
            st.success(f"The p-value ({p_value:.3f}) is less than 0.05. This suggests there is a **statistically significant difference** in the average {selected_num_col} between '{group1_name}' and '{group2_name}'.")
            st.write(f"In simpler terms, it's very unlikely that the observed difference in averages happened by random chance. We can be reasonably confident that the two groups are genuinely different in terms of {selected_num_col}.")
        else:
            st.info(f"The p-value ({p_value:.3f}) is greater than 0.05. This suggests there is **no statistically significant difference** in the average {selected_num_col} between '{group1_name}' and '{group2_name}'.")
            st.write(f"In simpler terms, any observed difference in averages could easily be due to random chance. We don't have enough evidence to conclude that the two groups are genuinely different in terms of {selected_num_col}.")

# ---- REGRESSION ANALYSIS ----
def render_regression(df, num_cols):
    st.markdown(thick_line, unsafe_allow_html=True)
    st.header("📈 Regression Analysis")
    st.write("Regression helps you understand how one or more 'predictor' variables influence a 'target' variable. It can also be used to predict the target variable's value.")

    if not num_cols:
        st.warning("No numeric columns available for regression analysis.")
        return

    target_col = st.selectbox("Select your Target (Predict) Numeric Column", num_cols, key="reg_target")
    available_predictors = [col for col in num_cols if col != target_col]

    if not available_predictors:
        st.warning("Not enough numeric columns to perform regression. Need at least one target and one predictor.")
        return

    predictor_cols = st.multiselect("Select Predictor (Feature) Numeric Columns", available_predictors, key="reg_predictors")

    if target_col and predictor_cols:
        # Prepare data for regression
        X = df[predictor_cols].copy()
        y = df[target_col].copy()

        # Drop rows with any NaN values in selected columns
        combined_data = pd.concat([X, y], axis=1).dropna()
        if combined_data.empty:
            st.warning("No complete data points for the selected target and predictor columns after dropping missing values.")
            return

        X = combined_data[predictor_cols]
        y = combined_data[target_col]

        if X.shape[0] < 2:
            st.warning("Not enough data points to perform regression after cleaning. Need at least 2 samples.")
            return

        # Multicollinearity check
        st.subheader("Multicollinearity Check:")
        corr_matrix = X.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

        if high_corr_pairs:
            st.warning("🚨 **Warning: High Multicollinearity Detected!**")
            st.write("Some of your selected predictor variables are highly correlated with each other (correlation coefficient > 0.8).")
            st.write("This means they provide similar information to the model and can make the individual impact of each predictor harder to interpret, and the model less stable.")
            st.write("Consider removing one of the highly correlated variables for a cleaner model.")
            for p1, p2, corr_val in high_corr_pairs:
                st.write(f"- '{p1}' and '{p2}' have a correlation of {corr_val:.2f}")
        else:
            st.success("No significant multicollinearity detected among selected predictors (all correlations < 0.8).")

        st.markdown("---")
        st.subheader("Model Training and Results:")

        # Train the model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        st.write(f"**R-squared (R²):** {r2:.3f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.3f}")

        st.markdown("---")
        st.subheader("Regression Equation:")
        equation = f"**{target_col}** = {model.intercept_:.2f}"
        for i, coef in enumerate(model.coef_):
            equation += f" + ({coef:.2f} * **{predictor_cols[i]}**)"
        st.code(equation)

        st.markdown("---")
        st.subheader("Layman's Interpretation:")
        st.write(f"The **R-squared (R²)** value tells us how much of the variation in **{target_col}** can be explained by your chosen predictors.")
        if r2 > 0.7:
            st.success(f"An R² of {r2:.3f} indicates a **strong fit**. Your predictors explain a large portion ({r2*100:.2f}%) of the changes in {target_col}.")
        elif r2 > 0.3:
            st.info(f"An R² of {r2:.3f} indicates a **moderate fit**. Your predictors explain some ({r2*100:.2f}%) of the changes in {target_col}.")
        else:
            st.warning(f"An R² of {r2:.3f} indicates a **weak fit**. Your predictors explain only a small portion ({r2*100:.2f}%) of the changes in {target_col}. Other factors might be more influential.")

        st.write(f"The **Root Mean Squared Error (RMSE)** of {rmse:.3f} represents the typical prediction error of the model.")
        # Try to infer units for RMSE
        target_lower = target_col.lower()
        if "score" in target_lower:
            st.write(f"This means your predictions for {target_col} are typically off by about {rmse:.2f} points.")
        elif "price" in target_lower or "cost" in target_lower or "revenue" in target_lower or "sales" in target_lower:
            st.write(f"This means your predictions for {target_col} are typically off by about ${rmse:.2f}.")
        elif "count" in target_lower or "number" in target_lower:
            st.write(f"This means your predictions for {target_col} are typically off by about {rmse:.2f} units.")
        else:
            st.write(f"This means your predictions for {target_col} are typically off by about {rmse:.2f} units of {target_col}.")

        st.markdown("---")
        st.subheader("Diagnostic Plots:")
        # Residuals vs Predicted
        fig_res = px.scatter(x=y_pred, y=y - y_pred,
                             labels={'x':'Predicted Values', 'y':'Residuals (Actual - Predicted)'},
                             title='Residuals vs. Predicted Values')
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)
        st.write("Ideally, residuals should be randomly scattered around zero, with no clear pattern. This indicates that the model is capturing the underlying relationships well.")

        # Histogram of Residuals
        fig_hist_res = px.histogram(y - y_pred, nbins=50,
                                    labels={'value':'Residuals'},
                                    title='Distribution of Residuals')
        st.plotly_chart(fig_hist_res, use_container_width=True)
        st.write("Ideally, residuals should be normally distributed (bell-shaped). This suggests that the model's errors are random and not systematically biased.")

# ---- QUADRANT ANALYSIS ----
def __render_quadrant_analysis__(df, num_cols, all_dataframe_cols):
    st.markdown(thick_line, unsafe_allow_html=True)
    st.header("🍀 Quadrant Analysis")
    st.write("Quadrant analysis helps categorize data points into four groups based on two key numeric variables and their thresholds (e.g., mean or median).")

    if len(num_cols) < 2:
        st.warning("Need at least two numeric columns for Quadrant Analysis.")
        return

    col_x = st.selectbox("Select X-axis (Numeric Column)", num_cols, key="quad_x")
    remaining_num_cols = [col for col in num_cols if col != col_x]
    col_y = st.selectbox("Select Y-axis (Numeric Column)", remaining_num_cols, key="quad_y")

    if col_x and col_y:
        # Drop rows with NaN in selected columns
        clean_df = df[[col_x, col_y]].dropna()
        if clean_df.empty:
            st.warning("No complete data points for the selected X and Y columns after dropping missing values.")
            return

        # Threshold selection
        st.subheader("Set Thresholds:")
        threshold_option_x = st.radio(f"Threshold for {col_x}:", ["Mean", "Median", "Custom"], key="thresh_x_opt")
        threshold_x = clean_df[col_x].mean() if threshold_option_x == "Mean" else \
                      clean_df[col_x].median() if threshold_option_x == "Median" else \
                      st.number_input(f"Enter custom threshold for {col_x}", value=float(clean_df[col_x].mean()), key="thresh_x_custom")

        threshold_option_y = st.radio(f"Threshold for {col_y}:", ["Mean", "Median", "Custom"], key="thresh_y_opt")
        threshold_y = clean_df[col_y].mean() if threshold_option_y == "Mean" else \
                      clean_df[col_y].median() if threshold_option_y == "Median" else \
                      st.number_input(f"Enter custom threshold for {col_y}", value=float(clean_df[col_y].mean()), key="thresh_y_custom")

        # Add an optional column for hover information
        hover_data_cols = st.multiselect(
            "Select additional columns to show on hover (optional)",
            [c for c in all_dataframe_cols if c not in [col_x, col_y]],
            key="quad_hover_cols"
        )

        # Assign quadrants
        clean_df['Quadrant'] = ''
        clean_df.loc[(clean_df[col_x] >= threshold_x) & (clean_df[col_y] >= threshold_y), 'Quadrant'] = 'High X, High Y'
        clean_df.loc[(clean_df[col_x] < threshold_x) & (clean_df[col_y] >= threshold_y), 'Quadrant'] = 'Low X, High Y'
        clean_df.loc[(clean_df[col_x] < threshold_x) & (clean_df[col_y] < threshold_y), 'Quadrant'] = 'Low X, Low Y'
        clean_df.loc[(clean_df[col_x] >= threshold_x) & (clean_df[col_y] < threshold_y), 'Quadrant'] = 'High X, Low Y'

        # Plotting
        fig = px.scatter(clean_df, x=col_x, y=col_y, color='Quadrant',
                         hover_data=[col_x, col_y] + hover_data_cols,
                         title=f"Quadrant Analysis of {col_x} vs {col_y}")

        fig.add_hline(y=threshold_y, line_dash="dash", line_color="red", annotation_text=f"Y Threshold ({threshold_y:.2f})", annotation_position="bottom right")
        fig.add_vline(x=threshold_x, line_dash="dash", line_color="red", annotation_text=f"X Threshold ({threshold_x:.2f})", annotation_position="top left")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Quadrant Counts:")
        quadrant_counts = clean_df['Quadrant'].value_counts().reset_index()
        quadrant_counts.columns = ['Quadrant', 'Count']
        st.dataframe(quadrant_counts)

        st.subheader("Layman's Interpretation:")
        st.write(f"This plot divides your data points into four groups based on their values for '{col_x}' and '{col_y}' relative to the chosen thresholds.")
        st.write(f"- **High {col_x}, High {col_y}:** Points in this quadrant are above average (or your custom threshold) for both '{col_x}' and '{col_y}'. These are often your 'top performers' or 'high value' items.")
        st.write(f"- **Low {col_x}, High {col_y}:** Points here are below average for '{col_x}' but above average for '{col_y}'.")
        st.write(f"- **Low {col_x}, Low {col_y}:** Points in this quadrant are below average for both. These might be 'underperformers' or 'low value' items.")
        st.write(f"- **High {col_x}, Low {col_y}:** Points here are above average for '{col_x}' but below average for '{col_y}'.")

def render_quadrant_analysis(df, num_cols, all_dataframe_cols):
    st.markdown(thick_line, unsafe_allow_html=True)
    st.header("🍀 Quadrant Analysis")
    st.write("Quadrant analysis helps categorize data points into four groups based on two key numeric variables and their thresholds (e.g., mean or median).")

    if len(num_cols) < 2:
        st.warning("Need at least two numeric columns for Quadrant Analysis.")
        return

    col_x = st.selectbox("Select X-axis (Numeric Column)", num_cols, key="quad_x")
    remaining_num_cols = [col for col in num_cols if col != col_x]
    col_y = st.selectbox("Select Y-axis (Numeric Column)", remaining_num_cols, key="quad_y")

    if col_x and col_y:
        # Add an optional column for hover information
        hover_data_cols = st.multiselect(
            "Select additional columns to show on hover (optional)",
            [c for c in all_dataframe_cols if c not in [col_x, col_y]],
            key="quad_hover_cols"
        )

        # Drop rows with NaN in selected columns, ensuring all hover_data_cols are included
        # This is the fix: include hover_data_cols in the subset for dropping NaNs
        columns_to_include = [col_x, col_y] + hover_data_cols
        clean_df = df[columns_to_include].dropna()

        if clean_df.empty:
            st.warning("No complete data points for the selected X, Y, and hover columns after dropping missing values.")
            return

        # Threshold selection
        st.subheader("Set Thresholds:")
        threshold_option_x = st.radio(f"Threshold for {col_x}:", ["Mean", "Median", "Custom"], key="thresh_x_opt")
        threshold_x = clean_df[col_x].mean() if threshold_option_x == "Mean" else \
                      clean_df[col_x].median() if threshold_option_x == "Median" else \
                      st.number_input(f"Enter custom threshold for {col_x}", value=float(clean_df[col_x].mean()), key="thresh_x_custom")

        threshold_option_y = st.radio(f"Threshold for {col_y}:", ["Mean", "Median", "Custom"], key="thresh_y_opt")
        threshold_y = clean_df[col_y].mean() if threshold_option_y == "Mean" else \
                      clean_df[col_y].median() if threshold_option_y == "Median" else \
                      st.number_input(f"Enter custom threshold for {col_y}", value=float(clean_df[col_y].mean()), key="thresh_y_custom")

        # Assign quadrants
        clean_df['Quadrant'] = ''
        clean_df.loc[(clean_df[col_x] >= threshold_x) & (clean_df[col_y] >= threshold_y), 'Quadrant'] = 'High X, High Y'
        clean_df.loc[(clean_df[col_x] < threshold_x) & (clean_df[col_y] >= threshold_y), 'Quadrant'] = 'Low X, High Y'
        clean_df.loc[(clean_df[col_x] < threshold_x) & (clean_df[col_y] < threshold_y), 'Quadrant'] = 'Low X, Low Y'
        clean_df.loc[(clean_df[col_x] >= threshold_x) & (clean_df[col_y] < threshold_y), 'Quadrant'] = 'High X, Low Y'

        # Plotting
        fig = px.scatter(clean_df, x=col_x, y=col_y, color='Quadrant',
                         hover_data=[col_x, col_y] + hover_data_cols,
                         title=f"Quadrant Analysis of {col_x} vs {col_y}")

        fig.add_hline(y=threshold_y, line_dash="dash", line_color="red", annotation_text=f"Y Threshold ({threshold_y:.2f})", annotation_position="bottom right")
        fig.add_vline(x=threshold_x, line_dash="dash", line_color="red", annotation_text=f"X Threshold ({threshold_x:.2f})", annotation_position="top left")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Quadrant Counts:")
        quadrant_counts = clean_df['Quadrant'].value_counts().reset_index()
        quadrant_counts.columns = ['Quadrant', 'Count']
        st.dataframe(quadrant_counts)

        st.subheader("Layman's Interpretation:")
        st.write(f"This plot divides your data points into four groups based on their values for '{col_x}' and '{col_y}' relative to the chosen thresholds.")
        st.write(f"- **High {col_x}, High {col_y}:** Points in this quadrant are above average (or your custom threshold) for both '{col_x}' and '{col_y}'. These are often your 'top performers' or 'high value' items.")
        st.write(f"- **Low {col_x}, High {col_y}:** Points here are below average for '{col_x}' but above average for '{col_y}'.")
        st.write(f"- **Low {col_x}, Low {col_y}:** Points in this quadrant are below average for both. These might be 'underperformers' or 'low value' items.")
        st.write(f"- **High {col_x}, Low {col_y}:** Points here are above average for '{col_x}' but below average for '{col_y}'.")

# ---- HISTOGRAM & KDE ----
def render_histogram_with_kde_groupby(df, num_cols, cat_cols):
    st.markdown(thick_line, unsafe_allow_html=True)
    st.header("📊 Histogram & Density Plot")
    st.markdown("This section visualizes the distribution of your numeric data using histograms and Kernel Density Estimates (KDE).")

    if not num_cols:
        st.info("No numeric columns found in your dataset to perform Histogram & Density Plot analysis.")
        return

    # User selects the numeric column for analysis
    selected_col = st.selectbox("Select a numeric column for Histogram & Density Plot", num_cols, key=f"hist_col_select_{[num_cols][0]}")

    if selected_col:
        # User selects an optional categorical column to group by
        group_by_options = ["None"] + cat_cols
        group_by_col = st.selectbox("Group histogram by (optional categorical column)", group_by_options, key=f"hist_group_by_select_{[num_cols][0]}")

        # Drop NaN values for the selected column(s)
        # If grouping, drop NaNs from both the selected numeric column and the group-by column
        if group_by_col != "None":
            plot_df = df[[selected_col, group_by_col]].dropna()
            if plot_df.empty:
                st.warning(f"No data available for '{selected_col}' when grouped by '{group_by_col}' after dropping missing values. Please check your data.")
                return
        else:
            plot_df = df[[selected_col]].dropna()
            if plot_df.empty:
                st.warning(f"No data available for '{selected_col}' after dropping missing values. Please check your data.")
                return

        col_data = plot_df[selected_col]

        st.subheader(f"Distribution of '{selected_col}'")

        # Create the histogram with KDE using Plotly Express
        # Use the 'color' argument if a group_by_col is selected
        if group_by_col != "None":
            fig = px.histogram(
                plot_df,
                x=selected_col,
                color=group_by_col, # Group by the selected categorical column
                marginal="box", # Add marginal box plot
                hover_data=plot_df.columns,
                title=f"Histogram and KDE of {selected_col} grouped by {group_by_col}",
                labels={selected_col: selected_col, group_by_col: group_by_col},
                template="plotly_white",
                histnorm='probability density' # Normalize to show density
            )
        else:
            fig = px.histogram(
                plot_df,
                x=selected_col,
                marginal="box", # Add marginal box plot
                hover_data=plot_df.columns,
                title=f"Histogram and KDE of {selected_col}",
                labels={selected_col: selected_col},
                template="plotly_white",
                histnorm='probability density' # Normalize to show density
            )

        fig.update_layout(
            bargap=0.1, # Gap between bars
            xaxis_title=selected_col,
            yaxis_title="Density",
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Key Statistics and Interpretation")
        st.markdown(f"**Column:** `{selected_col}`")

        # Calculate statistics only on the selected numeric column data
        mean_val = col_data.mean()
        median_val = col_data.median()
        std_dev_val = col_data.std()
        skewness_val = col_data.skew()

        st.write(f"- **Mean:** `{mean_val:.2f}` (The average value)")
        st.write(f"- **Median:** `{median_val:.2f}` (The middle value when data is ordered)")
        st.write(f"- **Standard Deviation:** `{std_dev_val:.2f}` (Measures the spread or dispersion of the data)")
        st.write(f"- **Skewness:** `{skewness_val:.2f}` (Indicates the asymmetry of the distribution)")

        st.markdown("---")
        st.markdown("#### Interpretation of Skewness:")
        if skewness_val > 0.5:
            st.write("📈 **Positive Skew (Right-skewed):** The tail on the right side of the distribution is longer or fatter. This means there are more values concentrated on the left side, with some higher values pulling the mean to the right (e.g., income distribution).")
        elif skewness_val < -0.5:
            st.write("📉 **Negative Skew (Left-skewed):** The tail on the left side of the distribution is longer or fatter. This means there are more values concentrated on the right side, with some lower values pulling the mean to the left (e.g., test scores where most students did well).")
        else:
            st.write("⚖️ **Approximately Symmetrical:** The distribution is relatively balanced, with similar tails on both sides. The mean and median are likely close to each other (e.g., height distribution).")
            
def render_histogram_with_kde(df, num_cols):
    st.markdown(thick_line, unsafe_allow_html=True)
    st.header("📊 Histogram & Kernel Density Estimate (KDE)")
    st.subheader(f"Distribution — {num_cols[0]}")
    st.write("This plot shows the distribution of a numeric variable. The bars (histogram) show how many data points fall into specific ranges, and the curve (KDE) provides a smoothed estimate of the data's probability density.")

    selected_col = st.selectbox("Select a Numeric Column", num_cols, key=f"hist_kde_col_{num_cols[0]}")

    if selected_col:
        data = df[selected_col].dropna()
        if data.empty:
            st.warning(f"No data available for {selected_col} after dropping missing values.")
            return

        # Plotly Histogram with KDE
        fig = px.histogram(data, x=selected_col, nbins=50, marginal="box",
                           title=f"Distribution of {selected_col} with KDE")
        fig.update_traces(marker_color='skyblue', selector=dict(type='histogram'))

        # Add KDE manually using gaussian_kde for matplotlib, then convert to Plotly compatible
        # For simplicity and direct Plotly integration, px.histogram with marginal="box" is usually sufficient
        # If a true KDE line is desired on top of histogram, it's more complex with Plotly Express directly.
        # Let's stick to the marginal box plot for now as it's a good summary.
        # Alternatively, use go.Figure and add traces.
        # For now, px.histogram with marginal="box" covers the spirit.

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Layman's Interpretation:")
        st.write(f"The **histogram (bars)** shows the frequency of values in different ranges. Taller bars mean more data points fall into that range.")
        st.write(f"The **box plot on top** gives a quick summary: the middle line is the median, the box covers the middle 50% of data, and the 'whiskers' extend to typical data range, with dots indicating outliers.")

        st.write(f"**Key Statistics for {selected_col}:**")
        st.write(f"- **Mean (Average):** {data.mean():.2f}")
        st.write(f"- **Median (Middle Value):** {data.median():.2f}")
        st.write(f"- **Standard Deviation (Spread):** {data.std():.2f}")
        st.write(f"- **Skewness:** {data.skew():.2f}")

        if data.skew() > 0.5:
            st.write("The distribution is **positively skewed (right-skewed)**. This means there's a longer tail on the right side, suggesting a few unusually high values pulling the average up.")
        elif data.skew() < -0.5:
            st.write("The distribution is **negatively skewed (left-skewed)**. This means there's a longer tail on the left side, suggesting a few unusually low values pulling the average down.")
        else:
            st.write("The distribution is relatively **symmetrical**. Values are distributed fairly evenly around the average.")

# ---- BOX PLOT ----
def render_boxplot(df, num_cols, all_dataframe_cols):
    st.markdown(thick_line, unsafe_allow_html=True)
    st.header(f"📦 Box Plot — {num_cols[0]}")
    st.write("A box plot visually summarizes the distribution of a numeric variable, showing its median, quartiles, and potential outliers.")

    selected_col = st.selectbox("Select a Numeric Column", num_cols, key=f"box_plot_col_{num_cols[0]}")

    if selected_col:
        data = df[selected_col].dropna()
        if data.empty:
            st.warning(f"No data available for {selected_col} after dropping missing values.")
            return

        # Optional: Select a column to color/group the box plot by
        group_by_col = st.selectbox(
            "Group box plot by (optional categorical column)",
            ['None'] + [col for col in all_dataframe_cols if col != selected_col and df[col].dtype in ['object', 'category']],
            key=f"box_group_by_{num_cols}"
        )

        hover_data_cols = st.multiselect(
            "Select additional columns to show on hover for outliers (optional)",
            [c for c in all_dataframe_cols if c not in [selected_col, group_by_col]],
            key=f"box_hover_cols_{num_cols}"
        )

        if group_by_col != 'None':
            fig = px.box(df.dropna(subset=[selected_col, group_by_col]), x=group_by_col, y=selected_col,
                         points="all", hover_data=[selected_col, group_by_col] + hover_data_cols,
                         title=f"Box Plot of {selected_col} by {group_by_col}")
        else:
            fig = px.box(df.dropna(subset=[selected_col]), y=selected_col,
                         points="all", hover_data=[selected_col] + hover_data_cols,
                         title=f"Box Plot of {selected_col}")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Layman's Interpretation:")
        st.write(f"The **box** in the plot represents the middle 50% of your data for '{selected_col}'.")
        st.write(f"- The line inside the box is the **Median** (the middle value).")
        st.write(f"- The bottom of the box is the **25th Percentile** (25% of data is below this value).")
        st.write(f"- The top of the box is the **75th Percentile** (75% of data is below this value).")
        st.write(f"The **'whiskers'** (lines extending from the box) show the typical range of data. Points beyond the whiskers are considered **outliers** (unusually high or low values).")

        # Calculate IQR and outliers for interpretation
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_low = data[data < lower_bound]
        outliers_high = data[data > upper_bound]

        st.write(f"**Key Statistics for {selected_col}:**")
        st.write(f"- **Median:** {data.median():.2f}")
        st.write(f"- **Interquartile Range (IQR):** {IQR:.2f} (the spread of the middle 50% of data)")
        st.write(f"- **Number of Low Outliers:** {len(outliers_low)}")
        st.write(f"- **Number of High Outliers:** {len(outliers_high)}")

        if len(outliers_low) > 0 or len(outliers_high) > 0:
            st.warning("There are outliers detected. These are data points that are significantly different from the rest of the data and might warrant further investigation.")
        else:
            st.info("No significant outliers detected in this column.")

# ---- ECDF ----
def render_ecdf(df, num_cols):
    st.markdown(thick_line, unsafe_allow_html=True)
    st.header(f"📈 ECDF Plots — {num_cols[0]}")
    st.write("An Empirical Cumulative Distribution Function (ECDF) plot shows the proportion of data points that are less than or equal to a given value. It's great for understanding the overall distribution and percentiles.")

    selected_col = st.selectbox("Select a Numeric Column", num_cols, key=f"ecdf_col_{num_cols}")

    if selected_col:
        data = df[selected_col].dropna()
        if data.empty:
            st.warning(f"No data available for {selected_col} after dropping missing values.")
            return

        fig = px.ecdf(data, x=selected_col, title=f"Empirical Cumulative Distribution Function of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Layman's Interpretation:")
        st.write(f"The ECDF plot for '{selected_col}' shows you, for any given value on the X-axis, what percentage of your data falls at or below that value on the Y-axis.")
        st.write("- A **steep section** means many data points are clustered around that value.")
        st.write("- A **flat section** means fewer data points are in that range.")
        st.write("- The curve always goes from 0% to 100%.")

        st.write(f"**Key Percentiles for {selected_col}:**")
        st.write(f"- **25th Percentile:** {data.quantile(0.25):.2f} (25% of values are at or below this)")
        st.write(f"- **50th Percentile (Median):** {data.quantile(0.50):.2f} (50% of values are at or below this)")
        st.write(f"- **75th Percentile:** {data.quantile(0.75):.2f} (75% of values are at or below this)")

# ---- CORRELATION MATRIX ----
def render_correlation(df, num_cols):
    st.markdown(thick_line, unsafe_allow_html=True)
    st.header("🔗 Correlation Matrix")
    st.write("A correlation matrix shows how strongly pairs of numeric variables are linearly related. Values range from -1 (strong negative correlation) to 1 (strong positive correlation), with 0 meaning no linear relationship.")

    if not num_cols or len(num_cols) < 2:
        st.warning("Need at least two numeric columns to compute a correlation matrix.")
        return

    corr_df = df[num_cols].corr(numeric_only=True)
    r_squared_df = corr_df.apply(lambda x: x**2) # R-squared is correlation squared

    # Create annotations for the heatmap
    annotations = []
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            corr_val = corr_df.iloc[i, j]
            r2_val = r_squared_df.iloc[i, j]
            text = f"ρ={corr_val:.2f}<br>R²={r2_val:.2f}"
            annotations.append(
                dict(
                    x=corr_df.columns[j],
                    y=corr_df.index[i],
                    text=text,
                    showarrow=False,
                    font=dict(color="black" if abs(corr_val) < 0.7 else "white", size=10) # Adjust font color for readability
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale='RdBu', # Red-Blue color scale
        zmin=-1, zmax=1,
        colorbar=dict(title='Correlation Coefficient (ρ)')
    ))

    fig.update_layout(
        title='Correlation Matrix (ρ) and R-squared (R²)',
        xaxis_title="Variables",
        yaxis_title="Variables",
        annotations=annotations,
        height=max(500, len(num_cols) * 50), # Adjust height dynamically
        width=max(600, len(num_cols) * 60) # Adjust width dynamically
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Layman's Interpretation:")
    st.write("This chart shows the **strength and direction of linear relationships** between pairs of numeric variables.")
    st.write("- Values close to **1** (dark red) mean a strong **positive** relationship: as one variable increases, the other tends to increase.")
    st.write("- Values close to **-1** (dark blue) mean a strong **negative** relationship: as one variable increases, the other tends to decrease.")
    st.write("- Values close to **0** (white/light colors) mean **no linear relationship**.")
    st.write("The **R-squared (R²)** value (displayed below ρ) tells you how much of the variation in one variable can be explained by the other. For example, if ρ is 0.7, R² is 0.49, meaning 49% of the variation in one variable can be explained by the other.")

# ---- OUTLIER DETECTION ----
def render_outliers(df, num_cols):
    st.markdown(thick_line, unsafe_allow_html=True)
    st.header("🔍 Outlier Detection (IQR Method)")
    st.write("This section identifies data points that are unusually far from the majority of the data in each numeric column, using the Interquartile Range (IQR) method.")

    if not num_cols:
        st.warning("No numeric columns available for outlier detection.")
        return

    outlier_results = []
    for col in num_cols:
        data = df[col].dropna()
        if data.empty:
            outlier_results.append({
                "Column": col,
                "Lower Cutoff": "N/A",
                "Upper Cutoff": "N/A",
                "Low Outliers Count": 0,
                "Low Outliers %": "0.00%",
                "High Outliers Count": 0,
                "High Outliers %": "0.00%",
                "Total Outliers Count": 0,
                "Total Outliers %": "0.00%"
            })
            continue

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        low_outliers = data[data < lower_bound]
        high_outliers = data[data > upper_bound]

        total_outliers = len(low_outliers) + len(high_outliers)
        total_data_points = len(data)

        outlier_results.append({
            "Column": col,
            "Lower Cutoff": f"{lower_bound:.2f}",
            "Upper Cutoff": f"{upper_bound:.2f}",
            "Low Outliers Count": len(low_outliers),
            "Low Outliers %": f"{(len(low_outliers) / total_data_points * 100):.2f}%" if total_data_points > 0 else "0.00%",
            "High Outliers Count": len(high_outliers),
            "High Outliers %": f"{(len(high_outliers) / total_data_points * 100):.2f}%" if total_data_points > 0 else "0.00%",
            "Total Outliers Count": total_outliers,
            "Total Outliers %": f"{(total_outliers / total_data_points * 100):.2f}%" if total_data_points > 0 else "0.00%"
        })

    st.dataframe(pd.DataFrame(outlier_results))

    st.subheader("Layman's Interpretation:")
    st.write("Outliers are data points that fall significantly outside the typical range of values for a column.")
    st.write("- **Lower Cutoff:** Any value below this is considered a 'low' outlier.")
    st.write("- **Upper Cutoff:** Any value above this is considered a 'high' outlier.")
    st.write("Outliers can be real extreme values or indicate data entry errors. It's often good practice to investigate them, as they can sometimes heavily influence statistical analyses and models.")

# ---- RANDOM FOREST ANALYSIS ----
def render_random_forest(df, num_cols, cat_cols, all_dataframe_cols):
    st.header("🌳 Random Forest Analysis")
    st.write("Random Forest is a powerful predictive model that builds many 'decision trees' and combines their predictions. It's versatile for both predicting continuous numbers (regression) and categories (classification).")

    if not num_cols and not cat_cols:
        st.warning("No numeric or categorical columns available for Random Forest analysis.")
        return

    # Target variable selection
    target_col = st.selectbox("Select your Target (Predict) Column", all_dataframe_cols, key="rf_target")

    if target_col:
        # Determine if target is numeric (regression) or categorical (classification)
        is_regression = pd.api.types.is_numeric_dtype(df[target_col])
        is_binary_classification = False
        if not is_regression:
            unique_target_values = df[target_col].dropna().nunique()
            if unique_target_values == 2:
                is_binary_classification = True
            elif unique_target_values > 2:
                st.warning(f"Random Forest Classification currently supports binary (two-class) target variables. '{target_col}' has {unique_target_values} unique values. Please select a binary target or a numeric target for regression.")
                return

        st.info(f"Performing Random Forest {'Regression' if is_regression else 'Classification'} on '{target_col}'.")

        # Predictor variable selection (excluding target)
        available_predictors = [col for col in all_dataframe_cols if col != target_col]
        predictor_cols = st.multiselect("Select Predictor (Feature) Columns", available_predictors, key="rf_predictors")

        if not predictor_cols:
            st.warning("Please select at least one predictor column.")
            return

        # Prepare data: One-hot encode categorical predictors and handle missing values
        X = df[predictor_cols].copy()
        y = df[target_col].copy()

        # Identify categorical predictors for encoding
        cat_predictors_for_encoding = [col for col in predictor_cols if col in cat_cols]

        # Store original categorical unique values for export
        categorical_unique_values = {col: df[col].dropna().unique().tolist() for col in cat_predictors_for_encoding}

        # Perform one-hot encoding and capture the new column names
        X_before_dummies = X.copy() # Keep a copy before one-hot encoding to map
        if cat_predictors_for_encoding:
            X = pd.get_dummies(X, columns=cat_predictors_for_encoding, drop_first=True)

        # Map original categorical columns to their one-hot encoded counterparts
        one_hot_encoded_feature_map = {}
        for original_cat_col in cat_predictors_for_encoding:
            # Find columns in X that start with the original_cat_col name + '_'
            # This is a heuristic and might need adjustment for complex column names
            generated_cols = [col for col in X.columns if col.startswith(f"{original_cat_col}_")]
            if generated_cols:
                one_hot_encoded_feature_map[original_cat_col] = generated_cols
            # If drop_first=True, one category is dropped, so it won't have a generated column.
            # We need to ensure the prediction app can handle this.
            # For simplicity, we'll just list the generated columns. The prediction app
            # will need to know to fill 0s for the dropped category.

        # Ensure all columns are numeric for the model
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                st.warning(f"Column '{col}' is not numeric after encoding and cannot be used as a predictor. Please check your data types.")
                return

        # Align X and y after dropping NaNs
        combined_data = pd.concat([X, y], axis=1).dropna()
        if combined_data.empty:
            st.warning("No complete data points for the selected target and predictor columns after dropping missing values.")
            return

        X_clean = combined_data[X.columns]
        y_clean = combined_data[target_col]

        if X_clean.shape[0] < 5: # Need at least a few samples for train/test split
            st.warning("Not enough data points to perform analysis after cleaning. Need at least 5 samples.")
            return

        # Encode target for classification if binary
        le = None
        if is_binary_classification:
            le = LabelEncoder()
            y_clean = le.fit_transform(y_clean)
            st.write(f"Target classes encoded: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

        st.markdown("---")
        st.subheader("Model Training and Evaluation:")

        model = None
        if is_regression:
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.write(f"**R-squared (R²):** {r2:.3f}")
            st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.3f}")

            st.subheader("Layman's Interpretation (Regression):")

            st.write(f"The **R-squared (R²)** value tells us how much of the variation in **{target_col}** can be explained by your chosen predictors.")
            if r2 > 0.7:
                st.success(f"An R² of {r2:.3f} indicates a **strong fit**. Your predictors explain a large portion ({r2*100:.2f}%) of the changes in {target_col}.")
            elif r2 > 0.3:
                st.info(f"An R² of {r2:.3f} indicates a **moderate fit**. Your predictors explain some ({r2*100:.2f}%) of the changes in {target_col}.")
            else:
                st.warning(f"An R² of {r2:.3f} indicates a **weak fit**. Your predictors explain only a small portion ({r2*100:.2f}%) of the changes in {target_col}. Other factors might be more influential.")

            ## st.write(f"The **R-squared (R²)** of {r2:.3f} indicates how well the model explains the variability in '{target_col}'. A higher value means a better fit.")
            st.write(f"The **RMSE** of {rmse:.3f} is the typical error in your predictions for '{target_col}'. This means your predictions for {target_col} are typically off by about {rmse:.3f} points.")

        elif is_binary_classification:
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)

            st.write(f"**Accuracy:** {accuracy:.3f}")
            st.write(f"**Precision:** {precision:.3f}")
            st.write(f"**Recall:** {recall:.3f}")
            st.write(f"**F1-Score:** {f1:.3f}")
            st.write(f"**ROC AUC:** {roc_auc:.3f}")

            st.subheader("Layman's Interpretation (Classification):")
            st.write(f"**Accuracy ({accuracy:.3f}):** The percentage of correct predictions the model made.")
            st.write(f"**Precision ({precision:.3f}):** Out of all the times the model predicted the positive class, how often was it correct?")
            st.write(f"**Recall ({recall:.3f}):** Out of all the actual positive cases, how many did the model correctly identify?")
            st.write(f"**F1-Score ({f1:.3f}):** A balance between precision and recall.")
            st.write(f"**ROC AUC ({roc_auc:.3f}):** How well the model distinguishes between the two classes. A value closer to 1 is better.")

        if model:
            st.markdown("---")
            st.subheader("Feature Importance:")
            if hasattr(model, 'feature_importances_'):
                feature_importance_df = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                fig_importance = px.bar(feature_importance_df.head(10), x='Importance', y='Feature', orientation='h',
                                        title='Top 10 Feature Importances',
                                        labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'})
                fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)

                st.write("This chart shows which features (predictors) the Random Forest model considered most influential in making its predictions. Higher 'Importance' means the feature had a greater impact.")
            else:
                st.info("Feature importances are not available for this model type.")

            # Model Export Section
            st.markdown("---")
            st.subheader("Export Trained Model for Prediction:")
            export_data = {
                'model': model,
                'feature_names': X_train.columns.tolist(), # Features after one-hot encoding
                'target_column': target_col,
                'is_regression': is_regression,
                'original_predictor_cols': predictor_cols, # Original predictor names
                'categorical_unique_values': categorical_unique_values, # Original categories for dropdowns
                'one_hot_encoded_feature_map': one_hot_encoded_feature_map # Map for reconstruction
            }
            if is_binary_classification and le is not None:
                export_data['label_encoder'] = le

            # Convert to bytes for download
            pickled_model = pickle.dumps(export_data)

            st.download_button(
                label="Download Random Forest Model (.pkl)",
                data=pickled_model,
                file_name=f"random_forest_model_{target_col}.pkl",
                mime="application/octet-stream"
            )
            st.info("You can download this model and metadata to use it for making predictions on new, unseen data in a separate application.")

# ---- GRADIENT BOOSTING ANALYSIS (LightGBM) ----
def render_gradient_boosting(df, num_cols, cat_cols, all_dataframe_cols):
    st.markdown(thick_line, unsafe_allow_html=True)
    st.header("🚀 Gradient Boosting Analysis (LightGBM)")
    st.write("Gradient Boosting (specifically LightGBM here) is another highly effective predictive model that builds trees sequentially, with each new tree correcting errors of the previous ones. It's known for its speed and accuracy.")

    if not num_cols and not cat_cols:
        st.warning("No numeric or categorical columns available for Gradient Boosting analysis.")
        return

    # Target variable selection
    target_col = st.selectbox("Select your Target (Predict) Column", all_dataframe_cols, key="gb_target")

    if target_col:
        # Determine if target is numeric (regression) or categorical (classification)
        is_regression = pd.api.types.is_numeric_dtype(df[target_col])
        is_binary_classification = False
        if not is_regression:
            unique_target_values = df[target_col].dropna().nunique()
            if unique_target_values == 2:
                is_binary_classification = True
            elif unique_target_values > 2:
                st.warning(f"Gradient Boosting Classification currently supports binary (two-class) target variables. '{target_col}' has {unique_target_values} unique values. Please select a binary target or a numeric target for regression.")
                return

        st.info(f"Performing Gradient Boosting {'Regression' if is_regression else 'Classification'} on '{target_col}'.")

        # Predictor variable selection (excluding target)
        available_predictors = [col for col in all_dataframe_cols if col != target_col]
        predictor_cols = st.multiselect("Select Predictor (Feature) Columns", available_predictors, key="gb_predictors")

        if not predictor_cols:
            st.warning("Please select at least one predictor column.")
            return

        # Prepare data: One-hot encode categorical predictors and handle missing values
        X = df[predictor_cols].copy()
        y = df[target_col].copy()

        # Identify categorical predictors for encoding
        cat_predictors_for_encoding = [col for col in predictor_cols if col in cat_cols]

        # Store original categorical unique values for export
        categorical_unique_values = {col: df[col].dropna().unique().tolist() for col in cat_predictors_for_encoding}

        # Perform one-hot encoding and capture the new column names
        X_before_dummies = X.copy() # Keep a copy before one-hot encoding to map
        if cat_predictors_for_encoding:
            X = pd.get_dummies(X, columns=cat_predictors_for_encoding, drop_first=True)

        # Map original categorical columns to their one-hot encoded counterparts
        one_hot_encoded_feature_map = {}
        for original_cat_col in cat_predictors_for_encoding:
            generated_cols = [col for col in X.columns if col.startswith(f"{original_cat_col}_")]
            if generated_cols:
                one_hot_encoded_feature_map[original_cat_col] = generated_cols


        # Ensure all columns are numeric for the model
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                st.warning(f"Column '{col}' is not numeric after encoding and cannot be used as a predictor. Please check your data types.")
                return

        # Align X and y after dropping NaNs
        combined_data = pd.concat([X, y], axis=1).dropna()
        if combined_data.empty:
            st.warning("No complete data points for the selected target and predictor columns after dropping missing values.")
            return

        X_clean = combined_data[X.columns]
        y_clean = combined_data[target_col]

        if X_clean.shape[0] < 5: # Need at least a few samples for train/test split
            st.warning("Not enough data points to perform analysis after cleaning. Need at least 5 samples.")
            return

        # Encode target for classification if binary
        le = None
        if is_binary_classification:
            le = LabelEncoder()
            y_clean = le.fit_transform(y_clean)
            st.write(f"Target classes encoded: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

        st.markdown("---")
        st.subheader("Model Training and Evaluation:")

        model = None
        if is_regression:
            model = lgb.LGBMRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.write(f"**R-squared (R²):** {r2:.3f}")
            st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.3f}")

            st.subheader("Layman's Interpretation (Regression):")

            st.write(f"The **R-squared (R²)** value tells us how much of the variation in **{target_col}** can be explained by your chosen predictors.")
            if r2 > 0.7:
                st.success(f"An R² of {r2:.3f} indicates a **strong fit**. Your predictors explain a large portion ({r2*100:.2f}%) of the changes in {target_col}.")
            elif r2 > 0.3:
                st.info(f"An R² of {r2:.3f} indicates a **moderate fit**. Your predictors explain some ({r2*100:.2f}%) of the changes in {target_col}.")
            else:
                st.warning(f"An R² of {r2:.3f} indicates a **weak fit**. Your predictors explain only a small portion ({r2*100:.2f}%) of the changes in {target_col}. Other factors might be more influential.")

            ## st.write(f"The **R-squared (R²)** of {r2:.3f} indicates how well the model explains the variability in '{target_col}'. A higher value means a better fit.")
            st.write(f"The **RMSE** of {rmse:.3f} is the typical error in your predictions for '{target_col}'. This means your predictions for {target_col} are typically off by about {rmse:.3f} points.")

        elif is_binary_classification:
            model = lgb.LGBMClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)

            st.write(f"**Accuracy:** {accuracy:.3f}")
            st.write(f"**Precision:** {precision:.3f}")
            st.write(f"**Recall:** {recall:.3f}")
            st.write(f"**F1-Score:** {f1:.3f}")
            st.write(f"**ROC AUC:** {roc_auc:.3f}")

            st.subheader("Layman's Interpretation (Classification):")
            st.write(f"**Accuracy ({accuracy:.3f}):** The percentage of correct predictions the model made.")
            st.write(f"**Precision ({precision:.3f}):** Out of all the times the model predicted the positive class, how often was it correct?")
            st.write(f"**Recall ({recall:.3f}):** Out of all the actual positive cases, how many did the model correctly identify?")
            st.write(f"**F1-Score ({f1:.3f}):** A balance between precision and recall.")
            st.write(f"**ROC AUC ({roc_auc:.3f}):** How well the model distinguishes between the two classes. A value closer to 1 is better.")

        if model:
            st.markdown("---")
            st.subheader("Feature Importance:")
            if hasattr(model, 'feature_importances_'):
                feature_importance_df = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                fig_importance = px.bar(feature_importance_df.head(10), x='Importance', y='Feature', orientation='h',
                                        title='Top 10 Feature Importances',
                                        labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'})
                fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)

                st.write("This chart shows which features (predictors) the Gradient Boosting model considered most influential in making its predictions. Higher 'Importance' means the feature had a greater impact.")
            else:
                st.info("Feature importances are not available for this model type.")

            # Model Export Section
            st.markdown("---")
            st.subheader("Export Trained Model for Prediction:")
            export_data = {
                'model': model,
                'feature_names': X_train.columns.tolist(), # Features after one-hot encoding
                'target_column': target_col,
                'is_regression': is_regression,
                'original_predictor_cols': predictor_cols, # Original predictor names
                'categorical_unique_values': categorical_unique_values, # Original categories for dropdowns
                'one_hot_encoded_feature_map': one_hot_encoded_feature_map # Map for reconstruction
            }
            if is_binary_classification and le is not None:
                export_data['label_encoder'] = le

            # Convert to bytes for download
            pickled_model = pickle.dumps(export_data)

            st.download_button(
                label="Download Gradient Boosting Model (.pkl)",
                data=pickled_model,
                file_name=f"gradient_boosting_model_{target_col}.pkl",
                mime="application/octet-stream"
            )
            st.info("You can download this model and metadata to use it for making predictions on new, unseen data in a separate application.")

# ---- MAIN APP FUNCTION ----
def main():
    st.title("📊 Basic Stats Explorer")
    st.write("Upload your data (CSV or Excel) and let's explore it together! This app helps you understand your data without needing to be a stats expert.")

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.success("File loaded successfully!")
            st.dataframe(df)
            st.write(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

            num_cols, cat_cols, dt_cols = detect_columns(df)
            all_dataframe_cols = df.columns.tolist() # All columns for selection

            st.sidebar.header("Choose Analysis Sections")
            show_layman = st.sidebar.checkbox("📝 Layman-Friendly Summary", True)
            show_ttest = st.sidebar.checkbox("🔬 Two-Group Comparison (T-Test)", True)
            show_hist_kde = st.sidebar.checkbox("📊 Histogram & KDE Plots", True)
            show_boxplot = st.sidebar.checkbox("📦 Box Plots", True)
            show_ecdf = st.sidebar.checkbox("📈 ECDF Plots", True)
            show_corr = st.sidebar.checkbox("🔗 Correlation Matrix", True)
            show_outliers = st.sidebar.checkbox("🔍 Outlier Detection", True)
            show_quad = st.sidebar.checkbox("🍀 Quadrant Analysis", True)
            show_reg = st.sidebar.checkbox("📈 Regression Analysis", True)
            # New checkboxes for Random Forest and Gradient Boosting
            show_random_forest = st.sidebar.checkbox("🌳 Random Forest Analysis", True)
            show_gradient_boosting = st.sidebar.checkbox("🚀 Gradient Boosting Analysis", True)


            # Section 1: Layman Summary
            if show_layman:
                render_layman_summary(df, num_cols, cat_cols, dt_cols)

            # Section 2: Two-Group Comparison (T-Test)
            if show_ttest:
                try:
                    render_ttest(df, num_cols, cat_cols)
                except Exception as e:
                    st.error(f"Error performing T-Test: {e}")

            # Section 3: Histogram & KDE
            if show_hist_kde:
                ## st.header("📊 Histogram & KDE Plots")
                for col in num_cols:
                    ## st.subheader(f"Distribution — {col}")
                    try:
                        render_histogram_with_kde_groupby(df, [col], all_dataframe_cols) # Pass as list for consistency
                    except Exception as e:
                        st.error(f"Error plotting histogram and KDE for {col}: {e}")

            # Section 4: Box Plots
            if show_boxplot:
                ## st.header("📦 Box Plots")
                for col in num_cols:
                    ## st.subheader(f"Box Plot — {col}")
                    try:
                        render_boxplot(df, [col], all_dataframe_cols) # Pass as list for consistency
                    except Exception as e:
                        st.error(f"Error plotting boxplot for {col}: {e}")

            # Section 5: ECDF
            if show_ecdf:
                ## st.header("📈 ECDF Plots")
                for col in num_cols:
                    ## st.subheader(f"ECDF — {col}")
                    try:
                        render_ecdf(df, [col]) # Pass as list for consistency
                    except Exception as e:
                        st.error(f"Error plotting ECDF for {col}: {e}")

            # Section 6: Correlation
            if show_corr:
                ## st.header("🔗 Correlation Matrix")
                try:
                    render_correlation(df, num_cols)
                except Exception as e:
                    st.error(f"Error plotting correlation matrix: {e}")

            # Section 7: Outlier Detection
            if show_outliers:
                try:
                    render_outliers(df, num_cols)
                except Exception as e:
                    st.error(f"Error detecting outliers: {e}")

            # Section 8: Quadrant
            if show_quad:
                ## st.header("🍀 Quadrant Analysis")
                try:
                    render_quadrant_analysis(df, num_cols, all_dataframe_cols)
                except Exception as e:
                    st.error(f"Error in quadrant analysis: {e}")

            # Section 9: Regression
            if show_reg:
                try:
                    render_regression(df, num_cols)
                except Exception as e:
                    st.error(f"Error running regression analysis: {e}")

            # New Section: Random Forest
            if show_random_forest:
                try:
                    render_random_forest(df, num_cols, cat_cols, all_dataframe_cols)
                except Exception as e:
                    st.error(f"Error running Random Forest analysis: {e}")

            # New Section: Gradient Boosting
            if show_gradient_boosting:
                try:
                    render_gradient_boosting(df, num_cols, cat_cols, all_dataframe_cols)
                except Exception as e:
                    st.error(f"Error running Gradient Boosting analysis: {e}")


        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Please ensure your file is a valid CSV or Excel format and try again.")

if __name__ == "__main__":
    main()
