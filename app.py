import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import plotly.express as px  # new import for interactive ECDFs
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import ttest_ind

st.set_page_config(page_title="Basic Stats Explorer", layout="wide")

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
    st.header("📝 Layman-Friendly Summary")
    epsilon = 1e-6

    for col in num_cols:
        data = df[col].dropna()
        if data.empty:
            continue
        mn = data.mean()
        med = data.median()
        std = data.std()
        rng = data.max() - data.min()
        ratio = std / (abs(mn) + epsilon)

        spread_desc = (
            "tightly clustered" if ratio < 0.1 else
            "moderately spread" if ratio < 0.5 else
            "quite spread out"
        )

        st.write(f"> **{col}** — Count: {len(data)}, Mean: {mn:.1f}, Median: {med:.1f}")
        st.write(f"> Range: {data.min():.1f} to {data.max():.1f} (span = {rng:.1f} units); data are {spread_desc} (σ ≈ {std:.1f})")

    for col in cat_cols:
        vc = df[col].dropna().value_counts(normalize=True)
        if vc.empty:
            continue
        top, pct = vc.index[0], vc.iloc[0]*100
        uniq = df[col].nunique()
        st.write(f"> **{col}** — Top category: **{top}** ({pct:.0f}%); {uniq} unique values")

    for col in dt_cols:
        dates = df[col].dropna().sort_values()
        if dates.empty:
            continue
        start, end = dates.min().date(), dates.max().date()
        days = (end - start).days
        st.write(f"> **{col}** — From **{start}** to **{end}**, spanning **{days} days**")
        
# ---- T TEST ----
def render_ttest(df, num_cols, cat_cols):
    """
    Renders a two-sample t-test for any binary categorical column vs. a numeric column.
    """
    st.header("⚖️ Two-Group Comparison (T-Test)")

    # 1) Pick the binary category and numeric column
    cat = st.selectbox("Choose a categorical column (exactly 2 levels)", cat_cols, key="ttest_cat")
    num = st.selectbox("Choose a numeric column to compare", num_cols, key="ttest_num")

    # 2) Check levels
    levels = df[cat].dropna().unique()
    if len(levels) != 2:
        st.error(f"Column **{cat}** has {len(levels)} levels; please pick one with exactly 2 groups.")
        return

    # 3) Extract the two samples
    grp1 = df.loc[df[cat] == levels[0], num].dropna()
    grp2 = df.loc[df[cat] == levels[1], num].dropna()

    # Guard against too-small samples
    if len(grp1) < 2 or len(grp2) < 2:
        st.error("Each group needs at least 2 observations.")
        return

    # 4) Run Welch’s t-test
    tstat, pval = ttest_ind(grp1, grp2, equal_var=False)

    # 5) Show results
    st.subheader("T-Test Results")
    st.write(f"- Comparing **{levels[0]}** vs **{levels[1]}** on **{num}**")
    st.write(f"- t-statistic = {tstat:.2f}")
    st.write(f"- p-value = {pval:.3f}")

    # 6) Layman interpretation
    sig = "a statistically significant" if pval < 0.05 else "no statistically significant"
    st.markdown(f"""
    > **Interpretation:**  
    > There is **{sig}** difference in the mean **{num}** between **{levels[0]}** and **{levels[1]}** (p = {pval:.3f}).
    """)

    # Optional: show group means
    mean1, mean2 = grp1.mean(), grp2.mean()
    st.write(f"- Mean of **{levels[0]}** = {mean1:.2f}")
    st.write(f"- Mean of **{levels[1]}** = {mean2:.2f}")

# ---- REGRESSION ----
def render_regression(df, num_cols):
    st.header("📈 Regression Analysis")

    # 1) Pick target & predictors
    y_col = st.sidebar.selectbox("Choose target (predict)", num_cols, key="reg_y")
    x_cols = st.sidebar.multiselect(
        "Choose predictors (features)", 
        [c for c in num_cols if c != y_col],
        key="reg_x"
    )
    if not x_cols:
        st.info("Select at least one predictor to run regression.")
        return

    # 2) Multicollinearity check
    corr_preds = df[x_cols].corr().abs()
    high_corr = [
        (i, j, corr_preds.loc[i,j])
        for i in x_cols for j in x_cols if i < j and corr_preds.loc[i,j] > 0.8
    ]
    if high_corr:
        st.warning("⚠️ High multicollinearity detected among predictors:")
        for i, j, rho in high_corr:
            st.write(f"- **{i}** & **{j}**: ρ = {rho:.2f}")

    # 3) Prepare data & fit model
    data = df[[y_col] + x_cols].dropna()
    X = data[x_cols].values
    y = data[y_col].values
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)

    # 4) Metrics
    r2 = r2_score(y, preds)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)

    # 5) Show formula
    terms = [f"{coef:.3f}·{name}" for coef, name in zip(model.coef_, x_cols)]
    formula = f"{y_col} ≈ {model.intercept_:.3f}"
    if terms:
        formula += " + " + " + ".join(terms)
    st.subheader("Model Formula")
    st.code(formula, language="text")

    # 6) Layman model quality
    if r2 >= 0.75:
        quality, advice = "strong", "explains most of the variation"
    elif r2 >= 0.5:
        quality, advice = "moderate", "captures some variation but could improve"
    else:
        quality, advice = "weak", "explains only a small part of the variation"

    # 7) Unit inference (simple heuristic)
    # You can extend this mapping as needed
    unit_map = {
        'Score': 'points',
        'Hours': 'hours',
        'Time': 'units',
    }
    # default to the raw column name if no match
    unit = next((u for k, u in unit_map.items() if k.lower() in y_col.lower()), y_col)

    st.write(f"**R² = {r2:.3f}** ({quality} fit)    **RMSE = {rmse:.3f} {unit}**")
    st.markdown(f"""
> **What this means:**  
> - The model explains about **{r2*100:.1f}%** of the variance in **{y_col}**.  
> - On average, predictions are off by about **{rmse:.1f} {unit}** (the RMSE).  
> - {advice.capitalize()}.
    """)

    # 8) Diagnostics toggle
    if st.sidebar.checkbox("Show diagnostics", key="reg_diag"):
        # Residuals vs. Predicted
        residuals = y - preds
        fig1, ax1 = plt.subplots()
        ax1.scatter(preds, residuals, alpha=0.6, edgecolor='k')
        ax1.axhline(0, color='gray', linestyle='--')
        ax1.set_xlabel("Predicted values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs. Predicted")
        st.pyplot(fig1)

        # Residuals histogram
        fig2, ax2 = plt.subplots()
        ax2.hist(residuals, bins=15, edgecolor='black', alpha=0.7)
        ax2.set_title("Residuals Distribution")
        ax2.set_xlabel("Residual")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
    
# ---- QUADRANT BABY! ----
def render_quadrant_analysis(df, num_cols):
    # Sidebar selectors
    x_col = st.sidebar.selectbox("Choose X-axis", num_cols, key="quad_x")
    y_col = st.sidebar.selectbox("Choose Y-axis", num_cols, key="quad_y")

    st.sidebar.markdown("---")
    chop_opts = ["Mean", "Median", "Custom"]
    x_method = st.sidebar.selectbox(f"{x_col} chop at", chop_opts, key="quad_x_method")
    y_method = st.sidebar.selectbox(f"{y_col} chop at", chop_opts, key="quad_y_method")

    # Determine thresholds
    def get_thresh(col, method):
        if method == "Mean":
            return df[col].mean()
        if method == "Median":
            return df[col].median()
        return st.sidebar.number_input(f"{col} custom threshold", value=float(df[col].mean()))

    x_thresh = get_thresh(x_col, x_method)
    y_thresh = get_thresh(y_col, y_method)

    # Prepare data
    data = df[[x_col, y_col]].dropna().copy()
    # Assign quadrants
    def quad_label(r):
        if   r[x_col] > x_thresh and r[y_col] > y_thresh: return "Q1: high/high"
        if   r[x_col] <= x_thresh and r[y_col] > y_thresh: return "Q2: low/high"
        if   r[x_col] <= x_thresh and r[y_col] <= y_thresh: return "Q3: low/low"
        return "Q4: high/low"
    data["Quadrant"] = data.apply(quad_label, axis=1)

    # Build interactive scatter
    fig = px.scatter(
        data, x=x_col, y=y_col, color="Quadrant",
        title=f"🍀 Quadrant Analysis: {x_col} vs {y_col}",
        labels={x_col: x_col, y_col: y_col},
        symbol="Quadrant", # different symbols per quadrant if desired
        hover_data={x_col: True, y_col: True, "Quadrant": True}
    )
    # Add chop lines
    fig.add_shape(dict(type="line", x0=x_thresh, x1=x_thresh, y0=data[y_col].min(),
                       y1=data[y_col].max(), line=dict(dash="dash", color="gray")))
    fig.add_shape(dict(type="line", y0=y_thresh, y1=y_thresh, x0=data[x_col].min(),
                       x1=data[x_col].max(), line=dict(dash="dash", color="gray")))

    fig.update_layout(legend_title_text="Quadrant",
                      margin=dict(t=50, l=40, r=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Summarize counts
    counts = data["Quadrant"].value_counts().reindex(
        ["Q1: high/high","Q2: low/high","Q3: low/low","Q4: high/low"], fill_value=0
    )
    summary = "\n".join(f"> **{quad}**: {cnt} records" for quad, cnt in counts.items())
    st.markdown(f"**Quadrant counts:**\n\n{summary}")

# ---- HISTOGRAM + KDE ----
def render_histogram_with_kde(df, col):
    
    # Generic histogram + KDE explanation
    st.markdown(
        """
        **How to use this chart:**  
        - Bars show how many records fall into each value range.  
        - The smooth curve overlays the overall “shape” of the distribution.  
        - The tallest bar (or highest peak) marks the most common range of values.  
        - Long tails indicate that a few records lie far from the bulk of the data.
        """
    )

    data = df[col].dropna()
    if data.empty:
        return

    # --- Prepare data ---
    counts, bins = np.histogram(data, bins=10)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # KDE
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 200)
    y_kde = kde(x_vals)

    # --- Build Plotly figure ---
    fig = go.Figure()

    # Histogram on primary y-axis
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        width=(bins[1] - bins[0]) * 0.9,
        name="Count",
        marker_color="lightblue",
        hovertemplate="Range: %{x:.1f}<br>Count: %{y}<extra></extra>"
    ))

    # KDE on secondary y-axis
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_kde,
        name="Density",
        line=dict(color="orange", width=2),
        hovertemplate="x: %{x:.1f}<br>Density: %{y:.3f}<extra></extra>",
        yaxis="y2"
    ))

    # Layout with two y-axes
    fig.update_layout(
        title=f"Histogram & KDE — {col}",
        xaxis_title=col,
        yaxis=dict(title="Count"),
        yaxis2=dict(
            title="Density",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        bargap=0.1,
        template="simple_white",
        margin=dict(t=40, b=20, l=40, r=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Layman Summary ---
    total = len(data)
    modal_idx = counts.argmax()
    modal_range = (bins[modal_idx], bins[modal_idx + 1])
    modal_pct = counts[modal_idx] / total * 100

    mn = data.mean()
    med = data.median()
    std = data.std()
    skew = data.skew()
    if skew > 0.5:
        shape = "Right-skewed: most values are lower with a tail toward higher values."
    elif skew < -0.5:
        shape = "Left-skewed: most values are higher with a tail toward lower values."
    else:
        shape = "Fairly symmetric around the middle."

    summary = f"""
> **Histogram & KDE for {col}:**
> - **Most common range:** {modal_range[0]:.1f}–{modal_range[1]:.1f} ({modal_pct:.0f}% of observations).
> - **Mean:** {mn:.1f}, **Median:** {med:.1f}, **Std Dev (σ):** {std:.1f}.
> - **Shape:** {shape}
"""
    st.markdown(summary)


# ---- BOX PLOT + EXPLANATION (Plotly) ----
def render_boxplot(df, col):
    # Generic box-plot explanation
    st.markdown(
        """
        **How to use this chart:**  
        - The center line of the box is the median (middle) value.  
        - The box edges are the 25th and 75th percentiles (middle 50% of records).  
        - “Whiskers” extend to the typical minimum and maximum; dots beyond them are outliers.  
        - Use this to see at a glance the typical range, the middle point, and any extreme values.
        """
    )

    data = df[col].dropna()
    if data.empty:
        return

    # Build interactive horizontal box plot
    fig = px.box(
        data_frame=data.to_frame(name=col),
        x=col,               # map to x for horizontal orientation
        points="outliers",   # show outlier points
        title=f"Box Plot — {col}",
        labels={col: col}
    )

    # Make the box narrower
    fig.update_traces(
        width=0.2,            # controls box "thickness"
        hovertemplate=(
            f"{col}: %{{x}}<br>"
            "Median: %{median}<br>"
            "Q1: %{q1}<br>"
            "Q3: %{q3}<br>"
            "Lower whisker: %{lowerfence}<br>"
            "Upper whisker: %{upperfence}"
        )
    )
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20))

    st.plotly_chart(fig, use_container_width=True)

    # Layman summary
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    low_cut, high_cut = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out_low = data[data < low_cut]
    out_high = data[data > high_cut]

    explanation = f"""
> **Median = {q2:.1f}**: Half of the observations for **{col}** are {q2:.1f} or less, and half are {q2:.1f} or more.

> **25th–75th percentile = {q1:.1f} to {q3:.1f}** (IQR = {iqr:.1f}): Exactly half of the records fall within this middle range.

> **Outliers**: {len(out_low)} unusually low and {len(out_high)} unusually high values have been detected.
"""
    st.write(explanation)

# ---- ECDF with Plotly + Explanation ----
def render_ecdf(df, col):
    # Generic ECDF explanation
    st.markdown(
        """
        **How to use this chart:**  
        - The curve shows what fraction of records are at or below each x-value.  
        - To find any percentile (e.g. 0.75), hover until the y-axis reads that fraction.  
        - Steep sections mean many records share similar values; flat sections mean gaps.  
        - ECDFs are great for reading exact percentiles and comparing distributions.
        """
    )

    data = df[col].dropna()
    if data.empty:
        return

    # build interactive ECDF
    fig = px.ecdf(
        data_frame=data.to_frame(name=col),
        x=col,
        title=f"ECDF — {col}",
        labels={col: col, "ecdf": "Proportion ≤ x"},
    )
    fig.update_traces(
        hovertemplate=f"{col}: %{{x:.1f}}<br>Proportion: %{{y:.2f}}"
    )
    fig.update_layout(
        xaxis_title=col,
        yaxis_title="Proportion ≤ x",
        hovermode="closest",
        template="simple_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Layman explanation of key percentiles
    p25, p50, p75 = np.percentile(data, [25, 50, 75])
    explanation = (
        f"> **ECDF for {col}** shows what fraction of data is at or below each value.\n\n"
        f"- At **{p25:.1f}**, about **25%** of values are ≤ {p25:.1f}.\n"
        f"- At **{p50:.1f}**, about **50%** of values are ≤ {p50:.1f} (the median).\n"
        f"- At **{p75:.1f}**, about **75%** of values are ≤ {p75:.1f}.\n"
    )
    st.write(explanation)

# ---- CORRELATION MATRIX WITH R² ----
def render_correlation(df, num_cols):
    if len(num_cols) < 2:
        st.info("Need at least two numeric columns for correlation matrix.")
        return

    # Compute Pearson correlation and R²
    corr = df[num_cols].corr()
    r2 = corr**2

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(cax, ax=ax)

    # Set tick labels
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=90)
    ax.set_yticks(range(len(num_cols)))
    ax.set_yticklabels(num_cols)

    # Annotate each cell with “ρ=…\nR²=…”
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            if i == j:
                # perfect correlation on diagonal
                text = f"ρ=1.00\nR²=1.00"
            else:
                text = f"ρ={corr.iloc[i,j]:.2f}\nR²={r2.iloc[i,j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)

    st.pyplot(fig)

# ---- Outlier Detection ----
def render_outliers(df, num_cols):
    """
    For each numeric column, detect values outside 1.5 × IQR and report counts & percentages.
    """
    st.header("🚩 Outlier Detection")
    summary = []
    for col in num_cols:
        data = df[col].dropna()
        if data.empty:
            continue
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        out_low = data[data < lower]
        out_high = data[data > upper]
        n = len(data)
        n_low, n_high = len(out_low), len(out_high)
        pct_low = n_low / n * 100
        pct_high = n_high / n * 100

        summary.append({
            "Column": col,
            "Total": n,
            "Low outliers": f"{n_low} ({pct_low:.1f}%)",
            "High outliers": f"{n_high} ({pct_high:.1f}%)",
            "Lower cutoff": f"{lower:.1f}",
            "Upper cutoff": f"{upper:.1f}"
        })

    if not summary:
        st.write("No numeric columns to analyze.")
        return

    # Display as a table
    out_df = pd.DataFrame(summary)
    st.dataframe(out_df)

# ---- MAIN APP ----
def main():
    st.title("🔎 Data Detective: Basic Stats Explorer")

    # ——— Fun Introduction ———
    st.markdown("""
    Welcome aboard the **Data Detective**! 🕵️‍♂️🔍  

    Imagine having a trusty sidekick who instantly sifts, summarizes, and visualizes **any** dataset you throw at it—no PhD in statistics required. Whether you’re:

    - Curious about how values cluster (histograms & KDE)  
    - Wanting to spot outliers at a glance (box plots)  
    - Eager to know exactly what proportion falls below a threshold (ECDF)  
    - Looking to compare two groups (t-test)  
    - Ready to dive into simple predictions (linear regression)  
    - Or even slice & dice with quadrant analysis…

    …this app has your back.  

    **How to get started:**  
    1. **Upload** your CSV/Excel file.  
    2. **Toggle** the analyses you want in the sidebar.  
    3. **Customize** any settings in the collapsible panels—no confusing stats jargon here.  
    4. **Explore** interactive charts, hover for details, and read plain-English insights.  

    Let’s turn your raw numbers into “aha!” moments—dive in and have fun with your data! 🚀📊
    """)
    st.write("---")
    
    uploaded = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
    if not uploaded:
        st.info("Please upload a file to get started.")
        return

    try:
        df = load_data(uploaded)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    num_cols, cat_cols, dt_cols = detect_columns(df)

    # Sidebar toggles
    st.sidebar.header("Show Sections")
    show_structure = st.sidebar.checkbox("Data Structure Overview", True)
    show_layman = st.sidebar.checkbox("Layman Summary", True)
    show_details = st.sidebar.checkbox("Detailed Stats & Charts", False)
    show_ecdf = st.sidebar.checkbox("ECDF Plots", False)
    show_corr = st.sidebar.checkbox("Correlation Matrix", False)
    show_outliers = st.sidebar.checkbox("Outlier Detection", False)
    show_quad = st.sidebar.checkbox("Quadrant Analysis", False)
    show_reg = st.sidebar.checkbox("Regression Analysis", False)
    show_ttest = st.sidebar.checkbox("Two-Group Comparison (T-Test)", False)

    # Section 1: Data structure
    if show_structure:
        st.header("📂 Data Structure Overview")
        st.write(f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write(f"- Numeric columns ({len(num_cols)}): {num_cols}")
        st.write(f"- Categorical/Text columns ({len(cat_cols)}): {cat_cols}")
        if dt_cols:
            st.write(f"- Datetime columns ({len(dt_cols)}): {dt_cols}")
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.subheader("Missing Values")
            st.table(missing.to_frame("n_missing"))

    # Section 2: Layman summary
    if show_layman:
        render_layman_summary(df, num_cols, cat_cols, dt_cols)

    # Section 3: Detailed stats & charts
    if show_details:
        st.header("📊 Detailed Charts")
        for col in num_cols:
            st.subheader(f"Histogram & KDE — {col}")
            try:
                render_histogram_with_kde(df, col)
            except Exception as e:
                st.error(f"Error plotting histogram for {col}: {e}")

            st.subheader(f"Box Plot — {col}")
            try:
                render_boxplot(df, col)
            except Exception as e:
                st.error(f"Error plotting boxplot for {col}: {e}")

    # Section 4: ECDF
    if show_ecdf:
        st.header("📈 ECDF Plots")
        for col in num_cols:
            st.subheader(f"ECDF — {col}")
            try:
                render_ecdf(df, col)
            except Exception as e:
                st.error(f"Error plotting ECDF for {col}: {e}")

    # Section 5: Correlation
    if show_corr:
        st.header("🔗 Correlation Matrix")
        try:
            render_correlation(df, num_cols)
        except Exception as e:
            st.error(f"Error plotting correlation matrix: {e}")

    # Section X: Outlier Detection
    if show_outliers:
        try:
            render_outliers(df, num_cols)
        except Exception as e:
            st.error(f"Error detecting outliers: {e}")

    # Section X: Quadrant
    if show_quad:
        st.header("🍀 Quadrant Analysis")
        try:
            render_quadrant_analysis(df, num_cols)
        except Exception as e:
            st.error(f"Error in quadrant analysis: {e}")
    
    # Section 7: Regression
    if show_reg:
        try:
            render_regression(df, num_cols)
        except Exception as e:
            st.error(f"Error running regression: {e}")

    # Section 8: Mr T!
    if show_ttest:
        render_ttest(df, num_cols, cat_cols)

if __name__ == "__main__":
    main()
