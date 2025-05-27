import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import plotly.express as px  # new import for interactive ECDFs

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

# ---- HISTOGRAM + KDE ----
def render_histogram_with_kde(df, col):
    data = df[col].dropna()
    if data.empty:
        return

    # --- PLOT HISTOGRAM + KDE WITH DUAL AXES (Matplotlib example) ---
    fig, ax1 = plt.subplots()
    counts, bins, _ = ax1.hist(data, bins=10, edgecolor='black', alpha=0.6)
    ax1.set_ylabel("Count")
    ax1.set_xlabel(col)

    # annotate counts
    for i, c in enumerate(counts):
        mid = (bins[i] + bins[i+1]) / 2
        ax1.annotate(f"{int(c)}", xy=(mid, c), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8)

    ax2 = ax1.twinx()
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 200)
    ax2.plot(x_vals, kde(x_vals), color='C1', linewidth=2)
    ax2.set_ylabel("Density")

    st.pyplot(fig)

    # --- LAYMAN SUMMARY ---
    # identify modal bin
    modal_idx = counts.argmax()
    modal_range = (bins[modal_idx], bins[modal_idx+1])
    modal_pct = counts[modal_idx] / len(data) * 100

    mn = data.mean()
    med = data.median()
    std = data.std()
    skew = data.skew()

    # shape descriptor
    if skew > 0.5:
        shape = "right-skewed (more high values)"
    elif skew < -0.5:
        shape = "left-skewed (more low values)"
    else:
        shape = "fairly symmetric"

    explanation = f"""
> **Histogram & KDE for {col}:**
> - Most common range: **{modal_range[0]:.1f}–{modal_range[1]:.1f}** ({modal_pct:.0f}% of observations fall here).
> - Average (mean) ≈ **{mn:.1f}**, typical (median) = **{med:.1f}**, variability (σ) ≈ **{std:.1f}**.
> - Distribution appears **{shape}**.
"""
    st.markdown(explanation)


# ---- BOX PLOT + EXPLANATION (Plotly) ----
def render_boxplot(df, col):
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
        width=0.3,            # controls box "thickness"
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

# ---- CORRELATION MATRIX ----
def render_correlation(df, num_cols):
    if len(num_cols) < 2:
        st.info("Need at least two numeric columns for correlation matrix.")
        return
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(cax)
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=90)
    ax.set_yticks(range(len(num_cols)))
    ax.set_yticklabels(num_cols)
    st.pyplot(fig)

# ---- MAIN APP ----
def main():
    st.title("Basic Stats Explorer")
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
    show_details = st.sidebar.checkbox("Detailed Stats & Charts", True)
    show_ecdf = st.sidebar.checkbox("ECDF Plots", False)
    show_corr = st.sidebar.checkbox("Correlation Matrix", False)

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

if __name__ == "__main__":
    main()
