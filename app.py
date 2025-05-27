import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# App title
st.set_page_config(page_title="Basic Stats Explorer", layout="wide")
st.title("📊 Basic Stats Explorer")

# Sidebar - file uploader and options
st.sidebar.header("1. Upload Data")
uploader = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Sidebar - options
st.sidebar.header("2. Display Options")
show_structure = st.sidebar.checkbox("Show Data Structure", value=True)
show_layman = st.sidebar.checkbox("Show Layman Summary", value=True)
show_details = st.sidebar.checkbox("Show Detailed Stats & Charts", value=False)
show_corr = st.sidebar.checkbox("Show Correlation Matrix", value=False)

if uploader is not None:
    # Load DataFrame
    if uploader.name.endswith(".csv"):
        df = pd.read_csv(uploader)
    else:
        df = pd.read_excel(uploader)

    # Detect column types
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    dt_cols  = df.select_dtypes(include=['datetime','datetimetz']).columns.tolist()

    # 1) Data Structure Overview
    if show_structure:
        st.subheader("🔍 Data Structure Overview")
        st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
        st.write(f"- Numeric columns ({len(num_cols)}): {num_cols}")
        st.write(f"- Categorical columns ({len(cat_cols)}): {cat_cols}")
        if dt_cols:
            st.write(f"- Datetime columns ({len(dt_cols)}): {dt_cols}")
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.table(missing.to_frame("n_missing"))
        else:
            st.write("✅ No missing values detected.")

    # 2) Layman Summary
    if show_layman:
        st.subheader("✏️ Layman-Friendly Summary")
        for col in num_cols:
            data = df[col].dropna()
            if data.empty:
                continue
            mean = data.mean(); median = data.median()
            minimum, maximum = data.min(), data.max()
            span = maximum - minimum; std = data.std()
            def fmt(x): return f"{x:,.1f}" if abs(x) < 1e4 else f"{x:,.0f}"
            mean_fmt, med_fmt, min_fmt, max_fmt, span_fmt, std_fmt = map(
                fmt, [mean, median, minimum, maximum, span, std]
            )
            ratio = std / abs(mean) if mean != 0 else 0
            spread_desc = "quite spread out" if ratio > 0.5 else "moderately spread" if ratio > 0.2 else "tightly clustered"
            st.write(
                f"> **{col}** ranges from **{min_fmt}** to **{max_fmt}** (span **{span_fmt}** units), "
                f"avg **{mean_fmt}**, median **{med_fmt}**; variability **{std_fmt}** so values are {spread_desc}."
            )
        for col in cat_cols:
            vc = df[col].dropna().value_counts(normalize=True)
            if vc.empty: continue
            top, pct = vc.idxmax(), vc.max() * 100; unique = df[col].nunique()
            st.write(f"> **{col}** is mostly **{top}** ({pct:.0f}%) across **{unique}** unique values.")
        for col in dt_cols:
            dates = df[col].dropna().sort_values()
            if dates.empty: continue
            start, end = dates.min().date(), dates.max().date()
            days = (end - start).days
            st.write(f"> **{col}** spans from **{start}** to **{end}** ({days} days).")
        text_cols = [c for c in cat_cols if df[c].dropna().astype(str).map(len).mean() > 20]
        for col in text_cols:
            n = df[col].dropna().shape[0]
            st.write(f"> **{col}** contains free-text for **{n}** records; consider word-clouds next.")

    # 3) Detailed Stats & Charts
    if show_details:
        st.subheader("📈 Detailed Statistics & Charts")
        # Numeric summary table
        if num_cols:
            st.write("**Numeric summary**")
            st.dataframe(df[num_cols].describe().T)
        # Categorical distributions
        for col in cat_cols:
            st.write(f"**{col} - value counts**")
            counts = df[col].value_counts()
            fig, ax = plt.subplots()
            bars = ax.bar(counts.index.astype(str), counts.values)
            ax.set_ylabel('Count'); ax.set_title(f"{col} Distribution")
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{int(h)}', xy=(bar.get_x()+bar.get_width()/2, h),
                            xytext=(0,3), textcoords='offset points', ha='center')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        # Numeric histograms with KDE overlay on twin y-axis
        for col in num_cols:
            st.write(f"**{col} - histogram + KDE**")
            data = df[col].dropna()
            fig, ax1 = plt.subplots()
            # Histogram on primary axis (counts)
            counts, bins, patches = ax1.hist(data, bins=10, edgecolor='black', alpha=0.6)
            ax1.set_ylabel('Count')
            ax1.set_xlabel(col)
            # Annotate counts
            for bar, count in zip(patches, counts):
                height = count
                ax1.annotate(f'{int(height)}', xy=(bar.get_x()+bar.get_width()/2, height),
                             xytext=(0,3), textcoords='offset points', ha='center')
            # KDE on secondary axis (density)
            ax2 = ax1.twinx()
            kde = gaussian_kde(data)
            x_vals = np.linspace(data.min(), data.max(), 200)
            ax2.plot(x_vals, kde(x_vals), linewidth=2)
            ax2.set_ylabel('Density')
            ax1.set_title(f"{col} Distribution & KDE")
            st.pyplot(fig)

            # Layman explanation
            if counts.sum() > 0:
                idx = counts.argmax()
                bin_start, bin_end = bins[idx], bins[idx+1]
                pct = counts[idx] / counts.sum() * 100
                skew = data.skew()
                if skew > 0.5:
                    shape = 'a right-skewed shape (long tail to the right)'
                elif skew < -0.5:
                    shape = 'a left-skewed shape (long tail to the left)'
                else:
                    shape = 'a fairly symmetric shape'
                st.write(
                    f"> 📊 About **{pct:.0f}%** of **{col}** values fall between {bin_start:.1f} and {bin_end:.1f}. "
                    f"The distribution shows {shape}."
                )

    # 4) Correlation matrix
    if show_corr and len(num_cols) > 1:
        st.subheader("🔗 Correlation Matrix")
        corr = df[num_cols].corr()
        fig, ax = plt.subplots()
        cax = ax.matshow(corr)
        fig.colorbar(cax)
        ticks = np.arange(len(num_cols))
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(num_cols, rotation=90); ax.set_yticklabels(num_cols)
        st.pyplot(fig)
else:
    st.info("Please upload a file to begin analysis.")
