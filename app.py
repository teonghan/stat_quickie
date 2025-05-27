import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        # Missing values
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.table(missing.to_frame("n_missing"))
        else:
            st.write("✅ No missing values detected.")

    # 2) Layman Summary
    if show_layman:
        st.subheader("✏️ Layman-Friendly Summary")
        # Numeric summaries
        for col in num_cols:
            data = df[col].dropna()
            if data.empty:
                continue
            mean = data.mean()
            median = data.median()
            span = data.max() - data.min()
            mean_fmt = f"{mean:,.1f}" if abs(mean) < 1e4 else f"{mean:,.0f}"
            median_fmt = f"{median:,.1f}" if abs(median) < 1e4 else f"{median:,.0f}"
            span_fmt = f"{span:,.1f}" if abs(span) < 1e4 else f"{span:,.0f}"
            st.write(f"> **{col}** has an average of **{mean_fmt}**, median **{median_fmt}**, spanning **{span_fmt}** units.")
        # Categorical summaries
        for col in cat_cols:
            vc = df[col].dropna().value_counts(normalize=True)
            if vc.empty:
                continue
            top = vc.idxmax()
            pct = vc.max() * 100
            unique = df[col].nunique()
            st.write(f"> **{col}** is mostly **{top}** ({pct:.0f}%) across **{unique}** unique values.")
        # Datetime summaries
        for col in dt_cols:
            dates = df[col].dropna().sort_values()
            if dates.empty:
                continue
            start, end = dates.min().date(), dates.max().date()
            days = (end - start).days
            st.write(f"> **{col}** spans from **{start}** to **{end}** ({days} days).")
        # Free-text detection
        text_cols = [c for c in cat_cols if df[c].dropna().astype(str).map(len).mean() > 20]
        for col in text_cols:
            n = df[col].dropna().shape[0]
            st.write(f"> **{col}** contains free-text for **{n}** records; consider word-clouds or sentiment analysis next.")

    # 3) Detailed Stats & Charts
    if show_details:
        st.subheader("📈 Detailed Statistics & Charts")
        # Numeric descriptive table
        if num_cols:
            st.write("**Numeric summary**")
            desc = df[num_cols].describe().T
            st.dataframe(desc)
        # Categorical frequencies
        for col in cat_cols:
            st.write(f"**{col} - value counts**")
            st.bar_chart(df[col].value_counts())
        # Histograms
        for col in num_cols:
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=10)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    # 4) Correlation matrix
    if show_corr and len(num_cols) > 1:
        st.subheader("🔗 Correlation Matrix")
        corr = df[num_cols].corr()
        fig, ax = plt.subplots()
        cax = ax.matshow(corr)
        fig.colorbar(cax)
        ticks = np.arange(len(num_cols))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(num_cols, rotation=90)
        ax.set_yticklabels(num_cols)
        st.pyplot(fig)

    # End of app
else:
    st.info("Please upload a file to begin analysis.")
