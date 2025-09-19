from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import altair as alt
import pandas as pd


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title = "AiSPRY Forecasting Dashboard",
    page_icon = ":bar_chart:",
    layout="wide"
    )
st.title("AiSPRY Transaction Data")
st.markdown("Prototype v0.1 Group 9")

@st.cache_data
def load_data(path: str):
    data = pd.read_csv(path)
    return data

df = load_data("./Data/hospital_transactions_2023_2024.csv")

# --- Sidebar ---
with st.sidebar:
    st.header("Filter Transactions")

    # Ensure transaction_date is datetime
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    # Date range filter
    min_date = df["transaction_date"].min()
    max_date = df["transaction_date"].max()
    date_range = st.date_input(
        "Transaction Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date,
    )

    # Step 1: Apply date filter first (base filter for others)
    filtered_df = df.copy()
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df["transaction_date"] >= pd.to_datetime(start_date))
            & (filtered_df["transaction_date"] <= pd.to_datetime(end_date))
        ]

    # Step 2: Filters that depend on current subset
    generic_options = sorted(filtered_df["generic_name"].dropna().unique())
    generic_selected = st.multiselect("Select Generic Name(s)", generic_options)

    # Apply generic filter before cascading
    if generic_selected:
        filtered_df = filtered_df[filtered_df["generic_name"].isin(generic_selected)]

    vendor_options = sorted(filtered_df["vendor"].dropna().unique())
    vendor_selected = st.multiselect("Select Vendor(s)", vendor_options)

    if vendor_selected:
        filtered_df = filtered_df[filtered_df["vendor"].isin(vendor_selected)]

    dept_options = sorted(filtered_df["department"].dropna().unique())
    dept_selected = st.multiselect("Select Department(s)", dept_options)

    if dept_selected:
        filtered_df = filtered_df[filtered_df["department"].isin(dept_selected)]

# --- Show filter summary ---
with st.container():
    st.subheader("Active Filters")

    filters_applied = []

    if len(date_range) == 2:
        filters_applied.append(f"📅 Date: {date_range[0]} → {date_range[1]}")
    if generic_selected:
        filters_applied.append(f"💊 Generic Name(s): {', '.join(generic_selected)}")
    if vendor_selected:
        filters_applied.append(f"🏢 Vendor(s): {', '.join(vendor_selected)}")
    if dept_selected:
        filters_applied.append(f"🏥 Department(s): {', '.join(dept_selected)}")

    if filters_applied:
        for f in filters_applied:
            st.markdown(f"- {f}")
    else:
        st.markdown("No filters applied — showing all data.")

# --- Show Filtered Data ---
with st.expander("Filtered Data Preview"):
    st.dataframe(filtered_df)
# --- Show Data Preview ---
with st.expander("Data Preview"):
    st.dataframe(df)
    
def plot_metric(label, value, prefix="", suffix=""):
    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=value,
            title={"text": label},
            number={"prefix": prefix, "suffix": suffix}
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def plot_gauge(
    indicator_number,
    indicator_color,
    indicator_suffix,
    indicator_title,
    max_bound=100,
    height=350,        # make the whole gauge taller
    number_font_size=40,  # bigger numeric value
    title_font_size=20     # bigger title
):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=indicator_number,
            title={"text": indicator_title, "font": {"size": title_font_size}, "align": "center"},
            number={"suffix": indicator_suffix, "font": {"size": number_font_size}},
            gauge={
                "axis": {"range": [0, max_bound]},
                "bar": {"color": indicator_color},
                "steps": [
                    {"range": [0, max_bound * 0.5], "color": "#ffcccc"},
                    {"range": [max_bound * 0.5, max_bound * 0.8], "color": "#fff0b3"},
                    {"range": [max_bound * 0.8, max_bound], "color": "#ccffcc"},
                ],
            },
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=height)
    st.plotly_chart(fig, use_container_width=True)

    
# Calculate fulfilment rate
total_requested = filtered_df["quantity_requested"].sum()
unfulfilled_requested = filtered_df.loc[filtered_df["event"] == "UNFULFILLED_REQUEST", "quantity_requested"].sum()

if total_requested > 0:
    fulfilment_rate = ((total_requested - unfulfilled_requested) / total_requested) * 100
else:
    fulfilment_rate = 0

# --- Compute metrics ---
fulfilled_requested = total_requested - unfulfilled_requested

# Bounce Rate (%)
if total_requested > 0:
    bounce_rate = (unfulfilled_requested / total_requested) * 100
else:
    bounce_rate = 0

# --- Display Cards ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    plot_metric("Number of Fulfilled Requests", fulfilled_requested)

with col2:
    plot_metric("Number of Unfulfilled Requests", unfulfilled_requested)

with col3:
    plot_gauge(
        indicator_number=bounce_rate,
        indicator_color="red",
        indicator_suffix="%",
        indicator_title="Bounce Rate",
        max_bound=100,
        height=350,
        number_font_size=50,
        title_font_size=24
    )

with col4:
    plot_gauge(
    indicator_number=fulfilment_rate,
    indicator_color="green",
    indicator_suffix="%",
    indicator_title="Fulfilment Rate",
    max_bound=100,
    height=350,
    number_font_size=50,
    title_font_size=24
    )
