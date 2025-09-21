from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px   # <-- added
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title="AiSPRY Forecasting Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)
st.title("AiSPRY Transaction Data")
st.markdown("Prototype v0.1 Group 9")

# -------------------------
# Data loading helpers
# -------------------------
@st.cache_data
def load_data(path: str):
    data = pd.read_csv(path)
    return data

# load main transactions file
df = load_data("./Data/hospital_transactions_2023_2024.csv")

# Try loading vendor lead time if available (optional)
@st.cache_data
def load_vendor_lead_time(path: str = "./Data/vendor_average_lead_time.csv"):
    try:
        vdf = pd.read_csv(path)
        return vdf
    except Exception:
        return None

vendor_lead_df = load_vendor_lead_time("./Data/vendor_average_lead_time.csv")

# --- Sidebar ---
with st.sidebar:
    st.header("Filter Transactions")

    # Ensure transaction_date is datetime
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    else:
        st.error("Column 'transaction_date' not found in dataset.")
        st.stop()

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
    # Use defensive coding in case columns missing
    if "generic_name" in filtered_df.columns:
        generic_options = sorted(filtered_df["generic_name"].dropna().unique())
        generic_selected = st.multiselect("Select Generic Name(s)", generic_options)
    else:
        generic_selected = []
        st.warning("Column 'generic_name' not found — can't filter by SKU/generic.")

    # Apply generic filter before cascading
    if generic_selected:
        filtered_df = filtered_df[filtered_df["generic_name"].isin(generic_selected)]

    if "vendor" in filtered_df.columns:
        vendor_options = sorted(filtered_df["vendor"].dropna().unique())
        vendor_selected = st.multiselect("Select Vendor(s)", vendor_options)
    else:
        vendor_selected = []
        st.warning("Column 'vendor' not found — vendor-based charts will be limited.")

    if vendor_selected:
        filtered_df = filtered_df[filtered_df["vendor"].isin(vendor_selected)]

    if "department" in filtered_df.columns:
        dept_options = sorted(filtered_df["department"].dropna().unique())
        dept_selected = st.multiselect("Select Department(s)", dept_options)
    else:
        dept_selected = []
        st.warning("Column 'department' not found — department-based charts will be limited.")

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

# -------------------------
# Metric / Gauge helpers
# -------------------------
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

# -------------------------
# Compute high-level metrics
# -------------------------
# defensive checks for required columns
if "quantity_requested" not in filtered_df.columns:
    st.error("Column 'quantity_requested' not found. Many metrics require this column.")
    st.stop()

total_requested = filtered_df["quantity_requested"].sum()
unfulfilled_requested = 0
if "event" in filtered_df.columns:
    unfulfilled_requested = filtered_df.loc[filtered_df["event"] == "UNFULFILLED_REQUEST", "quantity_requested"].sum()
else:
    # If no event column, try an alternative (e.g., status or fulfilled flag)
    unfulfilled_requested = 0

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
    plot_metric("Number of Fulfilled Requests", int(fulfilled_requested))

with col2:
    plot_metric("Number of Unfulfilled Requests", int(unfulfilled_requested))

with col3:
    plot_gauge(
        indicator_number=round(bounce_rate, 2),
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
        indicator_number=round(fulfilment_rate, 2),
        indicator_color="green",
        indicator_suffix="%",
        indicator_title="Fulfilment Rate",
        max_bound=100,
        height=350,
        number_font_size=50,
        title_font_size=24
    )

# -------------------------
# EDA section starts here
# -------------------------
st.markdown("---")
st.header("Exploratory Data Analysis (EDA)")

# helper functions
def top_n_by_requests(df, n=10):
    if "generic_name" in df.columns:
        top = (
            df.groupby("generic_name")["quantity_requested"]
            .sum()
            .reset_index()
            .sort_values("quantity_requested", ascending=False)
            .head(n)
        )
        return top
    else:
        return pd.DataFrame(columns=["generic_name", "quantity_requested"])

def requests_by_period(df, period="W"):
    # requires transaction_date, quantity_requested
    ts = (
        df.set_index("transaction_date")
        .resample(period)["quantity_requested"]
        .sum()
        .reset_index()
    )
    ts.rename(columns={"quantity_requested": "total_requested"}, inplace=True)
    return ts

# Tabs for EDA
tab_overview, tab_by_entity, tab_unfulfilled, tab_vendor = st.tabs(
    ["Overview", "By Vendor / Department", "Unfulfilled Analysis", "Vendor Performance"]
)

# ----- Overview tab -----
with tab_overview:
    st.subheader("Top Generics by Quantity Requested")
    top_gens = top_n_by_requests(filtered_df, n=12)
    if not top_gens.empty:
        fig_top = go.Figure(
            go.Bar(
                x=top_gens["quantity_requested"],
                y=top_gens["generic_name"],
                orientation="h",
                marker=dict(color="rgba(50,130,200,0.7)"),
                text=top_gens["quantity_requested"],
                textposition="auto",
            )
        )
        fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=120, r=20, t=40, b=40), height=450)
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No 'generic_name' information available to show Top Generics.")

    st.markdown("### Requests over time (weekly)")
    ts_week = requests_by_period(filtered_df, period="W")
    if not ts_week.empty:
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts_week["transaction_date"], y=ts_week["total_requested"], mode="lines+markers", name="Requested"))
        # also plot unfulfilled trend if available
        if "event" in filtered_df.columns:
            unfulfilled_ts = (
                filtered_df[filtered_df["event"] == "UNFULFILLED_REQUEST"]
                .set_index("transaction_date")
                .resample("W")["quantity_requested"]
                .sum()
                .reset_index()
                .rename(columns={"quantity_requested": "unfulfilled"})
            )
            if not unfulfilled_ts.empty:
                fig_ts.add_trace(go.Scatter(x=unfulfilled_ts["transaction_date"], y=unfulfilled_ts["unfulfilled"], mode="lines+markers", name="Unfulfilled"))
        fig_ts.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=420)
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Not enough date data to display time series.")

# ----- By Vendor / Department tab -----
with tab_by_entity:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Top Vendors (by requested qty)")
        if "vendor" in filtered_df.columns:
            vendor_agg = (
                filtered_df.groupby("vendor")["quantity_requested"].sum().reset_index().sort_values("quantity_requested", ascending=False).head(12)
            )
            fig_vendor = px.bar(vendor_agg, x="vendor", y="quantity_requested", title="Top Vendors", text_auto=True)
            fig_vendor.update_layout(xaxis_tickangle=-45, margin=dict(l=20, r=20, t=40, b=120), height=480)
            st.plotly_chart(fig_vendor, use_container_width=True)
        else:
            st.info("No vendor column found in filtered data.")

        st.markdown("### Department breakdown (stacked: Fulfilled vs Unfulfilled)")
        if "department" in filtered_df.columns:
            if "event" in filtered_df.columns:
                dept_event = (
                    filtered_df.groupby(["department", "event"])["quantity_requested"]
                    .sum()
                    .reset_index()
                )
                fig_dept = px.bar(dept_event, x="department", y="quantity_requested", color="event", title="Department x Event (stacked)")
                fig_dept.update_layout(xaxis_tickangle=-45, margin=dict(l=20, r=20, t=40, b=120), height=480)
                st.plotly_chart(fig_dept, use_container_width=True)
            else:
                dept_agg = (
                    filtered_df.groupby("department")["quantity_requested"].sum().reset_index().sort_values("quantity_requested", ascending=False)
                )
                fig_dept_simple = px.bar(dept_agg, x="department", y="quantity_requested", title="Department Requested Quantities", text_auto=True)
                fig_dept_simple.update_layout(xaxis_tickangle=-45, margin=dict(l=20, r=20, t=40, b=120), height=480)
                st.plotly_chart(fig_dept_simple, use_container_width=True)
        else:
            st.info("No 'department' column found to create department breakdown.")

    with c2:
        st.subheader("Vendor Share (Treemap)")
        if "vendor" in filtered_df.columns:
            vendor_tree = filtered_df.groupby("vendor")["quantity_requested"].sum().reset_index().sort_values("quantity_requested", ascending=False)
            fig_tree = px.treemap(vendor_tree, path=["vendor"], values="quantity_requested", title="Vendor share by quantity requested")
            st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("No vendor info available for treemap.")

# ----- Unfulfilled / Risk tab -----
with tab_unfulfilled:
    st.subheader("Unfulfilled Requests — Who / What / When")

    total_req = int(filtered_df["quantity_requested"].sum())
    unfulfilled_total = 0
    if "event" in filtered_df.columns:
        unfulfilled_total = int(filtered_df.loc[filtered_df["event"] == "UNFULFILLED_REQUEST", "quantity_requested"].sum())
    else:
        unfulfilled_total = 0

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Requested", total_req)
    col_b.metric("Total Unfulfilled", unfulfilled_total)
    col_c.metric("Unfulfilled %", f"{(unfulfilled_total / total_req * 100) if total_req else 0:.2f}%")

    st.markdown("### Top Generics with Unfulfilled Requests")
    if "event" in filtered_df.columns and "generic_name" in filtered_df.columns:
        unfulfilled_by_gen = (
            filtered_df[filtered_df["event"] == "UNFULFILLED_REQUEST"]
            .groupby("generic_name")["quantity_requested"]
            .sum()
            .reset_index()
            .sort_values("quantity_requested", ascending=False)
            .head(15)
        )
        if not unfulfilled_by_gen.empty:
            fig_unf = px.bar(unfulfilled_by_gen, x="quantity_requested", y="generic_name", orientation="h", title="Generics — unfulfilled requests", text_auto=True)
            fig_unf.update_layout(margin=dict(l=120, r=20, t=40, b=40), height=450)
            st.plotly_chart(fig_unf, use_container_width=True)
        else:
            st.info("No unfulfilled requests in the current filter window.")
    else:
        st.info("Either `event` or `generic_name` column missing — cannot compute unfulfilled by generic.")

    st.markdown("### Table: Unfulfilled Requests (detailed)")
    if "event" in filtered_df.columns:
        unf_table = filtered_df[filtered_df["event"] == "UNFULFILLED_REQUEST"].sort_values("transaction_date", ascending=False)
        if not unf_table.empty:
            show_cols = [c for c in ["transaction_date", "generic_name", "vendor", "department", "quantity_requested"] if c in unf_table.columns]
            st.dataframe(unf_table[show_cols].reset_index(drop=True))
        else:
            st.write("No unfulfilled rows to show.")
    else:
        st.write("No `event` column found — unfulfilled table unavailable.")

# ----- Vendor Performance (updated) -----
with tab_vendor:
    st.subheader("Vendor Performance & Lead Times")

    # If vendor lead times file is present, show it
    if vendor_lead_df is not None:
        st.markdown("**Vendor average lead times (from vendor_average_lead_time.csv)**")
        st.dataframe(vendor_lead_df)

        # Plot vendor lead times as a bar chart (uses actual column name)
        if "vendor" in vendor_lead_df.columns and "average_lead_time_days" in vendor_lead_df.columns:
            st.markdown("### Average Lead Time by Vendor")
            fig_lead = px.bar(
                vendor_lead_df.sort_values("average_lead_time_days", ascending=True),
                x="vendor",
                y="average_lead_time_days",
                title="Average Vendor Lead Time (Days)",
                text="average_lead_time_days"
            )
            fig_lead.update_layout(xaxis_tickangle=-45, margin=dict(l=20, r=20, t=40, b=120), height=420)
            st.plotly_chart(fig_lead, use_container_width=True)
        else:
            st.info("Vendor lead time file missing columns 'vendor' and/or 'average_lead_time_days'.")

        # merge lead times with fulfillment rate per vendor if possible
        if "vendor" in filtered_df.columns:
            # compute vendor-level requested and unfulfilled quantities
            if "event" in filtered_df.columns:
                vendor_perf = (
                    filtered_df.groupby("vendor").agg(
                        total_requested=("quantity_requested", "sum"),
                        unfulfilled_qty=("quantity_requested", lambda x: x[filtered_df.loc[x.index, "event"] == "UNFULFILLED_REQUEST"].sum() if "event" in filtered_df.columns else 0)
                    ).reset_index()
                )
            else:
                vendor_perf = (
                    filtered_df.groupby("vendor").agg(
                        total_requested=("quantity_requested", "sum"),
                        unfulfilled_qty=("quantity_requested", lambda x: 0)
                    ).reset_index()
                )

            # Merge on vendor
            if "vendor" in vendor_lead_df.columns:
                merged = vendor_lead_df.merge(vendor_perf, on="vendor", how="left").fillna(0)
                # compute fulfilment rate per vendor (guard against divide by zero)
                merged["fulfilment_rate_pct"] = merged.apply(
                    lambda row: ((row["total_requested"] - row["unfulfilled_qty"]) / row["total_requested"] * 100)
                    if row["total_requested"] > 0 else 0,
                    axis=1
                )

                st.markdown("### Lead time vs Fulfilment rate")
                # scatter: lead time vs fulfilment rate
                if "average_lead_time_days" in merged.columns:
                    fig_scatter = px.scatter(
                        merged,
                        x="average_lead_time_days",
                        y="fulfilment_rate_pct",
                        size="total_requested",
                        hover_data=["vendor", "location"] if "location" in merged.columns else ["vendor"],
                        title="Vendor Lead Time vs Fulfilment Rate (size = total requested)"
                    )
                    fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=520)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Vendor lead time column 'average_lead_time_days' not found after merge; check column names.")
            else:
                st.info("Vendor lead time table doesn't contain a 'vendor' column to merge on.")
        else:
            st.info("No 'vendor' column in transactions to compute performance.")
    else:
        st.info("No vendor lead time file loaded. If you have it, place it at ./Data/vendor_average_lead_time.csv")

# -------------------------
# End of EDA
# -------------------------
st.markdown("---")
st.caption("EDA section: Visualizations update with sidebar filters. Next: add forecasting and reorder recommendations.")
