from collections import defaultdict
from pathlib import Path
import pickle
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

# -------------------------
# App Configuration
# -------------------------
st.set_page_config(
    page_title="AiSPRY Forecasting Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["📊 Transaction Explorer", "📈 Forecasting Dashboard"]
)

# =========================================================
# PAGE 1 — Transaction Explorer (Original Page)
# =========================================================
if page == "📊 Transaction Explorer":
    st.title("AiSPRY Transaction Data")
    st.markdown("Prototype v0.1 Group 9")

    # -------------------------
    # Data loading helpers
    # -------------------------
    @st.cache_data
    def load_data(path: str):
        data = pd.read_csv(path)
        return data

    df = load_data("./Data/hospital_transactions_2023_2024.csv")

    @st.cache_data
    def load_vendor_lead_time(path: str = "./Data/vendor_average_lead_time.csv"):
        try:
            vdf = pd.read_csv(path)
            return vdf
        except Exception:
            return None

    vendor_lead_df = load_vendor_lead_time("./Data/vendor_average_lead_time.csv")

    # --- Sidebar filters ---
    with st.sidebar:
        st.header("Filter Transactions")

        if "transaction_date" in df.columns:
            df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        else:
            st.error("Column 'transaction_date' not found in dataset.")
            st.stop()

        min_date = df["transaction_date"].min()
        max_date = df["transaction_date"].max()
        date_range = st.date_input(
            "Transaction Date Range",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date,
        )

        filtered_df = df.copy()
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df["transaction_date"] >= pd.to_datetime(start_date))
                & (filtered_df["transaction_date"] <= pd.to_datetime(end_date))
            ]

        # cascading filters
        if "generic_name" in filtered_df.columns:
            generic_options = sorted(filtered_df["generic_name"].dropna().unique())
            generic_selected = st.multiselect("Select Generic Name(s)", generic_options)
        else:
            generic_selected = []
            st.warning("Column 'generic_name' not found — can't filter by SKU/generic.")

        if generic_selected:
            filtered_df = filtered_df[filtered_df["generic_name"].isin(generic_selected)]

        if "vendor" in filtered_df.columns:
            vendor_options = sorted(filtered_df["vendor"].dropna().unique())
            vendor_selected = st.multiselect("Select Vendor(s)", vendor_options)
        else:
            vendor_selected = []
            st.warning("Column 'vendor' not found — vendor-based charts limited.")

        if vendor_selected:
            filtered_df = filtered_df[filtered_df["vendor"].isin(vendor_selected)]

        if "department" in filtered_df.columns:
            dept_options = sorted(filtered_df["department"].dropna().unique())
            dept_selected = st.multiselect("Select Department(s)", dept_options)
        else:
            dept_selected = []
            st.warning("Column 'department' not found — department charts limited.")

        if dept_selected:
            filtered_df = filtered_df[filtered_df["department"].isin(dept_selected)]

    # --- Filter summary ---
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

    # --- Data previews ---
    with st.expander("Filtered Data Preview"):
        st.dataframe(filtered_df)
    with st.expander("Data Preview"):
        st.dataframe(df)

    # --- Metric / Gauge helpers ---
    def plot_metric(label, value, prefix="", suffix=""):
        fig = go.Figure()
        fig.add_trace(go.Indicator(mode="number", value=value, title={"text": label},
                                   number={"prefix": prefix, "suffix": suffix}))
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    def plot_gauge(number, color, suffix, title, max_bound=100):
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=number,
                title={"text": title},
                number={"suffix": suffix},
                gauge={
                    "axis": {"range": [0, max_bound]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, max_bound * 0.5], "color": "#ffcccc"},
                        {"range": [max_bound * 0.5, max_bound * 0.8], "color": "#fff0b3"},
                        {"range": [max_bound * 0.8, max_bound], "color": "#ccffcc"},
                    ],
                },
            )
        )
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # --- Compute metrics ---
    if "quantity_requested" not in filtered_df.columns:
        st.error("Column 'quantity_requested' not found.")
        st.stop()

    total_requested = filtered_df["quantity_requested"].sum()
    if "event" in filtered_df.columns:
        unfulfilled_requested = filtered_df.loc[
            filtered_df["event"] == "UNFULFILLED_REQUEST", "quantity_requested"
        ].sum()
    else:
        unfulfilled_requested = 0

    fulfilment_rate = ((total_requested - unfulfilled_requested) / total_requested * 100) if total_requested else 0
    bounce_rate = (unfulfilled_requested / total_requested * 100) if total_requested else 0
    fulfilled_requested = total_requested - unfulfilled_requested

    col1, col2, col3, col4 = st.columns(4)
    with col1: plot_metric("Fulfilled Requests", int(fulfilled_requested))
    with col2: plot_metric("Unfulfilled Requests", int(unfulfilled_requested))
    with col3: plot_gauge(bounce_rate, "red", "%", "Bounce Rate")
    with col4: plot_gauge(fulfilment_rate, "green", "%", "Fulfilment Rate")

    # --- EDA Tabs ---
    st.markdown("---")
    st.header("Exploratory Data Analysis (EDA)")

    def top_n_by_requests(df, n=10):
        if "generic_name" in df.columns:
            return (df.groupby("generic_name")["quantity_requested"].sum()
                    .reset_index().sort_values("quantity_requested", ascending=False).head(n))
        return pd.DataFrame(columns=["generic_name", "quantity_requested"])

    def requests_by_period(df, period="W"):
        ts = (df.set_index("transaction_date").resample(period)["quantity_requested"].sum().reset_index())
        ts.rename(columns={"quantity_requested": "total_requested"}, inplace=True)
        return ts

    tab_overview, tab_by_entity = st.tabs(["Overview", "By Vendor / Department"])

    with tab_overview:
        st.subheader("Top Generics by Quantity Requested")
        top_gens = top_n_by_requests(filtered_df, 12)
        if not top_gens.empty:
            fig_top = go.Figure(go.Bar(x=top_gens["quantity_requested"], y=top_gens["generic_name"],
                                       orientation="h", text=top_gens["quantity_requested"], textposition="auto"))
            fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)

        st.markdown("### Requests over Time (Weekly)")
        ts_week = requests_by_period(filtered_df, "W")
        if not ts_week.empty:
            fig_ts = go.Figure(go.Scatter(x=ts_week["transaction_date"], y=ts_week["total_requested"],
                                          mode="lines+markers", name="Requested"))
            st.plotly_chart(fig_ts, use_container_width=True)

    with tab_by_entity:
        st.subheader("Vendor Breakdown")
        if "vendor" in filtered_df.columns:
            vendor_agg = (filtered_df.groupby("vendor")["quantity_requested"].sum()
                          .reset_index().sort_values("quantity_requested", ascending=False).head(12))
            fig_vendor = px.bar(vendor_agg, x="vendor", y="quantity_requested", title="Top Vendors")
            st.plotly_chart(fig_vendor, use_container_width=True)

    st.markdown("---")
    st.caption("EDA section: Visualizations update with sidebar filters.")

# ==========================================
# 📈 Page 2: Forecasting Results & Cluster Insights
# ==========================================
elif page == "📈 Forecasting Dashboard":
    import pickle
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    import streamlit as st

    st.title("📈 Forecasting Dashboard")
    st.markdown("Visualize forecast results and cluster consumption proportions.")

    # -------------------------
    # Load Model outputs
    # -------------------------
    sarima_path = "./Data/sarima_outputs_20251101.pkl"
    prophet_path = "./Data/prophet_outputs_20251104.pkl"
    
    # Load SARIMA
    try:
        with open(sarima_path, "rb") as f:
            sarima_outputs = pickle.load(f)
        st.success("✅ SARIMA outputs loaded!")
    except Exception as e:
        st.warning(f"⚠️ Could not load SARIMA results: {e}")
        sarima_outputs = None
    
    # Load Prophet (weekly only)
    try:
        with open(prophet_path, "rb") as f:
            prophet_outputs = pickle.load(f)
        st.success("✅ Prophet outputs loaded!")
    except Exception as e:
        st.warning(f"⚠️ Could not load Prophet results: {e}")
        prophet_outputs = None

    def get_test_data(cluster_id, freq_key, sarima_outputs):
        if sarima_outputs and freq_key in sarima_outputs and cluster_id in sarima_outputs[freq_key]:
            cluster_sarima = sarima_outputs[freq_key][cluster_id]
            test_dates = pd.to_datetime(cluster_sarima.get('dates_test', []))
            test_actuals = cluster_sarima.get('y_test', [])
            return test_dates, test_actuals
        return [], []
    # -------------------------
    # Load Next-Week and Next-Day Predictions (Random Forest & XGBoost)
    # -------------------------
    next_weekly_pred_path = "./Data/next_weekly_predictions.pkl"
    next_daily_pred_path = "./Data/next_day_predictions.pkl"

    next_week_predictions = None
    next_daily_predictions = None

    try:
        with open(next_weekly_pred_path, "rb") as f:
            next_week_predictions = pickle.load(f)
        st.success("✅ Next-week predictions (RF/XGBoost) loaded!")
    except Exception as e:
        st.warning(f"⚠️ Could not load next-week predictions: {e}")
    
    try:
        with open(next_daily_pred_path, "rb") as f:
            next_daily_predictions = pickle.load(f)
        st.success("✅ Next-day predictions (RF/XGBoost) loaded!")
    except Exception as e:
        st.warning(f"⚠️ Could not load next-day predictions: {e}")

    # -------------------------
    # Load proportions and transaction info
    # -------------------------
    prop_path = "./Data/qtyconsumed_proportions_bycluster.csv"
    trans_path = "./Data/hospital_transactions_2023_2024.csv"

    prop_df = pd.read_csv(prop_path)
    trans_df = pd.read_csv(trans_path, parse_dates=['transaction_date'])

    # Merge item_name (and generic_name if available) into proportions
    if 'item_id' in trans_df.columns and 'item_name' in trans_df.columns:
        prop_df = prop_df.merge(
            trans_df[['item_id', 'item_name', 'generic_name']].drop_duplicates(),
            on='item_id',
            how='left'
        )
    else:
        st.warning("Item name mapping not found in transactions.")

    # -------------------------
    # Inline Filter Controls
    # -------------------------
    st.markdown("### 🔧 Forecasting Parameters")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        selected_freq = st.selectbox("Select Frequency", ["Daily", "Weekly"])
        freq_key = selected_freq.lower()  # 'daily' or 'weekly'

    with col_b:
        med_list = (
            prop_df["generic_name"].dropna().unique().tolist()
            if "generic_name" in prop_df.columns
            else prop_df["item_name"].dropna().unique().tolist()
        )
        selected_med = st.selectbox("Select Medicine", med_list if med_list else ["No data"])

    with col_c:
        # Model options depend on frequency
        if selected_freq == "Weekly":
            model_options = ["SARIMA", "Random Forest", "XGBoost", "Prophet"]
        else:  # Daily
            model_options = ["SARIMA", "Random Forest", "XGBoost"]
        selected_model = st.selectbox("Select Model", model_options)

    st.markdown("---")

    # -------------------------
    # Map selected medicine → cluster/item
    # -------------------------
    if "generic_name" in prop_df.columns:
        item_row = prop_df[prop_df['generic_name'] == selected_med].iloc[0]
    else:
        item_row = prop_df[prop_df['item_name'] == selected_med].iloc[0]

    cluster_id = item_row['cluster']
    item_id = item_row['item_id']
    item_prop = item_row['consumption_proportion']
    selected_item_name = item_row.get("item_name", selected_med)

    st.markdown(f"**Selected Item:** {selected_item_name} (ID: {item_id}, Cluster: {cluster_id})")

    # -------------------------
    # Prepare historical data
    # -------------------------
    hist_df = trans_df[trans_df['item_id'] == item_id].copy()
    if selected_freq == "Weekly":
        hist_df = hist_df.groupby(pd.Grouper(key='transaction_date', freq='W-SUN'))['quantity_consumed'].sum().reset_index()
    else:  # Daily
        hist_df = hist_df.groupby('transaction_date')['quantity_consumed'].sum().reset_index()
    
    hist_dates = hist_df['transaction_date']
    hist_values = hist_df['quantity_consumed']

    # -------------------------
    # Prepare forecast from cluster based on selected model and frequency
    # -------------------------
    forecast_dates = []
    item_forecast_values = []
    
    if selected_model == "SARIMA":
        if sarima_outputs and freq_key in sarima_outputs and cluster_id in sarima_outputs[freq_key]:
            cluster_forecast = sarima_outputs[freq_key][cluster_id]['forecast']
            forecast_dates = sarima_outputs[freq_key][cluster_id]['dates_test']
            item_forecast_values = np.array(cluster_forecast) * item_prop
        else:
            st.warning(f"No SARIMA forecast available for {freq_key} cluster {cluster_id}")
    
    elif selected_model in ["Random Forest", "XGBoost"]:
        # Choose the correct predictions dictionary based on frequency
        if selected_freq == "Weekly":
            predictions_dict = next_week_predictions
        else:  # Daily
            predictions_dict = next_daily_predictions
        
        if predictions_dict and cluster_id in predictions_dict and selected_model in predictions_dict[cluster_id]:
            model_data = predictions_dict[cluster_id][selected_model]
            # Check if it's the new format (with y_pred and transaction_date)
            if isinstance(model_data, pd.DataFrame):
                forecast_dates = pd.to_datetime(model_data["transaction_date"])
                item_forecast_values = np.array(model_data["y_pred"]) * item_prop            
            
            elif isinstance(model_data, dict):
                # Filter out the 'nan' at the end of the list
                model_data = model_data.get('h1')
                cluster_forecast = np.array([x for x in model_data if not pd.isna(x)])
                item_forecast_values = cluster_forecast * item_prop
                
                # Fetch corresponding dates from SARIMA output for the test period
                test_dates, _ = get_test_data(cluster_id, freq_key, sarima_outputs)
                if len(test_dates) == len(cluster_forecast):
                    forecast_dates = test_dates
                else:
                    st.warning(f"Prediction length ({len(cluster_forecast)}) does not match SARIMA test date length ({len(test_dates)}). Cannot plot {selected_model} forecast correctly.")
                    item_forecast_values = []
            
            else:
                st.warning(f"Unexpected format for {selected_model} predictions")
        else:
            st.warning(f"No {selected_model} forecast found for {freq_key} cluster {cluster_id}")
    
    elif selected_model == "Prophet":
        if selected_freq == "Weekly" and prophet_outputs:
            if "weekly" in prophet_outputs and cluster_id in prophet_outputs["weekly"]:
                cluster_forecast = prophet_outputs["weekly"][cluster_id]['forecast']
                forecast_dates = pd.to_datetime(prophet_outputs["weekly"][cluster_id]['dates_test'])
                item_forecast_values = np.array(cluster_forecast) * item_prop
            else:
                st.warning(f"No Prophet forecast available for cluster {cluster_id}")
        else:
            st.warning("Prophet is only available for weekly forecasts")

    # -------------------------
    # Plot Actual vs Forecast
    # -------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_values,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue')
    ))
    if len(forecast_dates) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=item_forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))

    fig.update_layout(
        title=f"{selected_item_name} - Actual vs Forecast ({selected_freq}, {selected_model})",
        xaxis_title="Date",
        yaxis_title="Quantity Consumed",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)