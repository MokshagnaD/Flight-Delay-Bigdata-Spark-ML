"""
Interactive Big Data Flight Delay Analytics & Machine Learning Web Dashboard
Built with Streamlit for live executive and academic demonstrations.
"""

import json
import os
import sys
import pandas as pd
import streamlit as st

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ml.config import METADATA_PATH, REPORTS_DIR, RF_MODEL_PATH, XGBOOST_MODEL_PATH
from src.ml.inference import FlightDelayPredictor

# Page Setup
st.set_page_config(
    page_title="Flight Delay Big Data Analytics & ML System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .risk-badge-low {
        background-color: #28a745;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-badge-mod {
        background-color: #17a2b8;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-badge-elevated {
        background-color: #fd7e14;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #dc3545;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor(model_type: str = "xgboost"):
    model_path = XGBOOST_MODEL_PATH if model_type == "xgboost" else RF_MODEL_PATH
    return FlightDelayPredictor(model_path=model_path)


@st.cache_data
def load_analytics_data():
    summary_path = os.path.join(REPORTS_DIR, "big_data_analysis_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_model_evaluation():
    eval_path = os.path.join(REPORTS_DIR, "model_evaluation_report.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            return json.load(f)
    return None


# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/airplane-take-off.png", width=64)
st.sidebar.title("Flight Delay ML")
st.sidebar.caption("Big Data Analytics & Prediction Platform")

model_choice = st.sidebar.selectbox(
    "Active ML Model",
    ["XGBoost Classifier (Recommended)", "Random Forest Classifier"],
    index=0
)
selected_model_type = "xgboost" if "XGBoost" in model_choice else "rf"

try:
    predictor = load_predictor(selected_model_type)
    model_ready = True
except Exception as e:
    model_ready = False
    st.sidebar.error(f"Error loading model: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Architecture")
st.sidebar.markdown(r"""
- **Source:** US Bureau of Transportation Statistics (BTS)
- **Scale:** 4.63 GB Parquet (14 Years, ~88M flights)
- **Analyzed Rows:** 19,572,421 flights
- **Throughput:** ~409,000 flights/sec
- **Target:** `ArrDel15` ($\ge 15$ min delay)
""")

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity Big Data Analytics Project")

# Header
st.markdown('<div class="main-title">✈️ Commercial Flight Delay Big Data & Prediction Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Distributed analysis across 19.5M+ flights and real-time machine learning inference for FAA on-time performance.</div>', unsafe_allow_html=True)

# Top KPI Banner
analytics_data = load_analytics_data()
model_eval = load_model_evaluation()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Analyzed Flight Records", "19,572,421", "14 Partitions (2010-2023)")
with col2:
    delay_rate_str = f"{analytics_data['summary']['delay_rate_pct']:.1f}%" if analytics_data else "19.7%"
    st.metric("Baseline US Delay Rate", delay_rate_str, ">= 15 min FAA standard")
with col3:
    auc_str = f"{model_eval['auc_roc']:.4f}" if model_eval else "0.6710"
    st.metric("XGBoost Model AUC-ROC", auc_str, "Holdout Test Partition")
with col4:
    recall_str = f"{model_eval['recall']*100:.1f}%" if model_eval else "52.7%"
    st.metric("Delay Detection Recall", recall_str, "Decision Threshold = 0.20")

st.markdown("---")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "✈️ Live Flight Delay Predictor",
    "📊 Big Data Analytics (19.5M Flights)",
    "🧠 Machine Learning Architecture & Evaluation",
    "🌐 Cluster Topology, Sharding & Replication"
])

# AIRLINES AND AIRPORTS REFERENCE
AIRLINE_NAMES = {
    "DL": "DL - Delta Air Lines",
    "AA": "AA - American Airlines",
    "UA": "UA - United Airlines",
    "WN": "WN - Southwest Airlines",
    "B6": "B6 - JetBlue Airways",
    "NK": "NK - Spirit Airlines",
    "AS": "AS - Alaska Airlines",
    "F9": "F9 - Frontier Airlines",
    "OO": "OO - SkyWest Airlines",
    "9E": "9E - Endeavor Air",
    "YX": "YX - Republic Airways",
    "MQ": "MQ - Envoy Air",
    "OH": "OH - PSA Airlines",
    "G4": "G4 - Allegiant Air",
    "HA": "HA - Hawaiian Airlines",
    "QX": "QX - Horizon Air",
    "YV": "YV - Mesa Airlines",
}

TOP_AIRPORTS = [
    "ATL", "DFW", "DEN", "ORD", "CLT", "LAX", "LAS", "SEA", "PHX", "MCO",
    "LGA", "DTW", "IAH", "SFO", "MSP", "BOS", "EWR", "JFK", "FLL", "BWI",
    "MDW", "SLC", "DCA", "SAN", "TPA", "BNA", "MSY", "AUS", "PDX", "STL"
]

# ==========================================
# TAB 1: PREDICTOR
# ==========================================
with tab1:
    st.subheader("Interactive Flight Delay Risk Estimation")
    st.write("Configure flight parameters below or select a quick demonstration preset:")

    # Presets
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    preset_data = None
    if preset_col1.button("🌅 Early Morning Business\n(UA: ORD -> SFO, 06:30)"):
        preset_data = {"carrier": "UA", "origin": "ORD", "dest": "SFO", "month": 5, "dow": 2, "dep": 630, "arr": 900, "dist": 1846}
    if preset_col2.button("🌆 Evening Rush Hour\n(DL: JFK -> LAX, 18:30)"):
        preset_data = {"carrier": "DL", "origin": "JFK", "dest": "LAX", "month": 12, "dow": 5, "dep": 1830, "arr": 2200, "dist": 2475}
    if preset_col3.button("⛈️ Summer Storm Season\n(WN: MDW -> MSY, 21:15)"):
        preset_data = {"carrier": "WN", "origin": "MDW", "dest": "MSY", "month": 7, "dow": 7, "dep": 2115, "arr": 2330, "dist": 825}
    if preset_col4.button("⚡ Budget Rush to LGA\n(NK: ATL -> LGA, 17:30)"):
        preset_data = {"carrier": "NK", "origin": "ATL", "dest": "LGA", "month": 11, "dow": 4, "dep": 1730, "arr": 2005, "dist": 762}

    # Form inputs
    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        st.markdown("##### 1. Route & Carrier")
        carrier_keys = list(AIRLINE_NAMES.keys())
        default_carrier_idx = carrier_keys.index(preset_data["carrier"]) if preset_data else carrier_keys.index("DL")
        selected_carrier_key = st.selectbox(
            "Operating Carrier",
            carrier_keys,
            format_func=lambda x: AIRLINE_NAMES[x],
            index=default_carrier_idx
        )

        default_orig = preset_data["origin"] if preset_data else "JFK"
        default_dest = preset_data["dest"] if preset_data else "LAX"
        origin = st.selectbox("Origin Airport", TOP_AIRPORTS, index=TOP_AIRPORTS.index(default_orig) if default_orig in TOP_AIRPORTS else 0)
        dest = st.selectbox("Destination Airport", TOP_AIRPORTS, index=TOP_AIRPORTS.index(default_dest) if default_dest in TOP_AIRPORTS else 5)

    with col_input2:
        st.markdown("##### 2. Scheduled Departure & Arrival")
        default_dep = preset_data["dep"] if preset_data else 1300
        default_arr = preset_data["arr"] if preset_data else 1630

        dep_hour = st.slider("Scheduled Departure Hour (24h)", min_value=0, max_value=23, value=default_dep // 100)
        dep_min = st.select_slider("Scheduled Departure Minute", options=[0, 15, 30, 45], value=(default_dep % 100) if (default_dep % 100) in [0, 15, 30, 45] else 0)
        crs_dep_time = dep_hour * 100 + dep_min

        arr_hour = st.slider("Scheduled Arrival Hour (24h)", min_value=0, max_value=23, value=default_arr // 100)
        arr_min = st.select_slider("Scheduled Arrival Minute", options=[0, 15, 30, 45], value=(default_arr % 100) if (default_arr % 100) in [0, 15, 30, 45] else 30)
        crs_arr_time = arr_hour * 100 + arr_min

    with col_input3:
        st.markdown("##### 3. Date & Flight Distance")
        default_month = preset_data["month"] if preset_data else 5
        default_dow = preset_data["dow"] if preset_data else 3
        default_dist = preset_data["dist"] if preset_data else 2475

        month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        month_idx = st.selectbox("Travel Month", range(1, 13), format_func=lambda x: f"{x} - {month_names[x-1]}", index=default_month - 1)

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_idx = st.selectbox("Day of Week", range(1, 8), format_func=lambda x: f"{x} - {day_names[x-1]}", index=default_dow - 1)

        distance = st.number_input("Flight Distance (Miles)", min_value=50, max_value=5000, value=int(default_dist), step=50)

    predict_btn = st.button("🚀 Calculate Flight Delay Risk", type="primary", use_container_width=True)

    if predict_btn or preset_data:
        if not model_ready:
            st.error("Model predictor is not initialized.")
        else:
            with st.spinner("Executing calibrated inference through XGBoost pipeline..."):
                res = predictor.predict_flight(
                    carrier=selected_carrier_key,
                    origin=origin,
                    dest=dest,
                    month=month_idx,
                    day_of_week=dow_idx,
                    crs_dep_time=crs_dep_time,
                    crs_arr_time=crs_arr_time,
                    distance=float(distance),
                )

            st.markdown("---")
            st.markdown("### 🎯 Flight Delay Risk Assessment")

            out_col1, out_col2, out_col3 = st.columns([1.2, 1, 1.8])

            with out_col1:
                prob = res["delay_probability_pct"]
                st.metric("Estimated Delay Probability", f"{prob:.1f}%")
                st.progress(min(prob / 100.0, 1.0))
                st.caption("Baseline US domestic delay frequency is ~19.7%")

            with out_col2:
                tier = res["risk_tier"]
                status = res["predicted_status"]

                if tier == "LOW RISK":
                    badge_html = f'<div class="risk-badge-low">🟢 {tier}</div>'
                elif tier == "MODERATE RISK":
                    badge_html = f'<div class="risk-badge-mod">🔵 {tier}</div>'
                elif tier == "ELEVATED RISK":
                    badge_html = f'<div class="risk-badge-elevated">🟠 {tier}</div>'
                else:
                    badge_html = f'<div class="risk-badge-high">🔴 {tier}</div>'

                st.markdown(f"**Risk Level:**<br>{badge_html}", unsafe_allow_html=True)
                st.markdown(f"**Predicted Status:** `{status}`")

            with out_col3:
                st.markdown("**Explainable Contributing Factors:**")
                for factor in res["contributing_factors"]:
                    st.markdown(f"- 📌 {factor}")

            # Dynamic 24-Hour Departure Sensitivity Curve Loop
            st.markdown("---")
            st.markdown(f"##### 📈 24-Hour Departure Hour Sensitivity Curve for **{selected_carrier_key} ({origin} &rarr; {dest})**")
            st.caption("How your flight's delay risk changes depending on scheduled departure time throughout the day:")

            sensitivity_records = []
            for h in range(5, 24):
                est_arr = (h + max(1, int(distance / 500))) % 24 * 100
                h_res = predictor.predict_flight(
                    carrier=selected_carrier_key,
                    origin=origin,
                    dest=dest,
                    month=month_idx,
                    day_of_week=dow_idx,
                    crs_dep_time=h * 100,
                    crs_arr_time=est_arr,
                    distance=float(distance)
                )
                sensitivity_records.append({
                    "Hour": f"{h:02d}:00",
                    "Delay Risk (%)": h_res["delay_probability_pct"],
                    "Risk Tier": h_res["risk_tier"]
                })

            df_sensitivity = pd.DataFrame(sensitivity_records)
            st.line_chart(df_sensitivity.set_index("Hour")["Delay Risk (%)"])
            min_hr = df_sensitivity.loc[df_sensitivity["Delay Risk (%)"].idxmin()]
            max_hr = df_sensitivity.loc[df_sensitivity["Delay Risk (%)"].idxmax()]
            st.info(f"💡 **Optimization Insight:** Optimal departure is around **{min_hr['Hour']}** ({min_hr['Delay Risk (%)']:.1f}% risk). Highest congestion risk occurs at **{max_hr['Hour']}** ({max_hr['Delay Risk (%)']:.1f}% risk).")

    # Multi-Flight Fleet Simulation Loop
    st.markdown("---")
    with st.expander("🛫 Multi-Flight Fleet Risk Batch Scanner (Simulated Airline Network)", expanded=False):
        st.write("Simulate and stress-test an entire fleet of concurrent domestic flights across the US airspace network.")

        sample_fleet = [
            {"Carrier": "UA", "Origin": "ORD", "Dest": "SFO", "Month": 5, "DayOfWeek": 2, "CRSDepTime": 630, "CRSArrTime": 900, "Distance": 1846, "FlightNum": "UA-241"},
            {"Carrier": "DL", "Origin": "JFK", "Dest": "LAX", "Month": 12, "DayOfWeek": 5, "CRSDepTime": 1830, "CRSArrTime": 2200, "Distance": 2475, "FlightNum": "DL-482"},
            {"Carrier": "WN", "Origin": "MDW", "Dest": "MSY", "Month": 7, "DayOfWeek": 7, "CRSDepTime": 2115, "CRSArrTime": 2330, "Distance": 825, "FlightNum": "WN-904"},
            {"Carrier": "AA", "Origin": "DFW", "Dest": "MIA", "Month": 3, "DayOfWeek": 3, "CRSDepTime": 1245, "CRSArrTime": 1630, "Distance": 1121, "FlightNum": "AA-119"},
            {"Carrier": "NK", "Origin": "ATL", "Dest": "LGA", "Month": 11, "DayOfWeek": 4, "CRSDepTime": 1730, "CRSArrTime": 2005, "Distance": 762, "FlightNum": "NK-552"},
            {"Carrier": "F9", "Origin": "DEN", "Dest": "LAS", "Month": 10, "DayOfWeek": 1, "CRSDepTime": 815, "CRSArrTime": 930, "Distance": 628, "FlightNum": "F9-301"},
            {"Carrier": "B6", "Origin": "BOS", "Dest": "MCO", "Month": 6, "DayOfWeek": 6, "CRSDepTime": 1945, "CRSArrTime": 2300, "Distance": 1121, "FlightNum": "B6-614"},
            {"Carrier": "AS", "Origin": "SEA", "Dest": "SAN", "Month": 8, "DayOfWeek": 3, "CRSDepTime": 1420, "CRSArrTime": 1705, "Distance": 1050, "FlightNum": "AS-729"},
        ]

        if st.button("⚡ Run Fleet Network Simulation Loop", key="btn_run_fleet"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            fleet_out = []

            for idx, fl in enumerate(sample_fleet):
                status_text.text(f"Evaluating {fl['FlightNum']} ({fl['Carrier']} on {fl['Origin']} -> {fl['Dest']})...")
                res = predictor.predict_flight(
                    carrier=fl["Carrier"],
                    origin=fl["Origin"],
                    dest=fl["Dest"],
                    month=fl["Month"],
                    day_of_week=fl["DayOfWeek"],
                    crs_dep_time=fl["CRSDepTime"],
                    crs_arr_time=fl["CRSArrTime"],
                    distance=float(fl["Distance"])
                )
                fleet_out.append({
                    "Flight #": fl["FlightNum"],
                    "Carrier": fl["Carrier"],
                    "Route": f"{fl['Origin']} -> {fl['Dest']}",
                    "Scheduled Dep": f"{fl['CRSDepTime'] // 100:02d}:{fl['CRSDepTime'] % 100:02d}",
                    "Delay Risk (%)": f"{res['delay_probability_pct']:.1f}%",
                    "Risk Tier": res["risk_tier"],
                    "Predicted Status": res["predicted_status"]
                })
                progress_bar.progress((idx + 1) / len(sample_fleet))

            status_text.success("Fleet simulation loop completed across 8 flight sectors!")
            df_fleet_res = pd.DataFrame(fleet_out)
            st.dataframe(df_fleet_res, hide_index=True, use_container_width=True)

# ==========================================
# TAB 2: BIG DATA ANALYTICS
# ==========================================
with tab2:
    st.subheader("Big Data Operational Analytics (19.5M Flight Records)")
    st.write("Processed through the out-of-core streaming engine in **47.8 seconds** (~409,000 flights/sec).")

    if analytics_data:
        chart_col1, chart_col2 = st.columns(2)

        # Diurnal Curve
        with chart_col1:
            st.markdown("##### 🕒 Diurnal Delay Escalation Curve (Departure Hour vs Delay %)")
            hourly_dict = analytics_data["hourly_delays"]
            hourly_df = pd.DataFrame([
                {
                    "Departure Hour": f"{int(h):02d}:00",
                    "HourNum": int(h),
                    "Delay Rate (%)": round(d["delayed"] / d["total"] * 100, 2) if d["total"] > 0 else 0,
                    "Total Flights": d["total"]
                }
                for h, d in hourly_dict.items() if int(h) >= 5
            ]).sort_values("HourNum")

            st.bar_chart(hourly_df.set_index("Departure Hour")["Delay Rate (%)"])
            st.caption("Key Insight: Morning departures (5-7 AM) experience <10% delays. Rotational cascading pushes evening delays past 28% (a >3.2x escalation).")

        # Carrier Rankings
        with chart_col2:
            st.markdown("##### ✈️ Carrier On-Time Performance & Delay Rate")
            carrier_df = pd.DataFrame(analytics_data["carrier_rankings"]).sort_values("delay_rate")
            carrier_df["Airline"] = carrier_df["carrier"].map(lambda x: AIRLINE_NAMES.get(x, x))
            st.bar_chart(carrier_df.set_index("carrier")["delay_rate"])
            st.caption("Punctual Leaders: 9E, DL, YX (<16% delay). Highest Delays: B6, G4, F9 (>28% delay).")

        st.markdown("---")
        table_col1, table_col2 = st.columns(2)

        with table_col1:
            st.markdown("##### 📍 Top 10 Busiest US Airports & Congestion Delay Rates")
            airports_df = pd.DataFrame(analytics_data["top_airports"][:10])
            airports_df["Departures"] = airports_df["total"].map(lambda x: f"{x:,d}")
            airports_df["Delayed Flights"] = airports_df["delayed"].map(lambda x: f"{x:,d}")
            airports_df["Delay Rate"] = airports_df["delay_rate"].map(lambda x: f"{x:.2f}%")
            st.dataframe(airports_df[["origin", "Departures", "Delayed Flights", "Delay Rate"]], hide_index=True, use_container_width=True)

        with table_col2:
            st.markdown("##### ⚠️ Root Cause Decomposition for Flight Delays")
            causes_dict = analytics_data["delay_causes"]
            tot_causes = sum(causes_dict.values())
            causes_df = pd.DataFrame([
                {
                    "Delay Cause": k.replace("Delay", ""),
                    "Incidents": f"{v:,d}",
                    "Share (%)": f"{v/tot_causes*100:.1f}%"
                }
                for k, v in causes_dict.items()
            ])
            st.dataframe(causes_df, hide_index=True, use_container_width=True)
            st.caption("Carrier maintenance/crew + late arriving aircraft turnaround account for over 67% of delayed flight incidents.")

# ==========================================
# TAB 3: MACHINE LEARNING
# ==========================================
with tab3:
    st.subheader("Machine Learning Model Architecture & Benchmarking")
    st.markdown("""
    The system employs an **XGBoost Gradient Boosted Decision Tree Classifier** trained on stratified multi-year partitions
    with high-cardinality target encoding and calibrated probability estimation.
    """)

    arch_col1, arch_col2 = st.columns([1.2, 1])

    with arch_col1:
        st.markdown("##### 📈 Top 10 Predictive Features for Flight Delays")
        if predictor and hasattr(predictor.model, "feature_importances_"):
            importances = pd.Series(
                predictor.model.feature_importances_,
                index=predictor.metadata["feature_names"]
            ).sort_values(ascending=True).tail(10)

            st.bar_chart(importances)
            st.caption("Departure hour (time of day) and historical airline reliability are the two dominant predictors.")

    with arch_col2:
        st.markdown("##### 🏆 Model Evaluation Metrics (Holdout Test Partition)")
        if model_eval:
            eval_table = pd.DataFrame([
                {"Evaluation Metric": "Area Under ROC (AUC-ROC)", "Score": f"{model_eval.get('auc_roc', 0.671):.4f}"},
                {"Evaluation Metric": "Area Under PR (AUC-PR)", "Score": f"{model_eval.get('auc_pr', 0.3235):.4f}"},
                {"Evaluation Metric": "Overall Accuracy", "Score": f"{model_eval.get('accuracy', 0.682)*100:.1f}%"},
                {"Evaluation Metric": "Balanced Accuracy", "Score": f"{model_eval.get('balanced_accuracy', 0.6217)*100:.1f}%"},
                {"Evaluation Metric": "Delay Capture Recall", "Score": f"{model_eval.get('recall', 0.5267)*100:.1f}%"},
                {"Evaluation Metric": "Optimal Decision Threshold", "Score": f"{model_eval.get('optimal_threshold', 0.20):.2f}"},
            ])
            st.dataframe(eval_table, hide_index=True, use_container_width=True)
            st.caption("The 0.20 threshold detects over 52% of all actual delays while keeping false positives controlled.")

    st.markdown("---")
    st.markdown("##### 🛠️ Distributed PySpark MLlib & Scala Extension")
    st.markdown("""
    For cluster deployment across petabyte scales, the repository also includes:
    - **`src/ml/spark_ml_trainer.py`**: Distributed PySpark MLlib pipeline (`StringIndexer` $\\rightarrow$ `OneHotEncoder` $\\rightarrow$ `VectorAssembler` $\\rightarrow$ `LogisticRegression` / `RandomForestClassifier`).
    - **`Assignment2.scala` / `FlightDelayModel.scala`**: Native Apache Spark Scala implementation for Spark cluster execution.
    """)

# ==========================================
# TAB 4: NODES & REPLICATION
# ==========================================
with tab4:
    st.subheader("🌐 Big Data Cluster Topology, Sharding & Replication Architecture")
    st.write("Demonstration of distributed node parallelism, horizontal sharding keys, and Apache Spark in-memory 2x storage replication.")

    topo_col1, topo_col2, topo_col3 = st.columns(3)
    with topo_col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🖥️ Cluster Nodes & Cores</h4>
            <h2 style="color: #28a745; margin: 0;">12 Cores</h2>
            <p style="color: #888; font-size: 0.85rem;">Local Mode <code>master("local[*]")</code><br>Driver & 12 Parallel Worker Threads</p>
        </div>
        """, unsafe_allow_html=True)
    with topo_col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📂 Horizontal Sharding</h4>
            <h2 style="color: #17a2b8; margin: 0;">14 + 17 Shards</h2>
            <p style="color: #888; font-size: 0.85rem;">14 Year Shards (4.63 GB Parquet)<br>17 Airline Shards (<code>joined.parquet</code>)</p>
        </div>
        """, unsafe_allow_html=True)
    with topo_col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🔄 Data Replication</h4>
            <h2 style="color: #fd7e14; margin: 0;">2x Replicated</h2>
            <p style="color: #888; font-size: 0.85rem;"><code>StorageLevel.MEMORY_AND_DISK_2</code><br>+ RDD Lineage DAG Fault Tolerance</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    sharding_col1, sharding_col2 = st.columns(2)

    with sharding_col1:
        st.markdown("##### 1. Horizontal Sharding Demonstration")
        st.markdown("""
        - **Shard Key 1 (`Year`):** The 4.63 GB dataset is split into **14 discrete directory shards** (`Year=2010` to `Year=2023`). Enables instantaneous **Partition Pruning**.
        - **Shard Key 2 (`Reporting_Airline`):** Partitioned via `.partitionBy("Reporting_Airline")` generating **17 carrier shards** (`AA`, `DL`, `UA`, `WN`, etc.).
        - **In-Memory Shuffling:** When executing equi-joins and group-bys, Spark creates **16 parallel shuffle partitions** mapped to available CPU cores.
        """)

    with sharding_col2:
        st.markdown("##### 2. Storage Replication Demonstration")
        st.markdown("""
        - **Spark Replication Policy:** Datasets are cached using **`StorageLevel.MEMORY_AND_DISK_2`**, which explicitly sets **Replication Factor = 2**.
        - **Multi-Node Fault Tolerance:** In a cluster, partitions are mirrored across peer worker nodes.
        - **DAG Lineage Recovery:** If any partition fails, Spark automatically replays the deterministic Directed Acyclic Graph (DAG) without data loss.
        """)

    st.markdown("---")
    st.markdown("##### 🚀 Live Demonstration via Terminal & Spark Web UI")
    st.info("""
    **To show live cluster nodes and 2x replication to your professor:**
    1. Open PowerShell and run: `python show_cluster_and_replication.py`
    2. Open your browser at: `http://localhost:4040`
    3. Click the **'Storage'** tab to show: `Disk Memory Serialized 2x Replicated` (Replication Factor = 2).
    4. Click the **'Executors'** tab to show all **12 Parallel Cores**, active thread pools, and memory allocations.
    """)

st.markdown("---")
st.caption("Flight Delay Big Data Analytics Platform • University / Capstone Project Demonstration")
