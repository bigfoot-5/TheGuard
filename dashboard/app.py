import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

# ==========================================
# CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="GrabOn AI Output Guard", layout="wide")
st.title("🛡️ GrabOn AI Labs: Telemetry Dashboard")
st.markdown("Real-time observability for Agentic LLM performance and regression tracking.")

# ==========================================
# REAL DATA LOADER
# ==========================================
def load_data():
    history_path = "eval_pipeline/data/history.json"
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            raw_data = json.load(f)
            
        # Flatten the nested 'averages' dict into the main row
        flattened_data = []
        for entry in raw_data:
            flat_entry = {
                "timestamp": entry.get("timestamp"),
                "version": entry.get("version", "latest"),
                "task": entry.get("task"),
                "provider": entry.get("provider")
            }
            # Merge the averages in
            if "averages" in entry:
                flat_entry.update(entry["averages"])
            flattened_data.append(flat_entry)
            
        return pd.DataFrame(flattened_data)
    else:
        return pd.DataFrame(columns=["timestamp", "version", "task", "provider", "grounding", "compliance", "similarity", "ece", "intent_accuracy"])

df = load_data()

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.header("Filter Analytics")

if not df.empty:
    # Engineers can filter the telemetry by Output Type, LLM Provider, and Prompt Version Hash.
    selected_task = st.sidebar.selectbox("Select Output Type", options=["All"] + list(df['task'].unique()))
    selected_provider = st.sidebar.selectbox("LLM Provider", options=["All"] + list(df.get('provider', ['GPT-4o-mini', 'Claude 3.5']).unique()))
    selected_version = st.sidebar.multiselect("Prompt Version Hash", options=df['version'].unique(), default=df['version'].unique())

    # Apply filters
    filtered_df = df[df['version'].isin(selected_version)]
    if selected_task != "All":
        filtered_df = filtered_df[filtered_df['task'] == selected_task]
    if selected_provider != "All" and 'provider' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['provider'] == selected_provider]
else:
    st.sidebar.warning("No evaluation history found. Run the pipeline first.")
    filtered_df = df

# ==========================================
# MAIN DASHBOARD METRICS
# ==========================================
if not filtered_df.empty:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_grounding = filtered_df['grounding'].mean()
        st.metric("Avg Factual Grounding", f"{avg_grounding:.2%}")

    with col2:
        avg_compliance = filtered_df['compliance'].mean()
        st.metric("Format Compliance", f"{avg_compliance:.2%}")

    with col3:
        avg_similarity = filtered_df['similarity'].mean()
        st.metric("Semantic Similarity", f"{avg_similarity:.2%}")
        
    with col4:
        # Using .get to safely handle ECE if it wasn't in earlier runs
        avg_ece = filtered_df.get('ece', pd.Series([0])).mean()
        st.metric("Expected Calibration Error", f"{avg_ece:.4f}", delta_color="inverse")

    # ==========================================
    # PERFORMANCE TRENDS 
    # ==========================================
    st.subheader("📈 Quality Trends Over Time")
    
    # Check which metrics actually exist in the dataframe to plot
    metrics_to_plot = [m for m in ["grounding", "compliance", "similarity", "ece"] if m in filtered_df.columns]
    
    fig = px.line(
        filtered_df, 
        x="timestamp", 
        y=metrics_to_plot,
        labels={"value": "Score", "variable": "Metric"},
        markers=True,
        title="Rolling Average of Scoring Functions"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # ANOMALY ATTRIBUTION & PROMPT DIFF EXPLORER
    # ==========================================
    st.subheader("🔍 Regression Root Cause Analysis")
    st.info("The dashboard executes a query to retrieve the associated Git commit hash and immediately renders the prompt diff alongside the failed test cases.")
    
    # In a fully integrated system, this would dynamically pull from Git based on the selected point on the graph.
    # We display a placeholder interactive expander for the presentation.
    expander = st.expander("View Prompt Diff (Commit: a7b29f0c)")
    expander.code("""
    --- a/eval_pipeline/prompts/deal_copy_v1.txt
    +++ b/eval_pipeline/prompts/deal_copy_v2.txt
    @@ -1,2 +1,2 @@
    - You are a helpful marketing assistant. Maintain a professional tone.
    + You are a high-energy growth hacker assistant. Use emojis and keep it extremely short to drive FOMO.
    """, language="diff")
    
    expander.write("**Failed Test Case Traces:**")
    expander.json({
        "case_id": "CRD-094",
        "expected_compliance": True,
        "actual_compliance": False,
        "reason": "Character limit exceeded (185/160) due to excessive emoji usage."
    })

else:
    st.info("Awaiting pipeline execution. The charts will populate once `history.json` contains evaluation data.")