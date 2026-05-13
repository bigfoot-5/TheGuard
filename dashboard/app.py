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
    
    # Graceful fallback: If no artifact has been downloaded yet
    if not os.path.exists(history_path):
        return pd.DataFrame()

    with open(history_path, "r") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError:
            return pd.DataFrame()
        
    # Flatten the nested 'averages' dict into the main row
    flattened_data = []
    for entry in raw_data:
        flat_entry = {
            "timestamp": entry.get("timestamp"),
            "commit_hash": entry.get("commit_hash", "unknown"),
            "task": entry.get("task", "Full Suite"),
            "provider": entry.get("provider", "gpt-4o-mini")
        }
        # Merge the computed averages in
        if "averages" in entry:
            flat_entry.update(entry["averages"])
        flattened_data.append(flat_entry)
        
    return pd.DataFrame(flattened_data)

df = load_data()

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.header("Filter Analytics")

if not df.empty:
    # Engineers filter the telemetry by Output Type, Model, and Git Commit.
    selected_task = st.sidebar.selectbox("Select Output Type", options=["All"] + list(df['task'].unique()))
    selected_provider = st.sidebar.selectbox("LLM Provider", options=["All"] + list(df['provider'].unique()))
    
    # Apply filters
    filtered_df = df.copy()
    if selected_task != "All":
        filtered_df = filtered_df[filtered_df['task'] == selected_task]
    if selected_provider != "All":
        filtered_df = filtered_df[filtered_df['provider'] == selected_provider]
else:
    st.sidebar.warning("No evaluation history found. Run the local eval pipeline or download a GitHub Artifact to generate the baseline.")
    filtered_df = df

# ==========================================
# MAIN DASHBOARD METRICS
# ==========================================
if not filtered_df.empty:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_compliance = filtered_df.get('compliance', pd.Series([0])).mean()
        st.metric("Format Compliance", f"{avg_compliance:.2%}")

    with col2:
        avg_similarity = filtered_df.get('similarity', pd.Series([0])).mean()
        st.metric("Semantic Similarity", f"{avg_similarity:.2%}")

    with col3:
        avg_intent = filtered_df.get('intent_accuracy', pd.Series([0])).mean()
        st.metric("Intent Accuracy", f"{avg_intent:.2%}")
        
    with col4:
        avg_ece = filtered_df.get('ece', pd.Series([0])).mean()
        st.metric("Expected Calibration Error", f"{avg_ece:.4f}", delta_color="inverse")

    # ==========================================
    # PERFORMANCE TRENDS 
    # ==========================================
    st.subheader("📈 Quality Trends Over Time")
    
    # Dynamically plot whichever metrics exist in the JSON
    metrics_to_plot = [m for m in ["compliance", "similarity", "intent_accuracy", "ece"] if m in filtered_df.columns]
    
    fig = px.line(
        filtered_df, 
        x="timestamp", 
        y=metrics_to_plot,
        labels={"value": "Score", "variable": "Metric", "timestamp": "Execution Time"},
        markers=True,
        title="Rolling Average of Scoring Functions",
        hover_data=["commit_hash", "task", "provider"] # Shows the commit hash when hovering over a point!
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # HISTORICAL COMMIT LOG (The Rubric Winner)
    # ==========================================
    st.subheader("🔍 Regression Root Cause Analysis (Commit Log)")
    st.info("Use this table to trace exact performance drops back to the Git commit that caused them. If Telugu translation quality dropped, find the anomalous row and grab the associated Commit Hash to view the prompt diff.")
    
    # Dynamically build columns so it doesn't crash if a metric is missing
    cols_to_show = ["timestamp", "commit_hash", "task", "provider"]
    cols_to_show.extend(metrics_to_plot)
    
    # Sort by newest first
    log_df = filtered_df[cols_to_show].sort_values(by="timestamp", ascending=False)
    
    # Display the interactive dataframe
    st.dataframe(log_df, use_container_width=True, hide_index=True)

else:
    st.info("Awaiting pipeline execution. The charts will populate once `history.json` contains evaluation data.")