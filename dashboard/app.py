import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import subprocess

# ==========================================
# CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="GrabOn AI Output Guard", layout="wide")
st.title("🛡️ GrabOn AI Labs: Telemetry Dashboard")
st.markdown("Real-time observability for Agentic LLM performance and regression tracking.")

# ==========================================
# LIVE GIT INTEGRATION
# ==========================================
def get_real_prompt_diff(commit_hash):
    """Dynamically pulls the git diff for the prompts folder at a specific commit."""
    if commit_hash == "unknown":
        return "No commit hash available to fetch diff."
    
    try:
        # git show <hash> -- <path> returns the exact diff for that folder
        result = subprocess.run(
            ["git", "show", commit_hash, "--", "eval_pipeline/prompts/"],
            capture_output=True,
            text=True,
            check=True
        )
        diff_text = result.stdout
        if not diff_text.strip():
            return "No prompt files were changed in this commit."
        return diff_text
    except subprocess.CalledProcessError:
        return f"Could not retrieve Git diff for commit: {commit_hash}. Ensure it exists in the local Git tree."

# ==========================================
# REAL DATA LOADER
# ==========================================
def load_data():
    history_path = "eval_pipeline/data/history.json"
    
    # Graceful fallback: If no artifact exists
    if not os.path.exists(history_path):
        return pd.DataFrame()

    with open(history_path, "r") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError:
            return pd.DataFrame()
        
    flattened_data = []
    for entry in raw_data:
        flat_entry = {
            "timestamp": entry.get("timestamp"),
            "commit_hash": entry.get("commit_hash", "unknown"),
            "status": entry.get("status", "GO (LEGACY)"), # Defaults to GO for older runs
            "task": entry.get("task", "Full Suite"),
            "provider": entry.get("provider", "gpt-4o-mini")
        }
        # Merge whatever averages exist (adapts dynamically to any schema)
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
    selected_task = st.sidebar.selectbox("Select Output Type", options=["All"] + list(df['task'].unique()))
    selected_status = st.sidebar.selectbox("Pipeline Status", options=["All", "GO", "NO-GO", "INCONCLUSIVE"])
    
    filtered_df = df.copy()
    if selected_task != "All":
        filtered_df = filtered_df[filtered_df['task'] == selected_task]
    if selected_status != "All":
        # Fuzzy match for status (e.g. "GO" matches "GO (STABLE)" and "GO (IMPROVED)")
        filtered_df = filtered_df[filtered_df['status'].str.contains(selected_status)]
else:
    st.sidebar.warning("No evaluation history found. Run the pipeline to generate baseline.")
    filtered_df = df

# ==========================================
# MAIN DASHBOARD METRICS
# ==========================================
if not filtered_df.empty:
    # Dynamically extract whatever metrics are in the dataframe (ignoring metadata columns)
    metadata_cols = ["timestamp", "commit_hash", "status", "task", "provider"]
    available_metrics = [col for col in filtered_df.columns if col not in metadata_cols]

    # Show top-level metric averages
    cols = st.columns(min(len(available_metrics), 4))
    for i, metric in enumerate(available_metrics[:4]):
        with cols[i]:
            avg_val = filtered_df[metric].mean()
            # Format ECE differently than percentages
            if "ece" in metric.lower():
                st.metric(metric.replace("_", " ").title(), f"{avg_val:.4f}", delta_color="inverse")
            else:
                st.metric(metric.replace("_", " ").title(), f"{avg_val:.2%}")

    # ==========================================
    # PERFORMANCE TRENDS 
    # ==========================================
    st.subheader("📈 Quality Trends Over Time")
    
    # We only plot runs that actually deployed (GO or LEGACY). 
    # NO-GOs shouldn't distort the active production trendline!
    production_df = filtered_df[~filtered_df['status'].str.contains("NO-GO")]
    
    if not production_df.empty:
        fig = px.line(
            production_df, 
            x="timestamp", 
            y=available_metrics,
            labels={"value": "Score", "variable": "Metric", "timestamp": "Execution Time"},
            markers=True,
            title="Production Quality (Excluding Blocked PRs)",
            hover_data=["commit_hash", "status", "task"] 
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No successful production deployments to plot yet.")

    # ==========================================
    # HISTORICAL COMMIT LOG
    # ==========================================
    st.subheader("🔍 Regression Root Cause Analysis (Commit Log)")
    st.info("Use this log to trace exact performance drops back to the Git commit that caused them. Failed NO-GO runs are recorded here for auditing.")
    
    cols_to_show = ["timestamp", "status", "commit_hash"] + available_metrics
    log_df = filtered_df[cols_to_show].sort_values(by="timestamp", ascending=False)
    
    # Optional: Highlight NO-GO rows using Pandas styling
    def highlight_status(val):
        if isinstance(val, str):
            if 'NO-GO' in val: return 'color: #ff4b4b; font-weight: bold'
            if 'IMPROVED' in val: return 'color: #00cc96; font-weight: bold'
            if 'STABLE' in val: return 'color: #888888'
        return ''
    
    st.dataframe(log_df.style.map(highlight_status), use_container_width=True, hide_index=True)

    # ==========================================
    # LIVE PROMPT DIFF EXPLORER
    # ==========================================
    st.subheader("🕵️ Live Prompt Diff Explorer")
    st.write("Select a commit hash from the telemetry log to query the local Git tree and view the exact prompt changes that triggered this run.")
    
    valid_commits = log_df[log_df['commit_hash'] != 'unknown']['commit_hash'].dropna().unique()
    
    if len(valid_commits) > 0:
        selected_commit = st.selectbox("Select Commit Hash to Inspect", valid_commits)
        
        # Look up the status of the selected commit to display a helpful warning
        commit_status = log_df[log_df['commit_hash'] == selected_commit]['status'].iloc[0]
        if "NO-GO" in commit_status:
            st.error(f"🚨 This commit was blocked by the CI/CD Pipeline ({commit_status}). See diff below:")
        elif "IMPROVED" in commit_status:
            st.success(f"🎉 This commit improved production quality ({commit_status}). See diff below:")
            
        if selected_commit:
            with st.spinner(f"Querying Git tree for commit {selected_commit}..."):
                live_diff = get_real_prompt_diff(selected_commit)
            st.code(live_diff, language="diff")
    else:
        st.info("No valid Git commits found in the telemetry history to generate diffs.")

else:
    st.info("Awaiting pipeline execution. The charts will populate once `history.json` contains evaluation data.")