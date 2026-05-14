import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
HISTORY_PATH = os.path.join(PROJECT_ROOT, "eval_pipeline", "data", "history.json")

st.set_page_config(page_title="GrabOn AI Output Guard", layout="wide")
st.title("🛡️ GrabOn AI Labs: Telemetry Dashboard")

raw_data = []
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "r") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError:
            pass

def build_dataframe(data: list) -> pd.DataFrame:
    if not data: return pd.DataFrame()
    
    flattened_data = []
    
    metric_task_map = {
        "format_compliance": "deal_copy",
        "semantic_similarity": "deal_copy",
        "persuasiveness": "deal_copy",
        "intent_accuracy": "insurance_intent",
        "confidence_calibration": "insurance_intent",
        "edge_case_handling": "insurance_intent",
        "factual_grounding": "credit_narrative"
    }
    
    for entry in data:
        task_providers = entry.get("task_providers", {})
        averages = entry.get("averages", {})
        
        if not task_providers:
            flat_entry = {
                "timestamp": entry.get("timestamp"),
                "commit_hash": entry.get("commit_hash", "unknown"),
                "status": entry.get("status", "GO (LEGACY)"),
                "task": "Legacy Suite",       
                "provider": "Legacy Models"   
            }
            for metric_name, score in averages.items():
                flat_entry[metric_name] = score
                
            flattened_data.append(flat_entry)
            continue
            
        for task_name, model_name in task_providers.items():
            flat_entry = {
                "timestamp": entry.get("timestamp"),
                "commit_hash": entry.get("commit_hash", "unknown"),
                "status": entry.get("status"),
                "task": task_name,         
                "provider": model_name     
            }
            
            for metric_name, score in averages.items():
                if metric_task_map.get(metric_name) == task_name:
                    flat_entry[metric_name] = score
                    
            flattened_data.append(flat_entry)
            
    return pd.DataFrame(flattened_data)

df = build_dataframe(raw_data)

pass_rate_str = "0.00%"
total_cases_str = "0"
latency_str = "0.0s"
cost_str = "$0.00000"

if raw_data:
    latest_run = raw_data[-1]
    
    averages = latest_run.get("averages", {})
    if averages:
        overall_avg = sum(averages.values()) / len(averages)
        pass_rate_str = f"{overall_avg:.2%}"
        
    # 2. Dynamic Total Cases (FIXED to sum across tasks)
    raw_arrays = latest_run.get("raw_arrays", {})
    if raw_arrays:
        # Map metrics to tasks so we don't double-count metrics in the same task
        metric_task_map = {
            "format_compliance": "deal_copy",
            "semantic_similarity": "deal_copy",
            "persuasiveness": "deal_copy",
            "intent_accuracy": "insurance_intent",
            "confidence_calibration": "insurance_intent",
            "edge_case_handling": "insurance_intent",
            "factual_grounding": "credit_narrative"
        }
        
        cases_per_task = {}
        for metric_name, arr in raw_arrays.items():
            parent_task = metric_task_map.get(metric_name)
            if parent_task:
                # Find the max cases for this specific task
                cases_per_task[parent_task] = max(cases_per_task.get(parent_task, 0), len(arr))
                
        # Sum the cases across all distinct tasks!
        total_cases_str = str(sum(cases_per_task.values()))
        
    raw_latency = latest_run.get('latency', 0.0)
    latency_str = f"{raw_latency:.1f}s"
    
    real_cost = latest_run.get("cost_data", {}).get("Total", 0.0)
    cost_str = f"${real_cost:.5f}"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Overall Accuracy Score", value=pass_rate_str)
with col2:
    st.metric(label="Cases Evaluated", value=total_cases_str)
with col3:
    st.metric(label="Total Latency", value=latency_str)
with col4:
    st.metric(label="Pipeline Cost (Eval + Gen)", value=cost_str)

st.markdown("Real-time observability for Agentic LLM performance and regression tracking.")

def get_commit_diff(commit_hash):
    if commit_hash == "unknown": return "No commit hash available to fetch diff."
    try:
        result = subprocess.run(
            ["git", "show", commit_hash, "--", "eval_pipeline/prompts/", "eval_config.yaml", "eval_pipeline/scoring/"],
            capture_output=True, text=True, check=True
        )
        diff_text = result.stdout
        if not diff_text.strip(): return "No prompt or configuration files were changed in this commit."
        return diff_text
    except subprocess.CalledProcessError:
        return f"Could not retrieve Git diff for commit: {commit_hash}."

st.sidebar.header("Filter Analytics")

if not df.empty:
    selected_task = st.sidebar.selectbox("Output Type (Task)", options=["All"] + list(df['task'].unique()))
    selected_model = st.sidebar.selectbox("LLM Model", options=["All"] + list(df['provider'].unique()))
    selected_status = st.sidebar.selectbox("Pipeline Status", options=["All", "GO", "NO-GO", "INCONCLUSIVE"])
    
    filtered_df = df.copy()
    if selected_task != "All": filtered_df = filtered_df[filtered_df['task'] == selected_task]
    if selected_model != "All": filtered_df = filtered_df[filtered_df['provider'] == selected_model]
    if selected_status != "All": filtered_df = filtered_df[filtered_df['status'].str.contains(selected_status)]
else:
    st.sidebar.warning("No evaluation history found. Run the pipeline to generate baseline.")
    filtered_df = df

if not filtered_df.empty:
    metadata_cols = ["timestamp", "commit_hash", "status", "task", "provider"]
    available_metrics = [col for col in filtered_df.columns if col not in metadata_cols]

    cols = st.columns(min(len(available_metrics), 4))
    for i, metric in enumerate(available_metrics[:4]):
        with cols[i]:
            avg_val = filtered_df[metric].mean()
            if "ece" in metric.lower():
                st.metric(metric.replace("_", " ").title(), f"{avg_val:.4f}", delta_color="inverse")
            else:
                st.metric(metric.replace("_", " ").title(), f"{avg_val:.2%}")

    if raw_data:
        latest_run = raw_data[-1] 
        if "NO-GO" in latest_run.get("status", "") and latest_run.get("failed_cases"):
            st.markdown("---")
            with st.expander("🚨 CI/CD Blocked: View Failing Edge Cases", expanded=True):
                st.error(f"Commit `{latest_run.get('commit_hash', 'unknown')}` caused regressions on the following test cases:")
                for metric_name, cases in latest_run["failed_cases"].items():
                    if cases: 
                        st.markdown(f"- **{metric_name}:** {', '.join(cases)}")
                st.info("💡 Tip: Look up these specific Case IDs in your JSON datasets to see exactly where the LLM got confused.")

    st.subheader("📈 Quality Trends Over Time")
    production_df = filtered_df[~filtered_df['status'].str.contains("NO-GO")]
    
    if not production_df.empty:
        fig = px.line(
            production_df, x="timestamp", y=available_metrics, color="provider", 
            labels={"value": "Score", "variable": "Metric", "timestamp": "Execution Time", "provider": "Model"},
            markers=True, title="Production Quality (Historical Accuracy)",
            hover_data=["commit_hash", "status", "task"] 
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No successful production deployments to plot yet.")

    st.subheader("🔍 Regression Root Cause Analysis (Commit Log)")
    cols_to_show = ["timestamp", "status", "task", "provider", "commit_hash"] + available_metrics
    log_df = filtered_df[cols_to_show].sort_values(by="timestamp", ascending=False)
    
    def highlight_status(val):
        if isinstance(val, str):
            if 'NO-GO' in val: return 'color: #ff4b4b; font-weight: bold'
            if 'IMPROVED' in val: return 'color: #00cc96; font-weight: bold'
            if 'STABLE' in val: return 'color: #888888'
        return ''
    
    st.dataframe(log_df.style.map(highlight_status), use_container_width=True, hide_index=True)

    st.subheader("🕵️ Live Config & Prompt Diff Explorer")
    st.markdown("---")
    st.subheader("🔬 Deep-Dive: Latest Run Dataset Explorer")
    st.write("Filter the raw, row-by-row telemetry from the most recent pipeline execution.")
    
    csv_path = os.path.join(PROJECT_ROOT, "eval_report_raw.csv")
    
    if os.path.exists(csv_path):
        raw_df = pd.read_csv(csv_path)
        
        colA, colB = st.columns([1, 3])
        
        with colA:
            show_only_failures = st.checkbox("🚨 Show only Regressions", value=False)
            
            filter_task = st.selectbox("Filter Dataset", ["All Tasks"] + list(raw_df["Task"].unique()))
            
        with colB:
            display_df = raw_df.copy()
            if show_only_failures:
                display_df = display_df[display_df["Regression_Detected"].astype(str).str.lower() == "true"]
            if filter_task != "All Tasks":
                display_df = display_df[display_df["Task"] == filter_task]
                
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            csv_export = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered CSV",
                data=csv_export,
                file_name="filtered_eval_report.csv",
                mime="text/csv",
            )
    else:
        st.info("No eval_report_raw.csv found. Run the pipeline to generate row-level data.")
    valid_commits = log_df[log_df['commit_hash'] != 'unknown']['commit_hash'].dropna().unique()
    
    if len(valid_commits) > 0:
        selected_commit = st.selectbox("Select Commit Hash to Inspect", valid_commits)
        commit_status = log_df[log_df['commit_hash'] == selected_commit]['status'].iloc[0]
        
        if "NO-GO" in commit_status:
            st.error(f"🚨 This commit was blocked by the CI/CD Pipeline ({commit_status}). See diff below:")
        elif "IMPROVED" in commit_status:
            st.success(f"🎉 This commit improved production quality ({commit_status}). See diff below:")
            
        if selected_commit:
            with st.spinner(f"Querying Git tree for commit {selected_commit}..."):
                live_diff = get_commit_diff(selected_commit)
            st.code(live_diff, language="diff")
    else:
        st.info("No valid Git commits found in the telemetry history to generate diffs.")