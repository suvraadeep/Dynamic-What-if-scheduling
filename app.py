

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data_loader import DataLoader
from models.completion_predictor import CompletionPredictor
from models.monte_carlo import MonteCarloSimulator
from engine.dag_builder import build_dag, get_descendants
from engine.ripple_engine import RippleEngine
from engine.whatif_scenarios import WhatIfScenarioEngine
from optimizer.schedule_optimizer import ScheduleOptimizer
from visualization.gantt import build_gantt
from visualization.dag_viz import build_dag_figure

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="What-if Scheduler",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f172a;
    color: #e2e8f0;
}
.stApp { background-color: #0f172a; }
section[data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }

.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 8px;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
.kpi-value { font-size: 2.2rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
.kpi-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; }

.section-header {
    font-size: 1.1rem; font-weight: 600; color: #f1f5f9;
    border-left: 4px solid #6366f1; padding-left: 12px; margin: 16px 0 10px 0;
}
.suggestion-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px;
}
.suggestion-card.critical { border-left: 4px solid #ef4444; }
.suggestion-card.high     { border-left: 4px solid #f97316; }
.suggestion-card.medium   { border-left: 4px solid #fbbf24; }
.suggestion-card.opport   { border-left: 4px solid #22c55e; }

div[data-testid="stMetric"] { background: #1e293b; border-radius: 10px; padding: 12px 16px; }
.stTabs [data-baseweb="tab"] { background: #1e293b; color: #94a3b8; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: #312e81; color: #e0e7ff; }

.dataframe thead th { background-color: #1e293b !important; color: #a5b4fc !important; }
.dataframe tbody tr:nth-child(odd) { background-color: #0f172a !important; }
.dataframe tbody tr:nth-child(even) { background-color: #1a2437 !important; }
</style>
""", unsafe_allow_html=True)

# ── Caching ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading data...")
def get_loader():
    return DataLoader()

@st.cache_resource(show_spinner="Training ML model...")
def get_predictor():
    dl = get_loader()
    cp = CompletionPredictor(loader=dl)
    metrics = cp.train()
    return cp, metrics

@st.cache_resource(show_spinner="Running predictions...", ttl=300)
def get_predictions(project_id):
    cp, _ = get_predictor()
    return cp.predict_all(project_id=project_id)

@st.cache_resource(show_spinner="Running Monte Carlo...", ttl=300)
def get_mc_results(project_id):
    dl = get_loader()
    mc = MonteCarloSimulator(loader=dl, n_sims=500)
    return mc.simulate_all(project_id=project_id)

@st.cache_resource(show_spinner="Loading optimizer...", ttl=300)
def _get_optimizer(project_id):
    dl = get_loader()
    opt = ScheduleOptimizer(project_id, loader=dl)
    opt.compute_cpm()
    return opt

def get_cpm(project_id):
    opt = _get_optimizer(project_id)
    cpm_df = opt._cpm_results if opt._cpm_results is not None else opt.compute_cpm()
    cp_ids = opt.get_critical_path()
    return cpm_df, cp_ids, opt

# ── Sidebar ───────────────────────────────────────────────────────────────────
loader = get_loader()
st.sidebar.markdown("## What-if Scheduler")
st.sidebar.markdown("---")

projs = loader.projects
if projs.empty:
    st.error("No project data found. Run `python dataset.py` first.")
    st.stop()

proj_name_map = {
    row["id"]: (
        "In Progress  " if row.get("status") == "in_progress" else
        "Completed    " if row.get("status") == "completed" else
        "Not Started  "
    ) + f"{row.get('name', row['id'])} ({row['id']})"
    for _, row in projs.iterrows()
}

inprog = projs[projs["status"] == "in_progress"]
default_proj = inprog.iloc[0]["id"] if not inprog.empty else projs.iloc[0]["id"]

selected_pid = st.sidebar.selectbox(
    "Select Project",
    options=list(proj_name_map.keys()),
    format_func=lambda x: proj_name_map[x],
    index=list(proj_name_map.keys()).index(default_proj) if default_proj in proj_name_map else 0,
)

ref_date = st.sidebar.date_input(
    "Reference Date (Today)",
    value=datetime(2024, 6, 1).date(),
    min_value=datetime(2022, 1, 1).date(),
    max_value=datetime(2026, 12, 31).date(),
)
today = datetime.combine(ref_date, datetime.min.time())

st.sidebar.markdown("---")
selected_proj = projs[projs["id"] == selected_pid].iloc[0]
st.sidebar.markdown(f"**Name:** {selected_proj.get('name', '-')}")
st.sidebar.markdown(f"**Status:** {selected_proj.get('status', '-').replace('_',' ').title()}")
st.sidebar.markdown(f"**Type:** {selected_proj.get('type', '-').title()}")
st.sidebar.markdown(f"**City:** {selected_proj.get('city', '-')}")

_, train_metrics = get_predictor()
if train_metrics:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Health**")
    st.sidebar.caption(f"Train samples: {train_metrics.get('n_train', '-')}")
    st.sidebar.caption(f"CV MAE: {train_metrics.get('cv_mae_mean', '-'):.3f} +/- {train_metrics.get('cv_mae_std', '-'):.3f}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.markdown(f"## What-if Schedule Predictor -- *{selected_proj.get('name', selected_pid)}*")

tabs = st.tabs([
    "Overview",
    "Gantt Chart",
    "Predictions",
    "Ripple Analysis",
    "What-if Scenarios",
    "Optimization",
    "DAG View",
])

acts = loader.get_project_activities(selected_pid)
preds = get_predictions(selected_pid) if selected_proj.get("status") in ("in_progress", "not_started") else pd.DataFrame()
cpm_df, cp_ids, optimizer = get_cpm(selected_pid)

# ═════════════════════════════════════════ TAB 0: OVERVIEW ════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Project At a Glance</div>', unsafe_allow_html=True)

    in_prog_acts = acts[acts["status"] == "in_progress"]
    completed_acts = acts[acts["status"] == "completed"]
    completed_n = len(completed_acts)
    in_prog_n = len(in_prog_acts)

    avg_progress = float(acts["progress"].fillna(0).mean()) if "progress" in acts.columns else 0

    all_issues = loader.get_activity_issues(project_id=selected_pid)
    open_issues = (
        all_issues[all_issues["status"] == "open"]
        if not all_issues.empty and "status" in all_issues.columns
        else pd.DataFrame()
    )
    critical_issues = (
        open_issues[open_issues["severity"].isin(["high", "critical"])]
        if not open_issues.empty and "severity" in open_issues.columns
        else pd.DataFrame()
    )

    sched_var = float(acts["schedule_variance_days"].dropna().mean()) if "schedule_variance_days" in acts.columns else 0
    boq_df = loader.get_project_boq(selected_pid)

    cols_kpi = st.columns(6)
    kpi_data = [
        (f"{avg_progress:.0f}%", "Overall Progress", "#6366f1"),
        (str(completed_n),       "Activities Done",  "#22c55e"),
        (str(in_prog_n),         "In Progress",      "#f59e0b"),
        (str(len(open_issues)),  "Open Issues",      "#f97316"),
        (str(len(critical_issues)), "Critical Issues", "#ef4444"),
        (f"{sched_var:+.0f}d",  "Avg Schedule Var.", "#8b5cf6"),
    ]
    for col, (val, label, color) in zip(cols_kpi, kpi_data):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:{color}">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-header">Activity Status Breakdown</div>', unsafe_allow_html=True)
        status_counts = acts["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        STATUS_COLOR_MAP = {"completed": "#22c55e", "in_progress": "#f59e0b", "not_started": "#64748b"}
        colors = [STATUS_COLOR_MAP.get(s, "#94a3b8") for s in status_counts["Status"]]
        fig_status = go.Figure(go.Pie(
            labels=status_counts["Status"],
            values=status_counts["Count"],
            marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
            hole=0.5,
            textinfo="percent+label",
        ))
        fig_status.update_layout(
            paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
            height=300, showlegend=False, margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_status, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Progress by Category</div>', unsafe_allow_html=True)
        if "category" in acts.columns and "progress" in acts.columns:
            cat_prog = acts.groupby("category")["progress"].mean().reset_index()
            fig_cat = go.Figure(go.Bar(
                x=cat_prog["progress"].round(0),
                y=cat_prog["category"],
                orientation="h",
                marker=dict(color=cat_prog["progress"], colorscale="RdYlGn", showscale=False),
                text=cat_prog["progress"].round(0).astype(str) + "%",
                textposition="auto",
            ))
            fig_cat.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"), height=300,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(range=[0, 105]),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown('<div class="section-header">Activity List</div>', unsafe_allow_html=True)
    display_cols = ["id", "name", "category", "status", "progress",
                    "planned_start_date", "planned_end_date", "schedule_variance_days"]
    disp_acts = acts[[c for c in display_cols if c in acts.columns]].copy()
    disp_acts["progress"] = disp_acts["progress"].fillna(0).round(1)
    st.dataframe(disp_acts, use_container_width=True, height=280)

# ══════════════════════════════════════════ TAB 1: GANTT ══════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Gantt Chart -- Planned vs Actual vs Forecast</div>', unsafe_allow_html=True)
    gantt_fig = build_gantt(
        selected_pid, loader=loader,
        predictions_df=preds,
        critical_path_ids=cp_ids,
        today=today,
    )
    st.plotly_chart(gantt_fig, use_container_width=True)
    st.caption("Critical path activities have a red prefix | Blue = Planned | Green = Actual | Amber = Forecasted")

# ══════════════════════════════════════ TAB 2: PREDICTIONS ════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Completion Date Predictions</div>', unsafe_allow_html=True)
    if preds.empty:
        st.info("No active activities to predict for this project (all may be completed).")
    else:
        mc_results = get_mc_results(selected_pid)

        disp_preds = preds.copy()
        for col in ["methodA_end", "methodB_end", "ensemble_end"]:
            if col in disp_preds.columns:
                disp_preds[col] = pd.to_datetime(disp_preds[col], errors="coerce").dt.strftime("%Y-%m-%d")

        table_cols = ["activity_id", "activity_name", "progress", "planned_end_date",
                      "methodA_end", "methodB_end", "ensemble_end",
                      "delay_multiplier_pred", "issue_count", "schedule_variance"]
        st.dataframe(
            disp_preds[[c for c in table_cols if c in disp_preds.columns]],
            use_container_width=True, height=260,
        )

        st.markdown("---")
        st.markdown('<div class="section-header">Monte Carlo Simulation -- Completion Distribution</div>', unsafe_allow_html=True)

        if not mc_results.empty:
            mc_sel_col, mc_display_col = st.columns([2, 3])
            with mc_sel_col:
                mc_act_options = {
                    row["activity_id"]: f"{row.get('activity_name', row['activity_id'])} ({row['activity_id']})"
                    for _, row in mc_results.iterrows()
                }
                selected_mc_act = st.selectbox(
                    "Select Activity for MC Histogram",
                    options=list(mc_act_options.keys()),
                    format_func=lambda x: mc_act_options[x],
                )
                mc_row = mc_results[mc_results["activity_id"] == selected_mc_act].iloc[0]
                st.metric("P50 (Median)",       str(mc_row["p50_date"])[:10])
                st.metric("P80 (Cautious)",     str(mc_row["p80_date"])[:10])
                st.metric("P90 (Conservative)", str(mc_row["p90_date"])[:10])
                st.metric("Avg days to complete", f"{mc_row['mean_days_to_complete']:.0f} days")

            with mc_display_col:
                dl_mc = get_loader()
                mc_sim = MonteCarloSimulator(loader=dl_mc, n_sims=500)
                prog_mc = float(mc_row.get("current_progress", 0))
                dist = mc_sim.get_distribution_for_plot(selected_mc_act, prog_mc)

                fig_mc = go.Figure()
                fig_mc.add_trace(go.Histogram(
                    x=dist, nbinsx=30,
                    marker=dict(color="#6366f1", opacity=0.8, line=dict(color="#e0e7ff", width=0.5)),
                    name="Simulations",
                ))
                p50 = int(mc_row["p50_days"])
                p80 = int(mc_row["p80_days"])
                p90 = int(mc_row["p90_days"])
                for day, label, color in [(p50, "P50", "#22c55e"), (p80, "P80", "#f59e0b"), (p90, "P90", "#ef4444")]:
                    fig_mc.add_vline(x=day, line=dict(color=color, dash="dash", width=2),
                                     annotation_text=label, annotation_position="top")
                fig_mc.update_layout(
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
                    xaxis_title="Days to Completion", yaxis_title="Simulations",
                    height=320, margin=dict(t=30, b=30), showlegend=False,
                )
                st.plotly_chart(fig_mc, use_container_width=True)

        cp_obj, _ = get_predictor()
        if hasattr(cp_obj, "feature_importances_") and cp_obj.feature_importances_ is not None:
            st.markdown("---")
            st.markdown('<div class="section-header">Feature Importances (GradientBoosting)</div>', unsafe_allow_html=True)
            fi = cp_obj.feature_importances_.head(10).reset_index()
            fi.columns = ["Feature", "Importance"]
            fig_fi = go.Figure(go.Bar(
                x=fi["Importance"], y=fi["Feature"], orientation="h",
                marker=dict(color="#818cf8"),
            ))
            fig_fi.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"), height=300,
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

# ═══════════════════════════════════ TAB 3: RIPPLE ANALYSIS ═══════════════════
with tabs[3]:
    st.markdown('<div class="section-header">Ripple Effect -- Delay Propagation</div>', unsafe_allow_html=True)

    G = build_dag(selected_pid, loader=loader)
    ripple_engine = RippleEngine(G, loader=loader)

    act_options = {
        str(row["id"]): f"{row.get('name', row['id'])} [{row.get('status', '-')}]"
        for _, row in acts.iterrows()
    }
    col_sel, col_delta = st.columns([3, 1])
    with col_sel:
        ripple_act = st.selectbox("Activity to delay:", options=list(act_options.keys()),
                                  format_func=lambda x: act_options[x])
    with col_delta:
        ripple_days = st.number_input("Delay (days)", min_value=-30, max_value=90, value=7, step=1)

    if st.button("Run Ripple Simulation", type="primary"):
        with st.spinner("Propagating delay..."):
            result = ripple_engine.propagate_delay(ripple_act, ripple_days, reference_date=today)

        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric("Activities Affected", result["num_activities_affected"])
        orig_end = result.get("original_project_end")
        new_end = result.get("new_project_end")
        r_col2.metric("Original Project End", str(orig_end)[:10] if orig_end else "N/A")
        r_col3.metric("New Project End", str(new_end)[:10] if new_end else "N/A",
                      delta=f"{result['total_project_delay_days']:+d} days",
                      delta_color="inverse")

        cascade = result["cascade_table"]
        if not cascade.empty:
            st.markdown('<div class="section-header">Cascade Impact Table</div>', unsafe_allow_html=True)
            display_cascade = cascade.copy()
            for dcol in ["original_start", "original_end", "new_start", "new_end"]:
                if dcol in display_cascade.columns:
                    display_cascade[dcol] = pd.to_datetime(display_cascade[dcol], errors="coerce").dt.strftime("%Y-%m-%d")
            st.dataframe(display_cascade, use_container_width=True)

            fig_ripple = go.Figure(go.Bar(
                x=cascade["cascade_delay_days"],
                y=cascade["activity_name"],
                orientation="h",
                marker=dict(
                    color=cascade["cascade_delay_days"],
                    colorscale="RdYlGn_r",
                    showscale=True,
                    colorbar=dict(title="Days delayed"),
                ),
            ))
            fig_ripple.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"),
                xaxis_title="Cascade Delay (days)",
                height=max(250, len(cascade) * 35),
                margin=dict(t=10, b=10),
                title="Cascade Delay per Activity",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_ripple, use_container_width=True)
        else:
            st.info("No downstream activities affected by this delay.")

    st.markdown("---")
    st.markdown('<div class="section-header">Top 5 High-Impact Activities (most downstream dependencies)</div>', unsafe_allow_html=True)
    top5 = ripple_engine.get_high_impact_activities()
    if not top5.empty:
        st.dataframe(top5, use_container_width=True)

# ══════════════════════════════════ TAB 4: WHAT-IF SCENARIOS ══════════════════
with tabs[4]:
    st.markdown('<div class="section-header">What-if Scenario Builder</div>', unsafe_allow_html=True)

    if "scenario_engine" not in st.session_state or st.session_state.get("scenario_project") != selected_pid:
        st.session_state.scenario_engine = WhatIfScenarioEngine(
            selected_pid, loader=loader, reference_date=today
        )
        st.session_state.scenario_project = selected_pid

    engine = st.session_state.scenario_engine

    scen_type = st.radio(
        "Scenario Type:",
        ["Delay", "Resource Boost", "Issue Resolved", "Parallelize"],
        horizontal=True,
    )

    s_col1, s_col2 = st.columns([3, 1])

    if scen_type == "Delay":
        with s_col1:
            delay_act = st.selectbox("Activity to delay:", options=list(act_options.keys()),
                                     format_func=lambda x: act_options[x], key="scen_delay_act")
        with s_col2:
            delay_d = st.number_input("Days delayed:", 1, 60, 7, key="scen_delay_days")
        if st.button("Add Delay Scenario", type="primary"):
            engine.scenario_delay(delay_act, delay_d)
            st.success(f"Scenario added: delay {delay_act} by {delay_d} days")

    elif scen_type == "Resource Boost":
        with s_col1:
            boost_act = st.selectbox("Activity to boost:", options=list(act_options.keys()),
                                     format_func=lambda x: act_options[x], key="scen_boost_act")
        with s_col2:
            boost_pct = st.slider("Duration reduction %:", 10, 50, 25, key="scen_boost_pct")
        if st.button("Add Resource Boost Scenario", type="primary"):
            engine.scenario_resource_boost(boost_act, boost_pct)
            st.success(f"Scenario added: boost {boost_act} by {boost_pct}%")

    elif scen_type == "Issue Resolved":
        all_iss = loader.get_activity_issues(project_id=selected_pid)
        if all_iss.empty:
            st.info("No issues found for this project.")
        else:
            open_iss = all_iss[all_iss["status"] == "open"] if "status" in all_iss.columns else all_iss
            if open_iss.empty:
                st.info("No open issues found.")
            else:
                issue_options = {
                    str(row["id"]): (
                        f"{row.get('id','-')} -- {row.get('category','-')}"
                        f" [{row.get('severity','-')}] -- {row.get('delay_impact_days', 0):.0f}d impact"
                    )
                    for _, row in open_iss.iterrows()
                }
                with s_col1:
                    sel_issue = st.selectbox("Issue to resolve:", options=list(issue_options.keys()),
                                             format_func=lambda x: issue_options[x])
                if st.button("Add Issue Resolved Scenario", type="primary"):
                    engine.scenario_issue_resolved(sel_issue)
                    st.success(f"Scenario added: resolved issue {sel_issue}")

    elif scen_type == "Parallelize":
        with s_col1:
            par_a = st.selectbox("Activity A:", options=list(act_options.keys()),
                                 format_func=lambda x: act_options[x], key="par_a")
        with s_col2:
            par_b = st.selectbox("Activity B:", options=list(act_options.keys()),
                                 format_func=lambda x: act_options[x], key="par_b")
        if st.button("Add Parallelization Scenario", type="primary"):
            engine.scenario_parallelize(par_a, par_b)
            st.success(f"Scenario added: parallelize {par_a} + {par_b}")

    comparison = engine.get_scenario_comparison()
    if not comparison.empty:
        st.markdown("---")
        st.markdown('<div class="section-header">Scenario Comparison</div>', unsafe_allow_html=True)
        disp_comp = comparison.copy()
        for dcol in ["original_project_end", "new_project_end"]:
            if dcol in disp_comp.columns:
                disp_comp[dcol] = pd.to_datetime(disp_comp[dcol], errors="coerce").dt.strftime("%Y-%m-%d")
        st.dataframe(disp_comp[["scenario_id", "type", "description", "original_project_end",
                                 "new_project_end", "total_project_delay_days", "days_saved",
                                 "cost_impact_inr"]], use_container_width=True)

        fig_scen = go.Figure(go.Bar(
            x=comparison["scenario_id"],
            y=comparison["days_saved"],
            marker=dict(color=comparison["days_saved"], colorscale="RdYlGn", showscale=False),
            text=comparison["days_saved"].apply(lambda d: f"{d:+d}d"),
            textposition="auto",
        ))
        fig_scen.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"), height=280,
            xaxis_title="Scenario", yaxis_title="Days Saved (-ve = delay)",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_scen, use_container_width=True)

        if st.button("Clear All Scenarios"):
            engine.clear_scenarios()
            st.rerun()
    else:
        st.info("No scenarios yet. Add one above.")

# ══════════════════════════════════════ TAB 5: OPTIMIZATION ═══════════════════
with tabs[5]:
    st.markdown('<div class="section-header">Critical Path Method (CPM) and Optimization</div>', unsafe_allow_html=True)

    if cpm_df is not None and not cpm_df.empty:
        cpm_disp = cpm_df.copy()
        cpm_disp["critical"] = cpm_disp["is_critical_path"].map({True: "YES", False: "--"})
        st.dataframe(
            cpm_disp[["activity_name", "planned_duration_days", "early_start_day",
                       "early_finish_day", "total_float_days", "progress", "critical"]],
            use_container_width=True, height=280,
        )

        st.markdown('<div class="section-header">Total Float per Activity (days of flexibility)</div>', unsafe_allow_html=True)
        float_fig = go.Figure(go.Bar(
            x=cpm_df["activity_name"].str[:25],
            y=cpm_df["total_float_days"],
            marker=dict(
                color=cpm_df["total_float_days"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Float"),
            ),
        ))
        float_fig.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"), height=320,
            yaxis_title="Float (days)", xaxis_tickangle=-30,
            margin=dict(t=10, b=80),
        )
        st.plotly_chart(float_fig, use_container_width=True)
    else:
        st.info("CPM computation requires activity dependency data.")

    st.markdown("---")
    st.markdown('<div class="section-header">Optimization Suggestions</div>', unsafe_allow_html=True)

    suggestions = optimizer.generate_suggestions(predictions_df=preds)
    if suggestions:
        for sug in suggestions:
            priority = sug["priority"]
            cls = (
                "critical" if "CRITICAL" in priority else
                "high"     if "HIGH"     in priority else
                "medium"   if "MEDIUM"   in priority else
                "opport"
            )
            savings = sug.get("estimated_savings_days", 0)
            savings_txt = f" | Potential saving: <strong>{savings} days</strong>" if savings > 0 else ""
            st.markdown(f"""
            <div class="suggestion-card {cls}">
                <strong>{priority} -- {sug['rule']}</strong>{savings_txt}<br/>
                {sug['suggestion']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No critical optimization issues detected for this project.")

# ══════════════════════════════════════ TAB 6: DAG VIEW ═══════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">Activity Dependency Graph</div>', unsafe_allow_html=True)

    dag_fig = build_dag_figure(
        selected_pid, loader=loader,
        critical_path_ids=cp_ids,
        predictions_df=preds,
    )
    st.plotly_chart(dag_fig, use_container_width=True)

    st.markdown('<div class="section-header">Dependency Details</div>', unsafe_allow_html=True)
    dep_rows = []
    G_disp = build_dag(selected_pid, loader=loader)
    act_name_map = {str(row["id"]): row.get("name", row["id"]) for _, row in acts.iterrows()}
    for edge in G_disp.edges():
        dep_rows.append({
            "Predecessor": act_name_map.get(edge[0], edge[0]),
            "Successor":   act_name_map.get(edge[1], edge[1]),
            "Critical Edge": "YES" if edge[0] in cp_ids and edge[1] in cp_ids else "--",
        })
    if dep_rows:
        st.dataframe(pd.DataFrame(dep_rows), use_container_width=True)
    else:
        st.info("No dependency edges found -- activities may have no depends_on data.")
