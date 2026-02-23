
import sys
sys.path.insert(0, ".")

print("=" * 60)
print("  What-if Scheduler — Integration Test")
print("=" * 60)

# ── Data Loader ──────────────────────────────────────────────
from data_loader import DataLoader
dl = DataLoader()
print(f"\n[1] DataLoader ✅")
print(f"    Projects: {len(dl.projects)}, Activities: {len(dl.activities)}")
print(f"    Historical: {len(dl.get_historical_activities())}, Active: {len(dl.get_active_activities())}")

# ── Feature Engineering ──────────────────────────────────────
from features.feature_engineering import engineer_features
hist = dl.get_historical_activities()
feats = engineer_features(hist, dl)
print(f"\n[2] Feature Engineering ✅")
print(f"    Shape: {feats.shape}  (rows×features)")
feature_cols = ["planned_duration","progress_rate","delay_ratio","issue_count","schedule_variance"]
print("   ", feats[feature_cols].describe().round(2).to_string())

# ── Predictions ───────────────────────────────────────────────
from models.completion_predictor import CompletionPredictor
cp = CompletionPredictor(loader=dl)
metrics = cp.train()
print(f"\n[3] Completion Predictor ✅")
print(f"    Training samples: {metrics.get('n_train')}")
print(f"    CV MAE: {metrics.get('cv_mae_mean', 0):.4f} ± {metrics.get('cv_mae_std', 0):.4f}")

preds = cp.predict_all(project_id="proj_008")
print(f"    Predictions for proj_008: {len(preds)} activities")
print(preds[["activity_id","progress","methodA_end","ensemble_end"]].to_string(index=False))

# ── Monte Carlo ───────────────────────────────────────────────
from models.monte_carlo import MonteCarloSimulator
mc = MonteCarloSimulator(loader=dl, n_sims=300)
mc_res = mc.simulate_all(project_id="proj_008")
print(f"\n[4] Monte Carlo ✅  ({len(mc_res)} activities simulated)")
if not mc_res.empty:
    print(mc_res[["activity_id","current_progress","p50_date","p80_date","p90_date"]].to_string(index=False))

# ── DAG + Ripple ──────────────────────────────────────────────
from engine.dag_builder import build_dag, get_topological_order
from engine.ripple_engine import RippleEngine
G = build_dag("proj_008", loader=dl)
print(f"\n[5] DAG ✅  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
topo = get_topological_order(G)
print(f"    Topological order (first 5): {topo[:5]}")

re = RippleEngine(G, loader=dl)
ripple = re.propagate_delay("act_008_04", 14)
print(f"\n[6] Ripple Engine ✅")
print(f"    Delaying act_008_04 by 14 days:")
print(f"    → Activities affected: {ripple['num_activities_affected']}")
print(f"    → Project delay: {ripple['total_project_delay_days']} days")
if not ripple["cascade_table"].empty:
    print(ripple["cascade_table"][["activity_name","cascade_delay_days"]].to_string(index=False))

# ── CPM Optimizer ─────────────────────────────────────────────
from optimizer.schedule_optimizer import ScheduleOptimizer
opt = ScheduleOptimizer("proj_008", loader=dl)
cpm_df = opt.compute_cpm()
cp_ids = opt.get_critical_path()
print(f"\n[7] CPM ✅  Critical path activities: {len(cp_ids)}")
print(cpm_df[["activity_name","total_float_days","is_critical_path"]].to_string(index=False))

suggestions = opt.generate_suggestions(predictions_df=preds)
print(f"\n[8] Suggestions ✅  ({len(suggestions)} rules fired)")
for s in suggestions[:4]:
    print(f"    [{s['priority']}] {s['rule']}: {s['suggestion'][:80]}")

# ── What-if Scenarios ─────────────────────────────────────────
from engine.whatif_scenarios import WhatIfScenarioEngine
wif = WhatIfScenarioEngine("proj_008", loader=dl)
s1 = wif.scenario_delay("act_008_05", 10)
s2 = wif.scenario_resource_boost("act_008_07", 25)
print(f"\n[9] What-if Scenarios ✅")
comp = wif.get_scenario_comparison()
print(comp[["type","description","total_project_delay_days","days_saved"]].to_string(index=False))

print("\n" + "=" * 60)
print("  ALL TESTS PASSED ✅")
print("=" * 60)
