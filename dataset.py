

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import sys
import argparse
from sqlalchemy import create_engine, text

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "data.db")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Activity Templates ───────────────────────────────────────────────────────
ACTIVITY_TEMPLATES = [
    {"name": "Site Survey & Mobilization",         "category": "prep",       "duration": 7,  "depends_on": None},
    {"name": "Demolition & Strip-Out",             "category": "demo",       "duration": 12, "depends_on": "Site Survey & Mobilization"},
    {"name": "Structural Repairs & Reinforcement", "category": "structural", "duration": 18, "depends_on": "Demolition & Strip-Out"},
    {"name": "Waterproofing & Damp Proofing",      "category": "structural", "duration": 8,  "depends_on": "Structural Repairs & Reinforcement"},
    {"name": "Plumbing Rough-in",                  "category": "mep",        "duration": 14, "depends_on": "Structural Repairs & Reinforcement"},
    {"name": "Electrical Wiring & Conduits",       "category": "mep",        "duration": 14, "depends_on": "Structural Repairs & Reinforcement"},
    {"name": "HVAC & Ventilation Installation",    "category": "mep",        "duration": 10, "depends_on": "Structural Repairs & Reinforcement"},
    {"name": "Floor Leveling & Screed",            "category": "finishing",  "duration": 10, "depends_on": "Plumbing Rough-in"},
    {"name": "Tiling — Floors & Wet Areas",        "category": "finishing",  "duration": 18, "depends_on": "Floor Leveling & Screed"},
    {"name": "Wall Plastering & Screeding",        "category": "finishing",  "duration": 14, "depends_on": "Electrical Wiring & Conduits"},
    {"name": "False Ceiling & Insulation",         "category": "finishing",  "duration": 10, "depends_on": "Wall Plastering & Screeding"},
    {"name": "Painting — Primer & Finish Coats",   "category": "finishing",  "duration": 14, "depends_on": "False Ceiling & Insulation"},
    {"name": "Carpentry, Joinery & Built-ins",     "category": "finishing",  "duration": 16, "depends_on": "Painting — Primer & Finish Coats"},
    {"name": "Fixtures, Fittings & Sanitaryware",  "category": "finishing",  "duration": 10, "depends_on": "Carpentry, Joinery & Built-ins"},
    {"name": "Final Inspection, Snag & Handover",  "category": "inspection", "duration": 6,  "depends_on": "Fixtures, Fittings & Sanitaryware"},
]

ISSUE_CATEGORIES = ["material_delay", "labor_shortage", "weather", "design_change",
                     "inspection_fail", "equipment_breakdown", "scope_creep", "safety"]
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
ASSIGNEES = ["site_manager", "contractor_A", "contractor_B", "project_engineer",
             "safety_officer", "qa_inspector", "procurement_lead", "client_pm"]

BOQ_TEMPLATES = {
    "prep":       [("Site Survey & Layout", 3300, 2310, "lumpsum", 1),
                   ("Temporary Site Office Setup", 8000, 5600, "lumpsum", 1),
                   ("Safety Hoarding & Signage", 2100, 1470, "lumpsum", 1)],
    "demo":       [("Demolition Labour & Equipment", 12000, 8400, "lumpsum", 1),
                   ("Debris Removal & Disposal", 5500, 3850, "trip", 6),
                   ("Asbestos Survey & Removal", 4000, 2800, "sqm", 30)],
    "structural": [("Reinforced Concrete M30", 9500, 6650, "m3", 15),
                   ("Steel Rebar TMT 12mm", 3800, 2660, "tonne", 6),
                   ("SBR Waterproofing Membrane", 4300, 3010, "sqm", 120)],
    "mep":        [("CPVC Pipes 1 inch", 2800, 1960, "meters", 180),
                   ("FR Copper Wire 4 sq mm", 4200, 2940, "meters", 250),
                   ("Cassette AC Unit 2 Ton", 14000, 9800, "unit", 4)],
    "finishing":  [("Self-Leveling Compound", 3300, 2310, "bags", 20),
                   ("Vitrified Tiles 800x800mm", 7500, 5250, "sqm", 200),
                   ("Gypsum Plaster 20mm", 2500, 1750, "bags", 50),
                   ("Premium Emulsion Paint", 3800, 2660, "liters", 120),
                   ("Teak Wood Panels Grade A", 9500, 6650, "sqm", 30),
                   ("EWC Rimless Toilet Suite", 8500, 5950, "unit", 4)],
    "inspection": [("Snagging Rectification Labour", 5000, 3500, "lumpsum", 1),
                   ("As-Built Drawings & O&M Manual", 3000, 2100, "set", 1)],
}

CONTRACTORS = ["Apex Interiors", "ProStruct Engineers", "Swift MEP Pvt Ltd",
               "Urban Build Solutions", "EliteFinish Works", "BuildRight Co.",
               "National Contractors"]

# ── Delay helper ─────────────────────────────────────────────────────────────
def add_delay(project_type="residential"):
    """Skewed delay distribution — more likely late than early."""
    delay = int(np.random.choice(
        [-1, 0, 0, 1, 2, 3, 5, 7],
        p=[0.05, 0.20, 0.20, 0.20, 0.15, 0.10, 0.07, 0.03]
    ))
    return max(0, delay)

def compute_schedule_start(template_name, template_start, activities_so_far):
    """Compute start based on dependency (predecessor end)."""
    dep = next((t["depends_on"] for t in ACTIVITY_TEMPLATES if t["name"] == template_name), None)
    if dep is None:
        return template_start
    pred = next((a for a in activities_so_far if a["name"] == dep), None)
    if pred is None:
        return template_start
    return pred["actual_end"] or pred["planned_end"]

# ── Data Generators ──────────────────────────────────────────────────────────
def generate_activities(project_id, project_type, planned_project_start, is_complete=True, today=None):
    if today is None:
        today = datetime.today()
    
    activities = []
    act_num = 0
    current_planned_start = planned_project_start
    
    for tmpl in ACTIVITY_TEMPLATES:
        act_num += 1
        act_id = f"act_{project_id.split('_')[1]}_{act_num:02d}"
        planned_dur = int(tmpl["duration"])
        planned_start = current_planned_start
        planned_end = planned_start + timedelta(days=planned_dur)

        # Delay injection
        actual_start_delay = int(add_delay(project_type))
        dep = tmpl["depends_on"]
        if dep and activities:
            pred = next((a for a in activities if a["name"] == dep), None)
            if pred:
                actual_start = pred["actual_end_date"]
            else:
                actual_start = planned_start + timedelta(days=actual_start_delay)
        else:
            actual_start = planned_start + timedelta(days=actual_start_delay)

        actual_dur = int(planned_dur + add_delay(project_type))
        actual_end = actual_start + timedelta(days=actual_dur) if is_complete else None

        schedule_var = (actual_start - planned_start).days

        status = "completed" if is_complete else "not_started"
        progress = 100 if is_complete else 0

        activities.append({
            "id": act_id,
            "project_id": project_id,
            "project_type": project_type,
            "name": tmpl["name"],
            "category": tmpl["category"],
            "planned_start_date": planned_start,
            "planned_end_date": planned_end,
            "planned_duration_days": planned_dur,
            "actual_start_date": actual_start,
            "actual_end_date": actual_end,
            "forecasted_start_date": actual_start,
            "forecasted_end_date": actual_end,
            "progress": progress,
            "status": status,
            "parent_id": None,
            "depends_on": f"act_{project_id.split('_')[1]}_{act_num-1:02d}" if act_num > 1 else None,
            "actual_duration_days": actual_dur if is_complete else None,
            "schedule_variance_days": schedule_var,
        })

        # Next planned start = this planned end
        current_planned_start = planned_end

    return activities


def generate_daily_updates(activity_id, project_id, actual_start, actual_end):
    updates = []
    if actual_start is None or actual_end is None:
        return updates
    total_days = (actual_end - actual_start).days
    if total_days <= 0:
        return updates
    progress = 0
    notes_pool = [
        "Good progress — crew at full strength.",
        "Safety toolbox talk held; no incidents.",
        "Weather caused brief stoppage.",
        "Equipment breakdown caused 2-hour downtime.",
        "Minor delays due to material delivery.",
        "Night shift deployed to catch up.",
        "Overtime shift completed to recover schedule.",
        "Client walkthrough conducted.",
        "Rework required on small section.",
        "Inspection checkpoint cleared.",
        "Material quality check done; passed.",
        "Productivity affected by heat; hydration breaks added.",
        "Waiting for subcontractor sign-off.",
        "Work progressing as planned.",
        "All tasks on track per daily plan.",
    ]
    upd_id = 0
    for day in range(total_days):
        increment = np.random.normal(loc=100 / total_days, scale=3)
        increment = max(0, min(increment, 16))
        progress = min(100, progress + increment)
        upd_id += 1
        updates.append({
            "id": f"upd_{activity_id}_{upd_id:04d}",
            "activity_id": activity_id,
            "project_id": project_id,
            "date": actual_start + timedelta(days=day),
            "reported_progress": round(progress, 1),
            "daily_increment": round(increment, 2),
            "image_uploaded": random.choice([True, False]),
            "weather_event": random.random() < 0.1,
            "crew_size": random.randint(4, 22),
            "notes": random.choice(notes_pool),
            "has_issue_logged": random.random() < 0.15,
        })
    return updates


def generate_issues(activity_id, project_id, activity_name, num_issues, activity_start):
    issues = []
    for i in range(num_issues):
        cat = random.choice(ISSUE_CATEGORIES)
        sev = random.choice(SEVERITY_LEVELS)
        delay_impact = random.choice([0, 0, 0, 1, 2, 3, 4, 5]) if sev in ["high", "critical"] else 0
        issues.append({
            "id": f"iss_{activity_id}_{i+1:03d}",
            "activity_id": activity_id,
            "project_id": project_id,
            "description": f"{cat.replace('_',' ').title()} encountered during {activity_name}.",
            "category": cat,
            "severity": sev,
            "status": random.choice(["open", "resolved", "resolved", "resolved"]),
            "assigned_to": random.choice(ASSIGNEES),
            "date_raised": activity_start + timedelta(days=random.randint(0, 10)),
            "delay_impact_days": delay_impact,
        })
    return issues


def generate_boq(activity_id, project_id, category):
    boq_items = []
    templates = BOQ_TEMPLATES.get(category, [])
    for tmpl in templates:
        name, price, cost, unit, qty = tmpl
        noise = np.random.uniform(0.90, 1.10)
        p = int(price * noise)
        c = int(cost * noise)
        boq_items.append({
            "id": f"boq_{activity_id}_{len(boq_items)+1:03d}",
            "activity_id": activity_id,
            "project_id": project_id,
            "name": name,
            "unit": unit,
            "quantity": qty,
            "unit_price": p,
            "unit_cost": c,
            "total_price": p * qty,
            "total_cost": c * qty,
            "margin_pct": round((p - c) / p * 100, 1),
            "currency": "INR",
        })
    return boq_items


def generate_resources(activity_id, project_id, actual_start, actual_end):
    if actual_start is None or actual_end is None:
        return []
    resource_types = ["Labour Gang", "Equipment", "Material Supplier", "Specialist Subcontractor"]
    resources = []
    num = random.randint(1, 3)
    for i in range(num):
        resources.append({
            "id": f"res_{activity_id}_{i+1:03d}",
            "activity_id": activity_id,
            "project_id": project_id,
            "contractor": random.choice(CONTRACTORS),
            "resource_type": random.choice(resource_types),
            "allocated_workers": random.randint(3, 20),
            "cost_per_day": random.randint(8000, 35000),
            "start_date": actual_start,
            "end_date": actual_end,
        })
    return resources


# ── Full project generator ────────────────────────────────────────────────────
def generate_project(proj_id, name, proj_type, city, planned_start_str, planned_end_str,
                      is_complete=True):
    planned_start = datetime.strptime(planned_start_str, "%Y-%m-%d")
    planned_end = datetime.strptime(planned_end_str, "%Y-%m-%d")

    activities = generate_activities(proj_id, proj_type, planned_start, is_complete=is_complete)

    if is_complete:
        actual_end = activities[-1]["actual_end_date"]
        status = "completed"
    else:
        actual_end = None
        status = "in_progress"

    project = {
        "id": proj_id,
        "name": name,
        "planned_start": planned_start,
        "planned_end": planned_end,
        "type": proj_type,
        "city": city,
        "actual_start": activities[0]["actual_start_date"],
        "actual_end": actual_end,
        "status": status,
    }

    all_updates, all_issues, all_boq, all_resources = [], [], [], []
    for act in activities:
        all_updates.extend(generate_daily_updates(
            act["id"], proj_id, act["actual_start_date"], act["actual_end_date"]))
        num_issues = random.randint(2, 5)
        all_issues.extend(generate_issues(
            act["id"], proj_id, act["name"], num_issues, act["actual_start_date"]))
        all_boq.extend(generate_boq(act["id"], proj_id, act["category"]))
        all_resources.extend(generate_resources(
            act["id"], proj_id, act["actual_start_date"], act["actual_end_date"]))

    return project, activities, all_updates, all_issues, all_boq, all_resources


# ── CSV Ingest ───────────────────────────────────────────────────────────────
def ingest_csvs(engine):
    """Load all CSVs from data/ folder into the SQLite DB."""
    csv_files = {
        "projects": "projects.csv",
        "activities": "activities.csv",
        "daily_updates": "daily_updates.csv",
        "issues": "issues.csv",
        "boq": "boq.csv",
        "resources": "resources.csv",
    }
    for table, fname in csv_files.items():
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            # Normalize column names for projects table
            if table == "projects":
                rename_map = {
                    "planned_start": "planned_start",
                    "planned_end": "planned_end",
                    "actual_start": "actual_start",
                    "actual_end": "actual_end",
                }
                df = df.rename(columns=rename_map)
            df.to_sql(table, engine, if_exists="replace", index=False)
            print(f"  ✅ Ingested {fname} → {table} ({len(df)} rows)")
        else:
            print(f"  ⚠️  {fname} not found, skipping.")


# ── Activity Dependencies Table ──────────────────────────────────────────────
def build_dependencies_table(engine):
    """Build activity_dependencies junction table from the depends_on column."""
    with engine.connect() as conn:
        try:
            df = pd.read_sql("SELECT id, depends_on FROM activities WHERE depends_on IS NOT NULL AND depends_on != ''", conn)
        except Exception:
            return
    if df.empty:
        return
    df = df.rename(columns={"id": "activity_id", "depends_on": "predecessor_id"})
    df = df.dropna(subset=["predecessor_id"])
    df = df[df["predecessor_id"].str.strip() != ""]
    df.to_sql("activity_dependencies", engine, if_exists="replace", index=False)
    print(f"  ✅ Built activity_dependencies table ({len(df)} rows)")


# ── Main ─────────────────────────────────────────────────────────────────────
def main(mode="both"):
    os.makedirs(DATA_DIR, exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")
    print(f"\n📦 Database: {DB_PATH}\n")

    if mode in ("csv", "both"):
        print("📂 Ingesting CSVs...")
        ingest_csvs(engine)

    if mode in ("gen", "both"):
        print("\n🔧 Generating synthetic projects (will add to DB if not present)...")
        # Check if already have projects — if CSV ingest already happened, skip duplicates
        with engine.connect() as conn:
            try:
                existing_ids = pd.read_sql("SELECT id FROM projects", conn)["id"].tolist()
            except Exception:
                existing_ids = []

        # Add 2 extra synthetic completed projects for richer training set
        extra_projects = [
            ("proj_011", "Beachfront Bungalow Reno", "residential", "Kochi",
             "2023-08-01", "2024-02-28", True),
            ("proj_012", "Shopping Mall Fit-out", "commercial", "Kolkata",
             "2023-10-01", "2024-05-31", True),
        ]
        all_p, all_a, all_u, all_i, all_b, all_r = [], [], [], [], [], []
        for args in extra_projects:
            pid = args[0]
            if pid in existing_ids:
                print(f"  ℹ️  {pid} already in DB, skipping.")
                continue
            p, a, u, i, b, r = generate_project(*args)
            all_p.append(p); all_a.extend(a); all_u.extend(u)
            all_i.extend(i); all_b.extend(b); all_r.extend(r)
            print(f"  ✅ Generated {args[1]} ({len(a)} activities, {len(u)} updates)")

        if all_p:
            pd.DataFrame(all_p).to_sql("projects", engine, if_exists="append", index=False)
            pd.DataFrame(all_a).to_sql("activities", engine, if_exists="append", index=False)
            if all_u: pd.DataFrame(all_u).to_sql("daily_updates", engine, if_exists="append", index=False)
            if all_i: pd.DataFrame(all_i).to_sql("issues", engine, if_exists="append", index=False)
            if all_b: pd.DataFrame(all_b).to_sql("boq", engine, if_exists="append", index=False)
            if all_r: pd.DataFrame(all_r).to_sql("resources", engine, if_exists="append", index=False)

    print("\n🔗 Building dependency graph table...")
    build_dependencies_table(engine)

    # Summary
    with engine.connect() as conn:
        for tbl in ["projects", "activities", "daily_updates", "issues", "boq", "resources", "activity_dependencies"]:
            try:
                cnt = pd.read_sql(f"SELECT COUNT(*) as n FROM {tbl}", conn)["n"].iloc[0]
                print(f"  📊 {tbl}: {cnt} rows")
            except Exception:
                pass

    print("\n✅ Database ready!")
    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="What-if Scheduler Data Setup")
    parser.add_argument("--csv-only", action="store_true", help="Only ingest CSVs")
    parser.add_argument("--gen-only", action="store_true", help="Only generate synthetic data")
    args = parser.parse_args()

    if args.csv_only:
        mode = "csv"
    elif args.gen_only:
        mode = "gen"
    else:
        mode = "both"

    main(mode=mode)
