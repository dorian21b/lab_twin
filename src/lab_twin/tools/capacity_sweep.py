# tools/capacity_sweep.py (new helper, minimal)
from pathlib import Path
from itertools import product
from lab_twin.workflow.steps_map import STEPS
from lab_twin.workflow.steps_scaled import make_steps_scaled
from lab_twin.domain.sample import Sample
from lab_twin.domain.batch import Batch
from lab_twin.sim.engine import run_sim_multi
from lab_twin.sim.resources import make_resources
from lab_twin.utils.kpis import write_kpi_summary
import simpy, pandas as pd, random, time

def _batches(n=3, n_samples=96):
    out=[]
    for i in range(n):
        b=Batch(batch_id=f"B{i+1:02d}")
        b.add_samples([Sample() for _ in range(n_samples)])
        out.append(b)
    return out

def run_one(overrides, seed=1, root="outputs_caps"):
    tag = "_".join(f"{k.replace(' ','')}-{v}" for k,v in sorted(overrides.items()))
    run_id = f"caps_{int(time.time())}_{tag}"
    run_dir = Path(root)/run_id; run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    env = simpy.Environment()
    resources = make_resources(env, overrides=overrides)

    # assemble + run
    from lab_twin.sim.arrivals import launch_batches
    from lab_twin.sim.line_full import build_full_graph
    from lab_twin.utils.eventlog import EventLogger, EventRow
    logger = EventLogger(run_dir/"events_report.csv")

    def log_event(**row):
        logger.write(EventRow(run_id, row["batch_id"], row["process_id"],
                              row["sim_start_min"], row["sim_end_min"], row["service_min"],
                              row["wait_min"], row["queue_len_on_arrival"], row["status"],
                              row.get("resource_name",""), row.get("note","")))

    head, _tails = build_full_graph(env, STEPS, log_event, resources)
    batches = _batches(3, 96)
    arrival_times = [0, 10, 20]
    launch_batches(env, head, batches, arrival_times)
    env.run(); logger.close()

    # KPIs
    from lab_twin.utils.logger import write_samples_report
    from datetime import datetime, timezone
    write_samples_report(batches, datetime.now(timezone.utc), run_dir=run_dir, run_id=run_id)
    write_kpi_summary(run_dir=run_dir, run_id=run_id, sla_min=5*12*60)

    k = pd.read_csv(run_dir/"kpi_overall.csv").iloc[0]
    return {"run_id": run_id, "dir": str(run_dir), "J_proxy": k["p90_tat_min"],  # simple comparator
            "avg": k["avg_tat_min"], "p90": k["p90_tat_min"], "SLA": k.get("sla_pct", 0.0)}

if __name__ == "__main__":
    scenarios = [
        {},  # baseline
        {"Novaseq X +": 2},
        {"Quantstudio 5": 2},
        {"Hamilton Star": 2},
        {"Novaseq X +": 2, "Quantstudio 5": 2},
    ]
    rows = [run_one(o) for o in scenarios]
    df = pd.DataFrame(rows); print(df.sort_values("p90"))
