from datetime import datetime, timezone
from pathlib import Path
import random
import pandas as pd
import matplotlib.pyplot as plt

from lab_twin.domain.sample import Sample
from lab_twin.domain.batch import Batch
from lab_twin.workflow.steps import GSTT_PROCESS_STEPS
from lab_twin.sim.engine import run_sim_multi
from lab_twin.utils.logger import write_samples_report
from lab_twin.utils.kpis import write_kpi_summary
from lab_twin.workflow.steps_map import STEPS  # the dict above

from lab_twin.utils.plots import _save_plots

def main():
    random.seed(42)

    run_id = f"multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("outputs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build several batches
    batches = []
    for i in range(3):
        b = Batch(batch_id=f"BATCH_{i+1:03d}")
        b.add_samples([Sample() for _ in range(96)])
        batches.append(b)

    # Staggered arrivals → create queues naturally
    arrival_times = [0, 1, 2]  # minutes

    # 1) Simulate
    run_sim_multi(
        batches=batches,
        steps= STEPS,
        arrival_times=arrival_times,
        seed=42,
        run_dir=run_dir,
        run_id=run_id,
    )

    # 2) Sample report
    end_time = datetime.now(timezone.utc)
    write_samples_report(batches, end_time, run_dir=run_dir, run_id=run_id)

    # 3) KPI summary (pick an SLA that makes sense for your use case; placeholder below)
    #    If you want to “ignore SLA” effects visually, just leave a large number (e.g., a month in minutes).

    write_kpi_summary(run_dir=run_dir, run_id=run_id, sla_min=5*12*60)

    # 4) Plots
    _save_plots(run_dir)

    print(f"\nRun artifacts in: {run_dir.resolve()}")
    for p in ["events_report.csv",
              f"kpi_overall_{run_id}.csv",
              f"kpi_batches_{run_id}.csv",
              f"kpi_process_{run_id}.csv",
              "plot_A_gantt.png",
              "plot_B_process_perf.png",
              "plot_C_batch_tat.png",
              "plot_D_decisions.png"]:
        cand = run_dir / p
        if cand.exists():
            print(" -", cand)



if __name__ == "__main__":
    main()
