# lab_twin/opt/time_sensitivity.py
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import random
import pandas as pd

from lab_twin.sim.engine import run_sim_multi
from lab_twin.workflow.steps_map import STEPS
from lab_twin.domain.sample import Sample
from lab_twin.domain.batch import Batch
from lab_twin.utils.kpis import write_kpi_summary
from lab_twin.utils.logger import write_samples_report

# ----------------------------------------
# 1. Parse scenario tags for duration scale
# ----------------------------------------
def parse_duration_scale_tag(tag: str) -> dict[str, float]:
    """Convert 'SEQUENCINGx0.9_QCx0.8' -> {'SEQUENCING':0.9, 'QC':0.8}"""
    if not tag or tag.lower() in ("none", "{}"):
        return {}
    scale = {}
    for part in tag.split("_"):
        if "x" in part:
            k, v = part.split("x", 1)
            scale[k] = float(v)
    return scale

# ----------------------------------------
# 2. Example run entrypoint
# ----------------------------------------
def main():
    scenario_tag = "SEQUENCINGx0.9"   # <--- change this for each test
    duration_scale = parse_duration_scale_tag(scenario_tag)

    run_id = f"sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{scenario_tag}"
    run_dir = Path("outputs_timeopt") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build a few batches
    batches = []
    for i in range(3):
        b = Batch(batch_id=f"BATCH_{i+1:03d}")
        b.add_samples([Sample() for _ in range(96)])
        batches.append(b)

    arrival_times = [0, 10, 20]  # example schedule

    run_sim_multi(
        batches=batches,
        steps=STEPS,
        arrival_times=arrival_times,
        seed=42,
        run_dir=run_dir,
        run_id=run_id,
        duration_scale=duration_scale,     # <-- HERE is where you pass it in
    )

    write_samples_report(batches, datetime.now(), run_dir, run_id)
    write_kpi_summary(run_dir, run_id, sla_min=5*12*60)
    print(f"\n✅ Run done: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
