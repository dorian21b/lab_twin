# src/lab_twin/main_multi.py
from datetime import datetime, timezone
from pathlib import Path
import random
from lab_twin.domain.sample import Sample
from lab_twin.domain.batch import Batch
from lab_twin.workflow.steps import GSTT_PROCESS_STEPS
from lab_twin.sim.engine import run_sim_multi
from lab_twin.utils.logger import write_samples_report 
from lab_twin.utils.kpis import write_kpi_summary

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

# Staggered arrivals → queues form at step 1
arrival_times = [0, 1, 2]  # minutes

run_sim_multi(
    batches=batches,
    steps=GSTT_PROCESS_STEPS,
    arrival_times=arrival_times,
    seed=42,
    run_dir=run_dir,
    run_id=run_id,
)

# Write one sample report per batch (optional)
end_time = datetime.now(timezone.utc)
write_samples_report(batches, end_time, run_dir=run_dir, run_id=run_id)
write_kpi_summary(run_dir=run_dir, run_id=run_id, sla_min=120.0)
print(f"Run artifacts in: {run_dir.resolve()}")
