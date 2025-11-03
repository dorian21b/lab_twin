# quick_smoke.py
from pathlib import Path
from lab_twin.domain.sample import Sample
from lab_twin.domain.batch import Batch
from lab_twin.workflow.steps_map import STEPS
from lab_twin.sim.engine import run_sim_multi

batches = []
for i in range(3):
    b = Batch(batch_id=f"B{i+1:02d}")
    b.add_samples([Sample() for _ in range(96)])
    batches.append(b)

arrival_times = [0, 10, 20]  # minutes
run_sim_multi(
    batches=batches,
    steps=STEPS,
    arrival_times=arrival_times,
    seed=1,
    run_dir=Path("outputs_smoke"),
    run_id="smoke",
    resource_overrides=None,  # try overrides next step
)
