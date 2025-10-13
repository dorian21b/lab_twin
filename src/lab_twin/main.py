from datetime import datetime, timezone
from pathlib import Path
from lab_twin.domain.sample import Sample
from lab_twin.domain.batch import Batch
from lab_twin.sim.engine import run_sim
from lab_twin.workflow.steps import GSTT_PROCESS_STEPS
from lab_twin.utils.logger import write_sample_report


#run_id = "day1_single" 
run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_dir = Path("outputs") / run_id
run_dir.mkdir(parents=True, exist_ok=True)

batch = Batch(batch_id="BATCH_001")
batch.add_samples([Sample() for _ in range(96)])

steps = GSTT_PROCESS_STEPS

run_sim(batch, steps, seed=42, run_dir=run_dir, run_id=run_id)

end_time = datetime.now(timezone.utc)
write_sample_report(batch, end_time, run_dir=run_dir, run_id=run_id)

print(f"Run artifacts in: {run_dir.resolve()}")
