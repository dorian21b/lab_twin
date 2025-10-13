import random
import simpy
from typing import Iterable
from ..domain.batch import Batch
from ..domain.process import Process


def run_pipeline(batch: Batch, steps: Iterable[Process]) -> None:
    env = simpy.Environment()
    first_actionable_seen = False
    last_end = None

    for step in steps:
        if not step.is_actionable:
            print(f"--- {step.name} --- (On-Page Reference)")
            continue

        # mark Day-1 START at the first actionable step
        if not first_actionable_seen:
            batch.mark_arrival(env.now)   # usually 0.0
            first_actionable_seen = True

        print(f"Running {step.name} ...")
        env.process(_run_step(env, batch, step))
        env.run()                         # run until this step completes
        last_end = env.now                # remember the latest end time

    # mark Day-1 END after the last actionable finishes
    if first_actionable_seen and last_end is not None:
        batch.mark_complete(last_end)


def _run_step(env: simpy.Environment, batch: Batch, step: Process):
    """
    Simulate one step (no resource handling yet).
    """
    duration = random.uniform(step.duration_min_min, step.duration_max_min)
    yield env.timeout(duration)
    if step.run is not None:
        step.run(batch)
    print(
        f"{step.name} complete for batch {batch.batch_id} "
        f"({batch.occupancy} samples) by {step.actor_role}."
    )
