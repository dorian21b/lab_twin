import random, simpy
from pathlib import Path
from typing import Iterable
from lab_twin.domain.batch import Batch
from lab_twin.domain.process import Process
from lab_twin.sim.resources import make_resources
from lab_twin.utils.eventlog import EventLogger, EventRow
from .line import build_line
from .arrivals import launch_batches
from .line_full import build_full_graph
from lab_twin.sim.duration_scale import set_duration_scale, set_known_steps, scale_service_time  # <-- ADD

def run_sim(
    batch: Batch,
    steps: Iterable[Process],
    seed: int | None = 42,
    run_dir: str | Path = "outputs",
    run_id: str | None = None,        # <-- add (optional)
    duration_scale: dict[str, float] | None = None,                 # <-- ADD
) -> None:
    if seed is not None:
        random.seed(seed)

    # ensure path
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # derive a run_id if not provided (use folder name or fallback)
    if run_id is None:
        run_id = run_dir.name or f"run_{seed}"

    set_duration_scale(duration_scale or {}, steps, verbose=True)

    env = simpy.Environment()
    resources = make_resources(env)
    logger = EventLogger(run_dir / "events_report.csv")

    print(f"\n=== Starting engine for batch {batch.batch_id} ===")

    first_actionable_seen = False
    last_end = None

    for step in steps:
        if not step.is_actionable:
            print(f"--- {step.name} --- (On-Page Reference)")
            continue

        if not first_actionable_seen:
            batch.mark_arrival(env.now)
            first_actionable_seen = True

        print(f"Running {step.name} ...")
        # pass run_id down
        env.process(_run_step(env, batch, step, resources, logger, run_id))
        env.run()
        last_end = env.now

    if first_actionable_seen and last_end is not None:
        batch.mark_complete(last_end)

    logger.close()
    print(f"=== Engine completed for batch {batch.batch_id} ===")

def _run_step(env, batch: Batch, step: Process, resources, logger: EventLogger, run_id: str):
    service_time = random.uniform(step.duration_min_min, step.duration_max_min)
    service_time = scale_service_time(step.process_id, service_time, log_fn=print)
    resource_name = step.resources[0] if step.resources else (step.actor_role or "")
    sim_start = env.now
    wait_min = 0.0
    q_len = 0

    try:
        if resource_name and resource_name in resources:
            res = resources[resource_name]
            q_len = len(res.queue)
            queue_enter = env.now
            with res.request() as req:
                yield req
                wait_min = env.now - queue_enter
                yield env.timeout(service_time)
        else:
            yield env.timeout(service_time)

        if step.run:
            step.run(batch)

        status = "COMPLETED"

    except Exception as e:
        status = "FAILED"
        logger.write(EventRow(
            run_id, batch.batch_id, step.process_id,
            sim_start, env.now, env.now - sim_start,
            wait_min, q_len, status, resource_name, note=str(e)
        ))
        raise

    sim_end = env.now
    logger.write(EventRow(
        run_id, batch.batch_id, step.process_id,
        sim_start, sim_end, service_time, wait_min, q_len, status, resource_name
    ))

    print(f"{step.name} complete for batch {batch.batch_id} ({batch.occupancy} samples) by {step.actor_role}.")


def run_sim_multi(
    batches, steps,
    arrival_times,                     # e.g., [0, 5, 10] minutes
    seed: int | None = 42,
    run_dir: str | Path = "outputs",
    run_id: str = "multi",
    duration_scale: dict[str, float] | None = None,
    resource_overrides: dict[str,int] | None = None,
):

    if seed is not None:
        random.seed(seed)
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)

    set_duration_scale(duration_scale or {}, steps, verbose=True)


    env = simpy.Environment()
    logger = EventLogger(run_dir / f"events_report.csv")

    def log_event(**row):
        logger.write(EventRow(
            run_id,
            row["batch_id"],
            row["process_id"],
            row["sim_start_min"],
            row["sim_end_min"],
            row["service_min"],
            row["wait_min"],
            row["queue_len_on_arrival"],
            row["status"],
            row.get("resource_name",""),
            row.get("note",""),
        ))

    resources = make_resources(env, overrides=resource_overrides)

    head, _tails = build_full_graph(env, steps, log_event, resources=resources)
    launch_batches(env, head, batches, arrival_times)
    env.run()
    logger.close()
