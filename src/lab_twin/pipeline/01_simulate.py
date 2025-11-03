# # lab_twin/pipeline/01_simulate.py
# from __future__ import annotations
# from pathlib import Path
# from datetime import datetime, timezone
# import random, yaml

# from lab_twin.domain.sample import Sample
# from lab_twin.domain.batch import Batch
# from lab_twin.sim.engine import run_sim_multi
# from lab_twin.workflow.steps_map import STEPS
# from lab_twin.utils.kpis import write_kpi_summary
# from lab_twin.utils.logger import write_samples_report

# # ---------- helpers ----------
# def arrivals_min_gap(n:int, gap:float)->list[float]:
#     return [i*gap for i in range(n)]

# def arrivals_wip_cap(n:int, min_gap:float, wip_cap:int, ct_est:float)->list[float]:
#     t=0.0; arr=[]
#     for _ in range(n):
#         if arr: t=max(t, arr[-1]+min_gap)
#         while True:
#             active=sum(1 for a in arr if t < a+ct_est)
#             if active < wip_cap: break
#             next_free=min(a+ct_est for a in arr if t < a+ct_est)
#             t=max(t, next_free)
#             if arr: t=max(t, arr[-1]+min_gap)
#         arr.append(t)
#     return arr

# def estimate_ct(samples_per_batch:int, seed:int)->float:
#     import pandas as pd
#     random.seed(seed)
#     b = Batch(batch_id="CT")
#     b.add_samples([Sample() for _ in range(samples_per_batch)])
#     run_id="ct_probe"; run_dir=Path("outputs_ct_probe"); run_dir.mkdir(parents=True, exist_ok=True)
#     run_sim_multi([b], STEPS, [0.0], seed=seed, run_dir=run_dir, run_id=run_id)
#     write_samples_report([b], datetime.now(timezone.utc), run_dir, run_id)
#     write_kpi_summary(run_dir, run_id, sla_min=999999)
#     return float(pd.read_csv(run_dir/"kpi_overall.csv").iloc[0]["makespan_min"])

# def main():
#     cfg = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())

#     # config with sensible defaults
#     outputs_root = Path(cfg.get("outputs_root", "outputs_pipeline"))
#     urgent_share = float(cfg.get("urgent_share", 0.20))      # 20% urgent
#     urgent_weeks = float(cfg.get("urgent_sla_weeks", 3.0))   # 3 weeks
#     routine_weeks = float(cfg.get("routine_sla_weeks", 6.0)) # 6 weeks
#     WEEK_MIN = 7*24*60
#     SLA_MAP = {"urgent": urgent_weeks*WEEK_MIN, "routine": routine_weeks*WEEK_MIN}

#     run_id = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     run_dir = outputs_root / run_id
#     run_dir.mkdir(parents=True, exist_ok=True)

#     n_batches = int(cfg.get("n_batches", 6))
#     spp       = int(cfg.get("samples_per_batch", 96))

#     # ---------- make batches with priority tags ----------
#     random.seed(42)
#     batches=[]
#     for i in range(n_batches):
#         b = Batch(batch_id=f"BATCH_{i+1:03d}")
#         # deterministic 20/80 split per batch (stable across runs)
#         n_urgent = max(1, int(round(urgent_share * spp)))
#         samples=[]
#         for j in range(spp):
#             s = Sample()
#             s.priority = "urgent" if j < n_urgent else "routine"
#             samples.append(s)
#         b.add_samples(samples)
#         batches.append(b)

#     # ---------- arrivals ----------
#     policy = cfg.get("arrival_policy", "min_gap")
#     min_gap = float(cfg.get("arrival_min_gap", 60.0))
#     if policy == "min_gap":
#         arrivals = arrivals_min_gap(n_batches, min_gap)
#     else:
#         wip_cap = int(cfg.get("wip_cap", 2))
#         ct_est = estimate_ct(spp, seed=7)
#         arrivals = arrivals_wip_cap(n_batches, min_gap, wip_cap, ct_est)

#     # ---------- simulate & KPIs ----------
#     run_sim_multi(batches, STEPS, arrivals, seed=42, run_dir=run_dir, run_id=run_id)
#
#     # Make sure the sample report includes 'priority' (see change #2 below)
#     write_samples_report(batches, datetime.now(timezone.utc), run_dir, run_id)

#     # NEW: pass the SLA MAP so KPIs include overall + per-priority SLA
#     write_kpi_summary(run_dir, run_id, sla_map=SLA_MAP)

#     print(f"✅ Baseline artifacts: {run_dir.resolve()}")

# if __name__=="__main__":
#     main()

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import random, sys, yaml
import pandas as pd

from lab_twin.domain.sample import Sample
from lab_twin.domain.batch import Batch
from lab_twin.sim.engine import run_sim_multi
from lab_twin.workflow.steps_map import STEPS
from lab_twin.utils.kpis import write_kpi_summary
from lab_twin.utils.logger import write_samples_report


def arrivals_min_gap(n:int, gap:float)->list[float]:
    # batch i released every `gap` minutes
    return [i*gap for i in range(n)]

def arrivals_wip_cap(n:int, min_gap:float, wip_cap:int, ct_est:float)->list[float]:
    """
    Release policy with WIP cap:
    - you can't start a new batch if there are already `wip_cap` batches
      still inside the system (based on estimated cycle time).
    - also enforces a minimum gap between starts.
    """
    t = 0.0
    arr = []
    for _ in range(n):
        # respect min_gap after last release
        if arr:
            t = max(t, arr[-1] + min_gap)

        while True:
            active = sum(1 for a in arr if t < a + ct_est)
            if active < wip_cap:
                break
            # wait until one active batch is expected to complete
            next_free = min(a + ct_est for a in arr if t < a + ct_est)
            t = max(t, next_free)
            if arr:
                t = max(t, arr[-1] + min_gap)

        arr.append(t)
    return arr

def estimate_ct(samples_per_batch:int, seed:int)->float:
    """
    Quick single-batch probe to guess cycle time (makespan).
    We simulate 1 batch alone, measure total duration.
    """
    random.seed(seed)
    b = Batch(batch_id="CT")
    b.add_samples([Sample() for _ in range(samples_per_batch)])

    run_id   = "ct_probe"
    run_dir  = Path("outputs_ct_probe")
    run_dir.mkdir(parents=True, exist_ok=True)

    run_sim_multi([b], STEPS, [0.0], seed=seed, run_dir=run_dir, run_id=run_id)
    write_samples_report([b], datetime.now(timezone.utc), run_dir, run_id)
    # very loose SLA so it doesn't matter for probe
    write_kpi_summary(run_dir, run_id, sla_map={"urgent": 1e9, "routine": 1e9})

    ct_df = pd.read_csv(run_dir / "kpi_overall.csv")
    return float(ct_df.iloc[0]["makespan_min"])


def load_scenario_config(scenario: str):
    """
    Load config.yaml, merge defaults + scenario block.
    """
    cfg_all = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())

    if "defaults" not in cfg_all:
        raise ValueError("config.yaml must contain a 'defaults' section")

    if scenario not in cfg_all:
        raise ValueError(f"Scenario '{scenario}' not found in config.yaml")

    base = dict(cfg_all["defaults"])
    override = dict(cfg_all[scenario])
    base.update(override)
    return base


def main(scenario: str):
    cfg = load_scenario_config(scenario)

    # pull values with safe defaults
    outputs_root    = Path(cfg.get("outputs_root", "outputs_pipeline"))
    n_batches       = int(cfg.get("n_batches", 6))
    spp             = int(cfg.get("samples_per_batch", 96))
    urgent_share    = float(cfg.get("urgent_share", 0.20))
    urgent_weeks    = float(cfg.get("urgent_sla_weeks", 6.0))
    routine_weeks   = float(cfg.get("routine_sla_weeks", 6.0))
    arrival_policy  = cfg.get("arrival_policy", "min_gap")
    min_gap         = float(cfg.get("arrival_min_gap", 60.0))
    wip_cap         = int(cfg.get("wip_cap", 2))

    # build per-priority SLA map (mins)
    WEEK_MIN = 5 * 12 * 60
    SLA_MAP = {
        "urgent":  urgent_weeks  * WEEK_MIN,
        "routine": routine_weeks * WEEK_MIN,
    }

    # create scenario folder
    outputs_root.mkdir(parents=True, exist_ok=True)
    run_id  = f"{scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = outputs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running scenario: {scenario} ===")
    print(f"Output folder: {run_dir}")

    # make batches with priority split
    random.seed(42)
    batches = []
    for i in range(n_batches):
        b = Batch(batch_id=f"BATCH_{i+1:03d}")
        n_urgent = max(1, int(round(urgent_share * spp)))
        samples = []
        for j in range(spp):
            s = Sample()
            s.priority = "urgent" if j < n_urgent else "routine"
            samples.append(s)
        b.add_samples(samples)
        batches.append(b)

    # build arrivals schedule
    if arrival_policy == "min_gap":
        arrivals = arrivals_min_gap(n_batches, min_gap)
    else:
        # controlled intake with WIP cap
        ct_est = estimate_ct(spp, seed=7)
        arrivals = arrivals_wip_cap(n_batches, min_gap, wip_cap, ct_est)

    # run sim
    run_sim_multi(batches, STEPS, arrivals, seed=42, run_dir=run_dir, run_id=run_id)

    # write logs
    write_samples_report(batches, datetime.now(timezone.utc), run_dir, run_id)
    write_kpi_summary(run_dir, run_id, sla_map=SLA_MAP)

    print(f"✅ Scenario {scenario} complete.")
    print(f"Artifacts saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    main(scenario)
