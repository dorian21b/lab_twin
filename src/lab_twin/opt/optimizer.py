from __future__ import annotations
import itertools, random, math, json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Dict

import simpy
import pandas as pd

from lab_twin.domain.sample import Sample
from lab_twin.domain.batch import Batch
from lab_twin.workflow.steps_map import STEPS
from lab_twin.workflow.steps_scaled import make_steps_scaled
from lab_twin.utils.eventlog import EventLogger, EventRow
from lab_twin.utils.logger import write_samples_report
from lab_twin.utils.kpis import write_kpi_summary
from lab_twin.sim.arrivals import launch_batches
from lab_twin.sim.line_full import build_full_graph
from lab_twin.sim.resources import make_resources


# -----------------------------
# Config & objective weights
# -----------------------------
@dataclass(frozen=True)
class OptConfig:
    # workload
    n_batches: int = 6
    samples_per_batch: int = 96
    seeds: Sequence[int] = tuple(range(1, 11))
    min_gap: float = 60.0               # minutes between batch arrivals
    sla_min: float = 5 * 12 * 60        # 5 working days @ 12h/day (comparative, not calendar-aware)

    # objective weights
    alpha_avg: float = 1.0              # weight on avg TAT
    beta_p90: float = 0.5               # weight on p90 TAT
    eta_sla: float = 2.0                # reward per % SLA hit
    zeta_wip: float = 0.0               # (optional) WIP proxy

    # search
    n_random: int = 20                  # random samples before greedy
    n_greedy_iters: int = 10            # local improvement steps

    # decision spaces
    # capacity choices per resource
    cap_choices: Dict[str, Sequence[int]] = None
    # duration scale choices per step
    dur_choices: Dict[str, Sequence[float]] = None

    # optional “cost” per capacity unit (for soft budget penalty)
    cap_unit_cost: Dict[str, float] = None
    lambda_cost: float = 0.0            # penalty multiplier on added capacity cost

    def __post_init__(self):
        object.__setattr__(self, "cap_choices", self.cap_choices or {
            "Quantstudio 5": (1, 2, 3),
            "Novaseq X +":   (1, 2),
            "Hamilton Star": (1, 2),
            "Firefly +":     (1, 2),
        })
        object.__setattr__(self, "dur_choices", self.dur_choices or {
            # “do nothing” (1.00) plus “10% faster”
            "SEQUENCING":        (1.00, 0.90),
            "DNA_QC_PLATE":      (1.00, 0.90),
            "QC_LIBRARY_PLATE":  (1.00, 0.90),
        })
        object.__setattr__(self, "cap_unit_cost", self.cap_unit_cost or {
            # soft cost per extra parallel server beyond 1
            "Quantstudio 5": 1.0,
            "Novaseq X +":   3.0,
            "Hamilton Star": 1.0,
            "Firefly +":     1.0,
        })


# -----------------------------
# Utilities
# -----------------------------
def _build_batches(n_batches: int, samples_per_batch: int) -> list[Batch]:
    batches = []
    for i in range(n_batches):
        b = Batch(batch_id=f"BATCH_{i+1:03d}")
        b.add_samples([Sample() for _ in range(samples_per_batch)])
        batches.append(b)
    return batches

def arrivals_from_min_gap(n: int, min_gap: float) -> list[float]:
    return [i * min_gap for i in range(n)]

def _objective(avg_tat, p90_tat, sla_pct, wip_proxy, cfg: OptConfig, cap_overrides: dict[str, int]) -> float:
    base = (
        cfg.alpha_avg * avg_tat
        + cfg.beta_p90 * p90_tat
        - cfg.eta_sla * sla_pct
        + cfg.zeta_wip * wip_proxy
    )
    if cfg.lambda_cost > 0:
        # simple soft penalty: pay for capacity beyond 1
        extra_cost = 0.0
        for res, cap in cap_overrides.items():
            unit = cfg.cap_unit_cost.get(res, 0.0)
            if cap > 1:
                extra_cost += unit * (cap - 1)
        base += cfg.lambda_cost * extra_cost
    return base


# -----------------------------
# One scenario eval across seeds
# -----------------------------
def eval_policy(
    out_root: Path,
    cfg: OptConfig,
    cap_overrides: dict[str, int],
    duration_scale: dict[str, float],
    tag: str | None = None
) -> dict:
    rows = []
    for seed in cfg.seeds:
        random.seed(seed)

        # make run dir
        tag_caps = "caps[" + "+".join(f"{k}-{cap_overrides[k]}" for k in sorted(cap_overrides.keys())) + "]"
        tag_dur  = "dur[" + "+".join(f"{k}x{duration_scale.get(k,1.0):.2f}" for k in sorted(duration_scale.keys())) + "]" if duration_scale else "dur[none]"
        run_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_s{seed}_{tag_caps}_{tag_dur}"
        if tag:
            run_id = f"{tag}_{run_id}"
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # steps (scaled)
        steps_used = make_steps_scaled(STEPS, duration_scale) if duration_scale else STEPS

        # env + resources + graph
        env = simpy.Environment()
        resources = make_resources(env, overrides=cap_overrides)

        logger = EventLogger(run_dir / "events_report.csv")
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

        head, _ = build_full_graph(env, steps_used, log_event, resources)
        batches = _build_batches(cfg.n_batches, cfg.samples_per_batch)
        arrival_times = arrivals_from_min_gap(cfg.n_batches, cfg.min_gap)
        # launch & run
        from lab_twin.sim.arrivals import launch_batches
        launch_batches(env, head, batches, arrival_times)
        env.run()
        logger.close()

        # reports / kpis (use UTC timestamp like your other runners)
        from datetime import datetime as dt
        write_samples_report(batches, dt.now(timezone.utc), run_dir=run_dir, run_id=run_id)
        write_kpi_summary(run_dir=run_dir, run_id=run_id, sla_min=cfg.sla_min)

        # KPIs
        overall = pd.read_csv(run_dir / "kpi_overall.csv").iloc[0]
        avg_tat = float(overall["avg_tat_min"])
        p90_tat = float(overall["p90_tat_min"])
        sla_pct = float(overall.get("sla_pct", 0.0))
        makespan = float(overall["makespan_min"])
        n_batches_out = int(overall["n_batches"])
        wip_proxy = (n_batches_out * cfg.samples_per_batch) * (avg_tat / max(1.0, makespan))
        J = _objective(avg_tat, p90_tat, sla_pct, wip_proxy, cfg, cap_overrides)

        rows.append({
            "run_id": run_id,
            "seed": seed,
            "cap_overrides": cap_overrides,
            "duration_scale": duration_scale,
            "avg_tat_min": avg_tat,
            "p90_tat_min": p90_tat,
            "sla_pct": sla_pct,
            "makespan_min": makespan,
            "J": J,
            "run_dir": str(run_dir),
        })

    df = pd.DataFrame(rows)
    # aggregate across seeds
    agg = df.agg({
        "avg_tat_min":"mean",
        "p90_tat_min":"mean",
        "sla_pct":"mean",
        "makespan_min":"mean",
        "J":"mean"
    }).to_dict()
    agg["J_sem"] = df["J"].sem()
    agg["n_runs"] = len(df)
    agg["cap_overrides"] = cap_overrides
    agg["duration_scale"] = duration_scale
    # keep last run_dir for convenience
    agg["last_run_dir"] = rows[-1]["run_dir"]
    return agg


# -----------------------------
# Search helpers
# -----------------------------
def _all_choices(space: Dict[str, Sequence]) -> list[dict]:
    keys = sorted(space.keys())
    combos = []
    for values in itertools.product(*[space[k] for k in keys]):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos

def _neighbor_caps(caps: dict[str,int], cfg: OptConfig) -> list[dict]:
    nbrs = []
    for k, choices in cfg.cap_choices.items():
        i = choices.index(caps[k])
        if i > 0: nbrs.append({**caps, k: choices[i-1]})
        if i < len(choices)-1: nbrs.append({**caps, k: choices[i+1]})
    return nbrs

def _neighbor_durs(durs: dict[str,float], cfg: OptConfig) -> list[dict]:
    nbrs = []
    for k, choices in cfg.dur_choices.items():
        cur = durs.get(k, 1.0)
        i = choices.index(cur)
        if i > 0: nbrs.append({**durs, k: choices[i-1]})
        if i < len(choices)-1: nbrs.append({**durs, k: choices[i+1]})
    return nbrs


# -----------------------------
# Main optimize()
# -----------------------------
def optimize(
    out_root: str | Path = "outputs_opt_caps_caps",
    cfg: OptConfig = OptConfig(),
    fix_caps: dict[str,int] | None = None,       # to pin certain resources
) -> pd.DataFrame:
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)

    # initialize at lower bounds
    caps0 = {k: choices[0] for k, choices in cfg.cap_choices.items()}
    durs0 = {k: 1.0 for k in cfg.dur_choices.keys()}

    # apply fixed capacities if provided
    if fix_caps:
        for k,v in fix_caps.items():
            if k in caps0:
                caps0[k] = v

    results = []
    best = eval_policy(out_root, cfg, caps0, durs0, tag="init")
    results.append(best)
    bestJ = best["J"]
    best_caps = caps0
    best_durs = durs0
    print(f"[INIT] J={bestJ:.1f} caps={best_caps} durs={best_durs}")

    # ---- random search
    for i in range(cfg.n_random):
        rand_caps = {k: random.choice(v) for k, v in cfg.cap_choices.items()}
        if fix_caps:
            for k,v in fix_caps.items():
                rand_caps[k] = v
        rand_durs = {k: random.choice(v) for k, v in cfg.dur_choices.items()}
        row = eval_policy(out_root, cfg, rand_caps, rand_durs, tag=f"rand{i+1}")
        results.append(row)
        if row["J"] < bestJ:
            bestJ = row["J"]; best_caps = rand_caps; best_durs = rand_durs
            print(f"[RAND]  i={i+1}  J={bestJ:.1f}  caps={best_caps} durs={best_durs}")

    # ---- greedy local improvement (coordinate descent on caps & durs)
    for it in range(cfg.n_greedy_iters):
        improved = False
        # capacities first
        for nbr_caps in _neighbor_caps(best_caps, cfg):
            if fix_caps:
                for k,v in fix_caps.items():
                    nbr_caps[k] = v
            row = eval_policy(out_root, cfg, nbr_caps, best_durs, tag=f"greedy{it+1}_caps")
            results.append(row)
            if row["J"] < bestJ:
                bestJ = row["J"]; best_caps = nbr_caps; improved = True
                print(f"[GREEDY-CAP] it={it+1} J={bestJ:.1f} caps={best_caps} durs={best_durs}")
        # durations next
        for nbr_durs in _neighbor_durs(best_durs, cfg):
            row = eval_policy(out_root, cfg, best_caps, nbr_durs, tag=f"greedy{it+1}_dur")
            results.append(row)
            if row["J"] < bestJ:
                bestJ = row["J"]; best_durs = nbr_durs; improved = True
                print(f"[GREEDY-DUR] it={it+1} J={bestJ:.1f} caps={best_caps} durs={best_durs}")
        if not improved:
            print(f"[GREEDY] early stop at it={it+1}")
            break

    df = pd.DataFrame(results)
    df.to_csv(out_root / "search_history.csv", index=False)
    print("\n=== BEST FOUND ===")
    print(f"J={bestJ:.1f}")
    print(f"caps={best_caps}")
    print(f"durs={best_durs}")
    print(f"mean SLA={df[df['cap_overrides'].apply(lambda d: d==best_caps) & df['duration_scale'].apply(lambda d: d==best_durs)]['sla_pct'].mean():.1f}%")
    return df


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    cfg = OptConfig(
        n_batches=6,
        samples_per_batch=96,
        seeds=tuple(range(1, 8)),   # fewer for speed; increase later
        min_gap=60.0,
        sla_min=5*12*60,
        n_random=12,
        n_greedy_iters=8,
        lambda_cost=0.0,            # set >0 to penalize added capacity
    )
    # Example: pin Novaseq capacity at 1 if you know you can't add one.
    fix_caps = {
        # "Novaseq X +": 1
    }
    optimize(out_root="outputs_opt_caps", cfg=cfg, fix_caps=fix_caps)
