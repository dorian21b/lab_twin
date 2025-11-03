# lab_twin/opt/runner.py
from __future__ import annotations
import itertools, math, random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import matplotlib.pyplot as plt

from lab_twin.domain.sample import Sample
from lab_twin.domain.batch import Batch
from lab_twin.sim.engine import run_sim_multi
from lab_twin.workflow.steps_map import STEPS        # dict of Process
from lab_twin.utils.logger import write_samples_report
from lab_twin.utils.kpis import write_kpi_summary


# -----------------------------
# 1) Config & objective weights
# -----------------------------
@dataclass(frozen=True)
class OptConfig:
    n_batches: int = 6
    samples_per_batch: int = 96
    # <<< run many seeds for robustness >>>
    seeds: Sequence[int] = tuple(range(1, 21))   # 20 seeds: 1..20
    sla_min: float = 5 * 12 * 60  # minutes

    # Objective weights (tune for demo)
    alpha_avg: float = 1.0
    beta_p90: float = 0.5
    eta_sla: float = 2.0   # subtracted in J (higher SLA lowers J)
    zeta_wip: float = 0.0  # penalty on WIP proxy (0 = off)


# -----------------------------
# 2) Utility: build batches
# -----------------------------
def _build_batches(n_batches: int, samples_per_batch: int) -> list[Batch]:
    batches = []
    for i in range(n_batches):
        b = Batch(batch_id=f"BATCH_{i+1:03d}")
        b.add_samples([Sample() for _ in range(samples_per_batch)])
        batches.append(b)
    return batches


# -----------------------------
# 3) Arrival scheduling (simple)
# -----------------------------
def arrivals_from_min_gap(n: int, min_gap: float) -> list[float]:
    return [i * min_gap for i in range(n)]


# -----------------------------
# 4) One policy evaluation
# -----------------------------
def run_one_policy(
    out_root: Path,
    seed: int,
    n_batches: int,
    samples_per_batch: int,
    sla_min: float,
    min_gap: float,
    weights: tuple[float, float, float, float],  # alpha, beta, eta, zeta
) -> dict:
    alpha, beta, eta, zeta = weights

    run_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_gap{min_gap:.0f}_s{seed}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    batches = _build_batches(n_batches, samples_per_batch)
    arrival_times = arrivals_from_min_gap(n_batches, min_gap)

    run_sim_multi(
        batches=batches,
        steps=STEPS,
        arrival_times=arrival_times,
        seed=seed,
        run_dir=run_dir,
        run_id=run_id,
    )

    write_samples_report(batches, datetime.now(timezone.utc), run_dir=run_dir, run_id=run_id)
    write_kpi_summary(run_dir=run_dir, run_id=run_id, sla_min=sla_min)

    overall = pd.read_csv(run_dir / "kpi_overall.csv").iloc[0]

    avg_tat = float(overall["avg_tat_min"])
    p90_tat = float(overall["p90_tat_min"])
    sla_pct = float(overall.get("sla_pct", 0.0))
    makespan = float(overall["makespan_min"])
    n_batches_out = int(overall["n_batches"])
    n_samples_out = int(overall["n_samples"])

    # simple WIP proxy
    wip_proxy = (n_batches_out * samples_per_batch) * (avg_tat / max(1.0, makespan))

    # objective (lower is better)
    J = alpha * avg_tat + beta * p90_tat - eta * sla_pct + zeta * wip_proxy

    return {
        "run_id": run_id,
        "seed": seed,
        "min_gap": float(min_gap),
        "n_batches": n_batches_out,
        "n_samples": n_samples_out,
        "makespan_min": makespan,
        "avg_tat_min": avg_tat,
        "p90_tat_min": p90_tat,
        "sla_pct": sla_pct,
        "wip_proxy": wip_proxy,
        "J": J,
        "run_dir": str(run_dir),
    }


# -----------------------------
# 5) Grid / random search
# -----------------------------
def grid_search(
    out_root: str | Path = "outputs_opt",
    config: OptConfig = OptConfig(),
    min_gap_grid: Sequence[float] = (5, 10, 20, 30, 60, 120),
) -> pd.DataFrame:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    weights = (config.alpha_avg, config.beta_p90, config.eta_sla, config.zeta_wip)

    for min_gap, seed in itertools.product(min_gap_grid, config.seeds):
        row = run_one_policy(
            out_root=out_root,
            seed=seed,
            n_batches=config.n_batches,
            samples_per_batch=config.samples_per_batch,
            sla_min=config.sla_min,
            min_gap=min_gap,
            weights=weights,
        )
        rows.append(row)
        print(f"[OK] gap={min_gap:>4} seed={seed:>2} → J={row['J']:.1f}, SLA={row['sla_pct']:.1f}%")

    df = pd.DataFrame(rows).sort_values(["min_gap", "seed"]).reset_index(drop=True)
    df.to_csv(out_root / "summary_grid_raw.csv", index=False)

    # Aggregate by min_gap for CI plots
    agg = (
        df.groupby("min_gap")
          .agg(J_mean=("J", "mean"),
               J_std=("J", "std"),
               SLA_mean=("sla_pct", "mean"),
               SLA_std=("sla_pct", "std"),
               n=("J", "count"))
          .reset_index()
    )
    # 95% CI (z≈1.96, normal approx)
    z = 1.96
    agg["J_ci"] = z * agg["J_std"] / (agg["n"] ** 0.5)
    agg["SLA_ci"] = z * agg["SLA_std"] / (agg["n"] ** 0.5)
    agg.to_csv(out_root / "summary_grid_agg.csv", index=False)

    # Plots with CI
    _plot_ci(agg, out_root)

    print("\nTop 5 single runs by J:")
    print(df.sort_values("J").head(5)[["min_gap","seed","J","sla_pct","avg_tat_min","p90_tat_min","run_dir"]])

    print("\nAggregate by min_gap (mean ± 95% CI):")
    print(agg[["min_gap","J_mean","J_ci","SLA_mean","SLA_ci"]])
    return df


# -----------------------------
# 6) CI plots
# -----------------------------
def _plot_ci(agg: pd.DataFrame, out_root: Path) -> None:
    # J mean ± CI
    plt.figure()
    plt.errorbar(agg["min_gap"], agg["J_mean"], yerr=agg["J_ci"], marker="o", capsize=4)
    plt.xlabel("min_gap (min)")
    plt.ylabel("Objective J (mean ± 95% CI, lower=better)")
    plt.title("Policy sweep (multi-seed): J vs min_gap")
    plt.tight_layout()
    plt.savefig(out_root / "plot_J_vs_min_gap_CI.png", dpi=160)

    # SLA mean ± CI
    plt.figure()
    plt.errorbar(agg["min_gap"], agg["SLA_mean"], yerr=agg["SLA_ci"], marker="o", capsize=4)
    plt.xlabel("min_gap (min)")
    plt.ylabel("SLA % (mean ± 95% CI, higher=better)")
    plt.title("Policy sweep (multi-seed): SLA vs min_gap")
    plt.tight_layout()
    plt.savefig(out_root / "plot_SLA_vs_min_gap_CI.png", dpi=160)


# -----------------------------
# 7) CLI
# -----------------------------
if __name__ == "__main__":
    grid_search()
