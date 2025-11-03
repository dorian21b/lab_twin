from __future__ import annotations
import itertools, random, math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import matplotlib.pyplot as plt

from lab_twin.domain.sample import Sample
from lab_twin.domain.batch import Batch
from lab_twin.sim.engine import run_sim_multi
from lab_twin.workflow.steps_map import STEPS
from lab_twin.workflow.steps_scaled import make_steps_scaled
from lab_twin.utils.logger import write_samples_report
from lab_twin.utils.kpis import write_kpi_summary

# -----------------------------
# Config & objective weights
# -----------------------------
@dataclass(frozen=True)
class TimeOptConfig:
    n_batches: int = 6
    samples_per_batch: int = 96
    seeds: Sequence[int] = tuple(range(1, 16))  # 15 seeds for robustness
    sla_min: float = 5 * 12 * 60               # 5 days in minutes

    # Objective weights (same philosophy as your runner)
    alpha_avg: float = 1.0
    beta_p90: float = 0.5
    eta_sla: float = 2.0
    zeta_wip: float = 0.0

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

def _objective(avg_tat, p90_tat, sla_pct, wip_proxy, cfg: TimeOptConfig) -> float:
    return (
        cfg.alpha_avg * avg_tat
        + cfg.beta_p90 * p90_tat
        - cfg.eta_sla * sla_pct
        + cfg.zeta_wip * wip_proxy
    )

# -----------------------------
# One policy eval (time-only)
# -----------------------------
def run_one_policy_time_only(
    out_root: Path,
    seed: int,
    cfg: TimeOptConfig,
    min_gap: float,
    duration_scale: dict[str, float],  # e.g. {"SEQUENCING":0.9}
) -> dict:
    random.seed(seed)

    tag_scale = "none" if not duration_scale else "_".join(f"{k}x{v:.2f}" for k,v in sorted(duration_scale.items()))
    run_id = f"topt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_gap{min_gap:.0f}_s{seed}_dur[{tag_scale}]"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Scale steps
    steps_scaled = make_steps_scaled(STEPS, duration_scale) if duration_scale else STEPS

    # Build & simulate
    batches = _build_batches(cfg.n_batches, cfg.samples_per_batch)
    arrival_times = arrivals_from_min_gap(cfg.n_batches, min_gap)

    run_sim_multi(batches, steps_scaled, arrival_times, seed=seed, run_dir=run_dir, run_id=run_id)

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

    # Little-ish WIP proxy
    wip_proxy = (n_batches_out * cfg.samples_per_batch) * (avg_tat / max(1.0, makespan))
    J = _objective(avg_tat, p90_tat, sla_pct, wip_proxy, cfg)

    return {
        "run_id": run_id,
        "seed": seed,
        "min_gap": min_gap,
        "duration_scale": duration_scale or {},
        "n_batches": n_batches_out,
        "n_samples": int(overall["n_samples"]),
        "makespan_min": makespan,
        "avg_tat_min": avg_tat,
        "p90_tat_min": p90_tat,
        "sla_pct": sla_pct,
        "wip_proxy": wip_proxy,
        "J": J,
        "run_dir": str(run_dir),
    }

# -----------------------------
# Grid / multi-seed sweep
# -----------------------------
def sweep_time_only(
    out_root: str | Path = "outputs_timeopt",
    cfg: TimeOptConfig = TimeOptConfig(),
    min_gaps: Sequence[float] = (10, 30, 60, 120),
    # Choose a small, convincing set of scenarios for the deck
    duration_scenarios: Sequence[dict[str, float]] = (
        {},  # baseline
        {"SEQUENCING": 0.9},
        {"QC_LIBRARY_PLATE": 0.9},
        {"DNA_QC_PLATE": 0.9},
        {"ROBOT_POOLING": 0.9},
        {"SEQUENCING": 0.9, "DNA_QC_PLATE": 0.9},  # combo
    ),
) -> pd.DataFrame:
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed, gap, scale in itertools.product(cfg.seeds, min_gaps, duration_scenarios):
        row = run_one_policy_time_only(out_root, seed, cfg, gap, scale)
        rows.append(row)
        print(f"[OK] seed={seed} gap={gap} dur={scale} → J={row['J']:.1f}, SLA={row['sla_pct']:.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "summary_time_only.csv", index=False)

    # Aggregate by (min_gap, scenario) across seeds
    def label(scale: dict[str,float]) -> str:
        if not scale: return "baseline"
        return "+".join(f"{k}x{v:.2f}" for k,v in sorted(scale.items()))
    df["scenario"] = df["duration_scale"].apply(label)

    grp = df.groupby(["min_gap","scenario"]).agg(
        J_mean=("J","mean"), J_sem=("J","sem"),
        SLA_mean=("sla_pct","mean"), SLA_sem=("sla_pct","sem"),
        avg_tat_mean=("avg_tat_min","mean"), p90_tat_mean=("p90_tat_min","mean")
    ).reset_index()
    grp["J_ci"] = 1.96 * grp["J_sem"].fillna(0.0)
    grp["SLA_ci"] = 1.96 * grp["SLA_sem"].fillna(0.0)
    grp.to_csv(out_root / "summary_time_only_agg.csv", index=False)

    try:
        _quick_plots_time_only(df, grp, out_root)
    except Exception as e:
        print("Plotting skipped:", e)

    print("\nTop (mean J) by scenario and min_gap:")
    print(grp.sort_values("J_mean").head(10))
    return df

# -----------------------------
# Per-process sensitivity (10%)
# -----------------------------
def sensitivity_time_only(
    out_root: str | Path = "outputs_timeopt",
    cfg: TimeOptConfig = TimeOptConfig(),
    min_gap: float = 60.0,
    target_keys: Sequence[str] = (
        "SEQUENCING","QC_LIBRARY_PLATE","DNA_QC_PLATE",
        "ROBOT_POOLING","QC_POOLS","QUANTIFICATION_READING",
        "NUCLEIC_ACID_QC","CREATE_DILUTION_TUBES","CREATE_PLATE_FOR_TRANSFER_TO_HUB"
    ),
    reduction: float = 0.9,   # 10% faster
) -> pd.DataFrame:
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)

    # baseline
    base_rows = [run_one_policy_time_only(out_root, s, cfg, min_gap, {}) for s in cfg.seeds]
    base = pd.DataFrame(base_rows); base["scenario"] = "baseline"
    base_J = base["J"].mean()

    rows = []
    for key in target_keys:
        scale = {key: reduction}
        for s in cfg.seeds:
            rows.append(run_one_policy_time_only(out_root, s, cfg, min_gap, scale))
    df = pd.DataFrame(rows)
    df["scenario"] = df["duration_scale"].apply(lambda d: next(iter(d)) + f"x{reduction:.2f}")

    grp = df.groupby("scenario").agg(
        J_mean=("J","mean"),
        J_sem=("J","sem"),
        SLA_mean=("sla_pct","mean"),
        avg_tat_mean=("avg_tat_min","mean"),
        p90_tat_mean=("p90_tat_min","mean")
    ).reset_index()
    grp["J_ci"] = 1.96 * grp["J_sem"].fillna(0.0)
    grp["delta_J_vs_base"] = grp["J_mean"] - base_J
    grp = grp.sort_values("delta_J_vs_base")
    grp.to_csv(out_root / f"sensitivity_min_gap_{int(min_gap)}.csv", index=False)

    try:
        plt.figure()
        plt.barh(grp["scenario"], -grp["delta_J_vs_base"])  # positive = improvement
        plt.xlabel("Improvement in J vs baseline (positive = better)")
        plt.title(f"Per-process sensitivity (10% faster) at min_gap={min_gap:.0f}")
        plt.tight_layout()
        plt.savefig(out_root / f"plot_sensitivity_gap{int(min_gap)}.png", dpi=160)
    except Exception as e:
        print("Sensitivity plotting skipped:", e)

    print("\nPer-process sensitivity (ΔJ mean vs baseline):")
    print(grp[["scenario","delta_J_vs_base","J_ci","SLA_mean","avg_tat_mean","p90_tat_mean"]].head(12))
    return grp

# -----------------------------
# Plots
# -----------------------------
def _quick_plots_time_only(df: pd.DataFrame, grp: pd.DataFrame, out_root: Path):
    # J (mean±CI) by scenario grouped by min_gap
    for gap in sorted(grp["min_gap"].unique()):
        g = grp[grp["min_gap"] == gap].sort_values("J_mean")
        plt.figure()
        plt.errorbar(range(len(g)), g["J_mean"], yerr=g["J_ci"], fmt="o-")
        plt.xticks(range(len(g)), g["scenario"], rotation=30, ha="right")
        plt.ylabel("Objective J (lower is better)")
        plt.title(f"J by scenario (min_gap={gap:.0f})")
        plt.tight_layout()
        plt.savefig(out_root / f"plot_J_gap_{int(gap)}.png", dpi=160)

    # SLA (mean±CI) by scenario for a single good gap (pick the best mean J gap)
    best_gap = grp.groupby("min_gap")["J_mean"].mean().idxmin()
    g = grp[grp["min_gap"] == best_gap]
    plt.figure()
    plt.errorbar(range(len(g)), g["SLA_mean"], yerr=g["SLA_ci"], fmt="o-")
    plt.xticks(range(len(g)), g["scenario"], rotation=30, ha="right")
    plt.ylabel("SLA % (higher is better)")
    plt.title(f"SLA by scenario (min_gap={best_gap:.0f})")
    plt.tight_layout()
    plt.savefig(out_root / f"plot_SLA_gap_{int(best_gap)}.png", dpi=160)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    # 1) time-only sweep across min_gap & scenarios
    sweep_time_only()

    # 2) per-process sensitivity at a chosen min_gap (e.g., 60)
    sensitivity_time_only(min_gap=60.0)
