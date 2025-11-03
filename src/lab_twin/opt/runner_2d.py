# lab_twin/opt/runner_2d.py
from __future__ import annotations
import itertools, math, random, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lab_twin.domain.sample import Sample
from lab_twin.domain.batch import Batch
from lab_twin.sim.engine import run_sim_multi
from lab_twin.workflow.steps_map import STEPS
from lab_twin.utils.logger import write_samples_report
from lab_twin.utils.kpis import write_kpi_summary
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# Config & objective weights
# -----------------------------
@dataclass(frozen=True)
class OptConfig:
    n_batches: int = 6
    samples_per_batch: int = 96
    seeds: Sequence[int] = tuple(range(1, 21))  # 20 seeds for robustness
    sla_min: float = 5 * 12 * 60               # 5 working days in minutes

    # Objective weights
    alpha_avg: float = 1.0   # avg TAT
    beta_p90: float = 0.5    # P90 TAT
    eta_sla: float = 2.0     # SLA% (subtracted → higher SLA lowers J)
    zeta_wip: float = 0.0    # WIP proxy penalty (optional)


# -----------------------------
# Shared dark theme bits
# -----------------------------
_BG_COLOR = "#002E54"  # deep navy dashboard background

# SINGLE_HUE_CMAP = LinearSegmentedColormap.from_list(
#     "navy_cyan",
#     [
#         "#001233",  # very dark blue (low values)
#         "#0056A3",  # mid-dark blue
#         "#00B4D8",  # teal/cyan mid-high
#         "#90E0EF",  # pale cyan (high values)
#     ],
# )

SINGLE_HUE_CMAP = LinearSegmentedColormap.from_list(
    "navy_cyan",
    [
        "#2c3170",  # mid-dark blue
        "#4361ee",  # very dark blue (low values)
    ],
)


def _plot_heatmap(
    Z: np.ndarray,
    x_vals,
    y_vals,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: Path,
    cmap_name="viridis",  # can be a string OR a Colormap object
    vmin=None,
    vmax=None,
    annotate: bool = True,
    value_fmt: str = ".0f",
) -> None:
    """
    Generic dark-themed heatmap saver.
    Z should be shape (len(y_vals), len(x_vals)).
    """

    width = 15.92 * 0.618 * 0.618 * 0.618
    fig, ax = plt.subplots(figsize=(width, width))

    hm = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
    )

    # ticks / ticklabels
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([str(v) for v in y_vals])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # colorbar
    cbar = plt.colorbar(hm, ax=ax)

    # --- Apply dark theme styling ---
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    # keep only bottom spine, make it visible + white
    for spine_name, spine in ax.spines.items():
        if spine_name == "bottom":
            spine.set_visible(True)
            spine.set_color("white")
            spine.set_linewidth(1.0)
        else:
            spine.set_visible(False)

    # ticks outward, white, padded
    ax.tick_params(
        axis="both",
        which="major",
        length=8,
        width=1,
        color="white",
        pad=6,
        labelsize=11,
    )
    # set tick label colors explicitly
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color("white")

    # axis label styles
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)

    # title, bold, left-aligned
    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
        color="white",
        fontsize=11,
        pad=15,
    )

    # colorbar styling to match dark bg
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(1.0)
    cbar.ax.tick_params(colors="white", length=6, width=1, labelsize=10)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.figure.patch.set_facecolor(_BG_COLOR)
    cbar.ax.set_facecolor(_BG_COLOR)

    # annotate values in each cell (centered white text)
    if annotate:
        n_y, n_x = Z.shape
        for i in range(n_y):
            for j in range(n_x):
                val = Z[i, j]
                ax.text(
                    j,
                    i,
                    format(val, value_fmt),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                )

    fig.tight_layout()

    # --- save raster (PNG) and vector (PDF) ---
    png_path = outpath.with_suffix(".png")
    pdf_path = outpath.with_suffix(".pdf")

    # High-res PNG for slides / docs
    fig.savefig(png_path, dpi=300, facecolor=_BG_COLOR, bbox_inches="tight")

    # Vector PDF for infinite zoom / Illustrator
    fig.savefig(pdf_path, facecolor=_BG_COLOR, bbox_inches="tight")

    plt.close(fig)


# -----------------------------
# Build batches
# -----------------------------
def _build_batches(n_batches: int, samples_per_batch: int) -> List[Batch]:
    batches = []
    for i in range(n_batches):
        b = Batch(batch_id=f"BATCH_{i+1:03d}")
        b.add_samples([Sample() for _ in range(samples_per_batch)])
        batches.append(b)
    return batches


# -----------------------------
# Cycle-time estimate
# -----------------------------
def estimate_ct_single_batch(
    out_root: Path,
    samples_per_batch: int,
    seed: int,
) -> float:
    """
    Run a quick single-batch simulation to get a data-driven cycle-time estimate.
    We use the overall makespan as ct_est (close enough for release control).
    """
    run_id = f"ctprobe_{datetime.now().strftime('%Y%m%d_%H%M%S')}_s{seed}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    batches = _build_batches(n_batches=1, samples_per_batch=samples_per_batch)
    arrival_times = [0.0]

    run_sim_multi(
        batches,
        STEPS,
        arrival_times,
        seed=seed,
        run_dir=run_dir,
        run_id=run_id,
    )
    write_samples_report(
        batches,
        datetime.now(timezone.utc),
        run_dir=run_dir,
        run_id=run_id,
    )
    write_kpi_summary(run_dir=run_dir, run_id=run_id, sla_min=6 * 5 * 12 * 60)

    overall = pd.read_csv(run_dir / "kpi_overall.csv").iloc[0]
    ct_est = float(overall["makespan_min"])  # conservative & simple
    return ct_est


# -----------------------------
# Arrival scheduler with WIP cap
# -----------------------------
def schedule_with_wip_cap(
    n: int,
    min_gap: float,
    wip_cap: int,
    ct_est: float,
) -> List[float]:
    """
    Create arrival times that respect both:
      - a minimum release spacing (min_gap),
      - a WIP cap (no more than wip_cap active batches at any time).

    Active if: t < arrival_k + ct_est
    """
    arrivals: List[float] = []
    t = 0.0
    for i in range(n):
        # Respect min_gap against last release
        if arrivals:
            t = max(t, arrivals[-1] + min_gap)

        # Enforce WIP cap: if too many active at candidate t, push t forward
        while True:
            active = sum(1 for a in arrivals if t < a + ct_est)
            if active < wip_cap:
                break
            # jump to earliest completion among active to free capacity
            next_free = min(a + ct_est for a in arrivals if t < a + ct_est)
            t = max(t, next_free)
            # also ensure min_gap vs the last arrival still holds
            if arrivals:
                t = max(t, arrivals[-1] + min_gap)

        arrivals.append(t)
    return arrivals


# -----------------------------
# One policy evaluation
# -----------------------------
def run_one_policy(
    out_root: Path,
    seed: int,
    n_batches: int,
    samples_per_batch: int,
    sla_min: float,
    min_gap: float,
    wip_cap: int,
    ct_est: float,
    weights: tuple[float, float, float, float],  # alpha, beta, eta, zeta
) -> dict:
    alpha, beta, eta, zeta = weights

    run_id = (
        f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_gap{min_gap:g}_w{wip_cap}_s{seed}"
    )
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    random.seed(seed)
    batches = _build_batches(n_batches, samples_per_batch)
    arrival_times = schedule_with_wip_cap(n_batches, min_gap, wip_cap, ct_est)

    # Simulate
    run_sim_multi(
        batches,
        STEPS,
        arrival_times,
        seed=seed,
        run_dir=run_dir,
        run_id=run_id,
    )
    write_samples_report(
        batches,
        datetime.now(timezone.utc),
        run_dir=run_dir,
        run_id=run_id,
    )
    write_kpi_summary(
        run_dir=run_dir,
        run_id=run_id,
        sla_min=sla_min,
    )

    # KPIs
    overall = pd.read_csv(run_dir / "kpi_overall.csv").iloc[0]
    avg_tat = float(overall["avg_tat_min"])
    p90_tat = float(overall["p90_tat_min"])
    sla_pct = float(overall.get("sla_pct", 0.0))
    makespan = float(overall["makespan_min"])
    n_batches_out = int(overall["n_batches"])
    n_samples_out = int(overall["n_samples"])

    # WIP proxy (optional, rough)
    wip_proxy = (n_batches_out * samples_per_batch) * (
        avg_tat / max(1.0, makespan)
    )

    # Objective (lower is better)
    J = alpha * avg_tat + beta * p90_tat - eta * sla_pct + zeta * wip_proxy

    return {
        "run_id": run_id,
        "seed": seed,
        "min_gap": float(min_gap),
        "wip_cap": int(wip_cap),
        "n_batches": n_batches_out,
        "n_samples": n_samples_out,
        "makespan_min": makespan,
        "avg_tat_min": avg_tat,
        "p90_tat_min": p90_tat,
        "sla_pct": sla_pct,
        "wip_proxy": wip_proxy,
        "J": J,
        "ct_est": ct_est,
        "run_dir": str(run_dir),
    }


# -----------------------------
# Heatmaps (wrapper for both J and SLA)
# -----------------------------
def _heatmaps(agg: pd.DataFrame, out_root: Path) -> None:
    piv_J = agg.pivot(index="wip_cap", columns="min_gap", values="J_mean")
    piv_SLA = agg.pivot(index="wip_cap", columns="min_gap", values="SLA_mean")

    _plot_heatmap(
        Z=piv_J.values,
        x_vals=list(piv_J.columns),
        y_vals=list(piv_J.index),
        xlabel="Minimum Release Gap (min)",
        ylabel="WIP cap",
        # title="Mean Objective J (lower = better)",
        title="Average Objective Score",
        outpath=out_root / "heatmap_J",          # <-- no extension
        cmap_name=SINGLE_HUE_CMAP,
        vmin=None,
        vmax=None,
        annotate=True,
        value_fmt=".0f",
    )

    _plot_heatmap(
        Z=piv_SLA.values,
        x_vals=list(piv_SLA.columns),
        y_vals=list(piv_SLA.index),
        xlabel="Minimum Release Gap (min)",
        ylabel="WIP cap",
        # title="Mean SLA % (higher = better)",
        title="Average SLA Compliance (%)",
        outpath=out_root / "heatmap_SLA",        # <-- no extension
        cmap_name=SINGLE_HUE_CMAP,
        vmin=0,
        vmax=100,
        annotate=True,
        value_fmt=".0f",
    )




# -----------------------------
# 2-D grid search (multi-seed)
# -----------------------------
def grid_search_2d(
    out_root: str | Path = "outputs_opt_2d",
    config: OptConfig = OptConfig(),
    min_gap_grid: Sequence[float] = (5, 10, 20, 40, 60, 80),
    wip_caps: Sequence[int] = (2, 3, 4, 5),
) -> pd.DataFrame:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Cycle-time estimate once (use first seed)
    ct_seed = config.seeds[0] if len(config.seeds) else 42
    ct_est = estimate_ct_single_batch(
        out_root,
        config.samples_per_batch,
        seed=ct_seed,
    )
    print(
        f"[ct_est] Using cycle-time estimate ≈ {ct_est:.1f} min (from single-batch probe)"
    )

    rows = []
    weights = (
        config.alpha_avg,
        config.beta_p90,
        config.eta_sla,
        config.zeta_wip,
    )

    # 2) Sweep
    for min_gap, wip_cap, seed in itertools.product(
        min_gap_grid, wip_caps, config.seeds
    ):
        row = run_one_policy(
            out_root=out_root,
            seed=seed,
            n_batches=config.n_batches,
            samples_per_batch=config.samples_per_batch,
            sla_min=config.sla_min,
            min_gap=min_gap,
            wip_cap=wip_cap,
            ct_est=ct_est,
            weights=weights,
        )
        rows.append(row)
        print(
            f"[OK] gap={min_gap:>4}  wip_cap={wip_cap}  seed={seed:>2} "
            f"→ J={row['J']:.1f}, SLA={row['sla_pct']:.1f}%"
        )

    df = (
        pd.DataFrame(rows)
        .sort_values(["J", "min_gap", "wip_cap"])
        .reset_index(drop=True)
    )
    df.to_csv(out_root / "summary_2d.csv", index=False)

    # 3) Aggregate across seeds
    agg = (
        df.groupby(["min_gap", "wip_cap"])
        .agg(
            J_mean=("J", "mean"),
            J_std=("J", "std"),
            SLA_mean=("sla_pct", "mean"),
            SLA_std=("sla_pct", "std"),
            n=("J", "count"),
        )
        .reset_index()
    )

    # 95% CI
    agg["J_ci"] = 1.96 * agg["J_std"] / np.sqrt(agg["n"].clip(lower=1))
    agg["SLA_ci"] = 1.96 * agg["SLA_std"] / np.sqrt(agg["n"].clip(lower=1))

    agg.to_csv(out_root / "agg_2d.csv", index=False)

    # 4) Plots
    try:
        _heatmaps(agg, out_root)
    except Exception as e:
        print("Heatmap plotting skipped:", e)

    # 5) Report best policies
    best = agg.sort_values("J_mean").head(5)
    print("\nTop 5 policies by mean J (±95% CI):")
    print(best[["min_gap", "wip_cap", "J_mean", "J_ci", "SLA_mean", "SLA_ci"]])

    return df


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    df = grid_search_2d(
        out_root="outputs_opt_2d",
        config=OptConfig(
            n_batches=6,
            samples_per_batch=96,
            seeds=tuple(range(1, 21)),   # 20 seeds for robustness
            sla_min=5 * 12 * 60,
            alpha_avg=1.0,
            beta_p90=0.5,
            eta_sla=2.0,
            zeta_wip=0.0,
        ),
        min_gap_grid=(5, 10, 20, 40, 60, 80),
        wip_caps=(2, 3, 4, 5),
    )
    # Results & plots are in outputs_opt_2d/
