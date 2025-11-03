from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Helpers
# ---------------------------
def _load_proc_kpis(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "kpi_process.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p)
    # expected: process_id, n_events, avg_service_min, p90_service_min,
    #           avg_wait_min, p90_wait_min, avg_queue_len, max_queue_len
    return df

def _load_events(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "events_report.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p)
    # Normalize names used by your engine
    if "sim_start_min" in df.columns:
        df = df.rename(columns={"sim_start_min":"sim_start", "sim_end_min":"sim_end"})
    return df

def _load_samples(run_dir: Path) -> pd.DataFrame | None:
    for name in ("sample_report.csv", "samples_report.csv"):
        p = run_dir / name
        if p.exists():
            df = pd.read_csv(p)
            if {"batch_id","tat_minutes"}.issubset(df.columns):
                return df
    return None

def _load_batches_tat_from_events(run_dir: Path) -> pd.DataFrame:
    """Fallback when per-sample TAT isn't available: derive batch TAT from events."""
    ev = _load_events(run_dir)
    is_decision = ev["process_id"].astype(str).str.startswith("decision:")
    ev = ev.loc[~is_decision].copy()
    g = ev.groupby("batch_id").agg(start=("sim_start","min"), end=("sim_end","max")).reset_index()
    g["tat_minutes"] = g["end"] - g["start"]
    # create a sample-like column so the plotting code just works
    g["barcode"] = g["batch_id"]
    return g[["batch_id","barcode","tat_minutes"]]

def _safe_title_from_run(run_dir: Path) -> str:
    # Use folder name as run label
    return run_dir.name

# ---------------------------
# Core compare function
# ---------------------------
def compare_runs(baseline_dir: str | Path, candidate_dir: str | Path, out_dir: str | Path = "outputs_diag") -> None:
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    label_base = _safe_title_from_run(baseline_dir)
    label_cand = _safe_title_from_run(candidate_dir)

    # ---- Load process KPIs
    kpi_b = _load_proc_kpis(baseline_dir)
    kpi_c = _load_proc_kpis(candidate_dir)

    # Align by process_id
    merged = kpi_b.merge(kpi_c, on="process_id", suffixes=("_base","_cand"), how="outer").fillna(0.0)

    # Sort by baseline avg_wait desc for nice plotting order
    merged = merged.sort_values("avg_wait_min_base", ascending=False)

    # ---- BAR: Per-process avg wait (side-by-side)
    idx = np.arange(len(merged))
    width = 0.42
    plt.figure(figsize=(12, 5))
    plt.bar(idx - width/2, merged["avg_wait_min_base"], width, label=f"{label_base}")
    plt.bar(idx + width/2, merged["avg_wait_min_cand"], width, label=f"{label_cand}")
    plt.xticks(idx, merged["process_id"], rotation=75, ha="right")
    plt.ylabel("Avg wait (min)")
    plt.title("Per-process average wait time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "01_process_avg_wait_bars.png", dpi=160)

    # ---- Identify top-2 bottlenecks from baseline
    top2 = merged.head(2)["process_id"].tolist()

    # ---- QUEUE LEN distributions at those bottlenecks
    ev_b = _load_events(baseline_dir)
    ev_c = _load_events(candidate_dir)

    # Only real processes
    mask_b = ~ev_b["process_id"].astype(str).str.startswith("decision:")
    mask_c = ~ev_c["process_id"].astype(str).str.startswith("decision:")
    ev_b = ev_b.loc[mask_b]
    ev_c = ev_c.loc[mask_c]

    for pidx, proc in enumerate(top2, start=1):
        qb = ev_b.loc[ev_b["process_id"] == proc, "queue_len_on_arrival"].astype(float)
        qc = ev_c.loc[ev_c["process_id"] == proc, "queue_len_on_arrival"].astype(float)

        plt.figure(figsize=(7, 4))
        plt.hist(qb, bins=range(0, int(max(qb.max() if len(qb) else 0, qc.max() if len(qc) else 0)+2)), alpha=0.6, label=label_base, density=True)
        plt.hist(qc, bins=range(0, int(max(qb.max() if len(qb) else 0, qc.max() if len(qc) else 0)+2)), alpha=0.6, label=label_cand, density=True)
        plt.xlabel("Queue length on arrival")
        plt.ylabel("Density")
        plt.title(f"Queue length distribution — {proc}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"02_queue_dist_{pidx}_{proc}.png", dpi=160)

    # ---- TAT distribution (per-sample if available; else per-batch from events)
    samp_b = _load_samples(baseline_dir)
    samp_c = _load_samples(candidate_dir)
    if samp_b is None:
        samp_b = _load_batches_tat_from_events(baseline_dir)
    if samp_c is None:
        samp_c = _load_batches_tat_from_events(candidate_dir)

    # Add run label
    samp_b = samp_b.copy(); samp_b["run"] = label_base
    samp_c = samp_c.copy(); samp_c["run"] = label_cand
    tat_df = pd.concat([samp_b[["tat_minutes","run"]], samp_c[["tat_minutes","run"]]], ignore_index=True)

    # Boxplot
    plt.figure(figsize=(6, 5))
    data = [tat_df.loc[tat_df["run"] == label_base, "tat_minutes"],
            tat_df.loc[tat_df["run"] == label_cand, "tat_minutes"]]
    plt.boxplot(data, labels=[label_base, label_cand], showfliers=False)
    plt.ylabel("TAT (min)")
    plt.title("Turnaround time distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "03_tat_boxplot.png", dpi=160)

    # Optional violin (can be nice for talks)
    try:
        plt.figure(figsize=(6, 5))
        parts = plt.violinplot(data, showmeans=True, showextrema=False)
        plt.xticks([1,2], [label_base, label_cand])
        plt.ylabel("TAT (min)")
        plt.title("Turnaround time distribution (violin)")
        plt.tight_layout()
        plt.savefig(out_dir / "03b_tat_violin.png", dpi=160)
    except Exception:
        pass

    # ---- Summary CSV with deltas
    def safe_mean(x): return float(np.mean(x)) if len(x) else np.nan
    def safe_p90(x):  return float(np.quantile(x, 0.9)) if len(x) else np.nan

    summary = {
        "baseline_run": label_base,
        "candidate_run": label_cand,
        "tat_mean_base": safe_mean(data[0]),
        "tat_mean_cand": safe_mean(data[1]),
        "tat_p90_base": safe_p90(data[0]),
        "tat_p90_cand": safe_p90(data[1]),
        "tat_mean_delta": safe_mean(data[1]) - safe_mean(data[0]),
        "tat_p90_delta": safe_p90(data[1]) - safe_p90(data[0]),
    }

    # Add a few bottleneck rows (avg_wait deltas for top-2)
    for i, proc in enumerate(top2, start=1):
        base_w = float(merged.loc[merged["process_id"] == proc, "avg_wait_min_base"].values[0])
        cand_w = float(merged.loc[merged["process_id"] == proc, "avg_wait_min_cand"].values[0])
        summary[f"bottleneck{i}_process"] = proc
        summary[f"bottleneck{i}_avg_wait_base"] = base_w
        summary[f"bottleneck{i}_avg_wait_cand"] = cand_w
        summary[f"bottleneck{i}_avg_wait_delta"] = cand_w - base_w

    pd.DataFrame([summary]).to_csv(out_dir / "summary_before_after.csv", index=False)

    print(f"\nDiagnostics written to: {out_dir.resolve()}")
    print(" - 01_process_avg_wait_bars.png")
    for pidx, proc in enumerate(top2, start=1):
        print(f" - 02_queue_dist_{pidx}_{proc}.png")
    print(" - 03_tat_boxplot.png (and 03b_tat_violin.png if available)")
    print(" - summary_before_after.csv")
    print("\nTip: put these four figures on one slide for a crisp before/after story.")


# ---------------------------
# CLI example
# ---------------------------
if __name__ == "__main__":
    # Example:
    # python -m lab_twin.opt.diagnostics \
    #   outputs_opt/opt_YYYYMMDD_HHMMSS_gap10_s42 \
    #   outputs_opt/opt_YYYYMMDD_HHMMSS_gap60_s42 \
    #   outputs_diag/baseline_vs_gap60
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m lab_twin.opt.diagnostics <BASELINE_RUN_DIR> <CANDIDATE_RUN_DIR> [OUT_DIR]")
        sys.exit(1)
    baseline = sys.argv[1]
    candidate = sys.argv[2]
    outdir = sys.argv[3] if len(sys.argv) > 3 else "outputs_diag"
    compare_runs(baseline, candidate, outdir)
