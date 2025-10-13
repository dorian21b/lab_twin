from __future__ import annotations
from pathlib import Path
import glob
import math
import pandas as pd

def write_kpi_summary(run_dir: str | Path, run_id: str, sla_min: float = 120.0) -> None:
    """
    Build a comprehensive KPI dashboard by merging process-level events
    and sample-level outcomes.
    """
    run_dir = Path(run_dir)

    # ---------- 1) Load Events ----------
    events_path = run_dir / f"events_report.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")
    ev = pd.read_csv(events_path)

    # ensure numeric
    num_cols = ["sim_start","sim_end","service_min","wait_min","queue_len_on_arrival"]
    ev[num_cols] = ev[num_cols].apply(pd.to_numeric, errors="coerce")

    # ---------- 2) Load Samples ----------
    sample_files = sorted(glob.glob(str(run_dir / f"sample_report_{run_id}_BATCH_*.csv")))
    if not sample_files:
        # fallback if single file only
        sample_path = run_dir / "sample_report.csv"
        if not sample_path.exists():
            raise FileNotFoundError(f"No sample reports found in {run_dir}")
        samples = pd.read_csv(sample_path)
    else:
        samples = pd.concat((pd.read_csv(p) for p in sample_files), ignore_index=True)

    samples["tat_minutes"] = pd.to_numeric(samples["tat_minutes"], errors="coerce")

    # ---------- 3) Process-level KPIs ----------
    proc_kpis = (
        ev.groupby("process_id", as_index=False)
          .agg(
              n_events=("process_id","count"),
              avg_service_min=("service_min","mean"),
              p90_service_min=("service_min", lambda s: s.quantile(0.9)),
              avg_wait_min=("wait_min","mean"),
              p90_wait_min=("wait_min", lambda s: s.quantile(0.9)),
              avg_queue_len=("queue_len_on_arrival","mean"),
              max_queue_len=("queue_len_on_arrival","max"),
          )
          .sort_values("avg_wait_min", ascending=False)
    )

    # ---------- 4) Resource utilization ----------
    # approximate busy fraction per resource
    util_df = (
        ev.groupby("resource_name", as_index=False)
          .agg(
              total_service_min=("service_min","sum"),
              total_wait_min=("wait_min","sum"),
              n_ops=("process_id","count"),
              first_start=("sim_start","min"),
              last_end=("sim_end","max")
          )
    )
    util_df["makespan_min"] = util_df["last_end"] - util_df["first_start"]
    util_df["utilization_pct"] = 100 * (util_df["total_service_min"] / util_df["makespan_min"].replace(0, math.nan))

    # ---------- 5) Batch-level KPIs ----------
    # from events (actual sim flow)
    batch_from_events = (
        ev.groupby("batch_id", as_index=False)
          .agg(
              start_min=("sim_start","min"),
              end_min=("sim_end","max"),
              total_wait_min=("wait_min","sum"),
              total_service_min=("service_min","sum"),
              mean_queue_len=("queue_len_on_arrival","mean")
          )
    )
    batch_from_events["tat_min_events"] = batch_from_events["end_min"] - batch_from_events["start_min"]

    # join sample-level TAT (outcome)
    batch_from_samples = (
        samples.groupby("batch_id", as_index=False)
               .agg(
                   n_samples=("barcode","count"),
                   tat_mean_min=("tat_minutes","mean"),
                   tat_p90_min=("tat_minutes", lambda s: s.quantile(0.9)),
                   tat_min_min=("tat_minutes","min"),
                   tat_max_min=("tat_minutes","max"),
                   sla_pct=("tat_minutes", lambda s: (s <= sla_min).mean() * 100.0),
               )
    )
    batch_kpis = pd.merge(batch_from_events, batch_from_samples, on="batch_id", how="outer")

    # ---------- 6) Overall summary ----------
    makespan_min = float(ev["sim_end"].max() - ev["sim_start"].min())
    avg_tat = samples["tat_minutes"].mean()
    p90_tat = samples["tat_minutes"].quantile(0.9)
    sla_pct = (samples["tat_minutes"] <= sla_min).mean() * 100.0

    bottleneck_row = proc_kpis.iloc[0]
    overall = pd.DataFrame([{
        "run_id": run_id,
        "n_batches": batch_kpis["batch_id"].nunique(),
        "n_samples": len(samples),
        "makespan_min": round(makespan_min, 3),
        "avg_tat_min": round(avg_tat, 3),
        "p90_tat_min": round(p90_tat, 3),
        "sla_threshold_min": sla_min,
        "sla_pct": round(sla_pct, 2),
        "bottleneck_process": bottleneck_row["process_id"],
        "bottleneck_avg_wait_min": round(bottleneck_row["avg_wait_min"], 3),
        "bottleneck_avg_service_min": round(bottleneck_row["avg_service_min"], 3),
        "max_queue_len": int(bottleneck_row["max_queue_len"]),
    }])

    # ---------- 7) Write outputs ----------
    overall.to_csv(run_dir / f"kpi_overall.csv", index=False)
    batch_kpis.to_csv(run_dir / f"kpi_batches.csv", index=False)
    proc_kpis.to_csv(run_dir / f"kpi_process.csv", index=False)
    util_df.to_csv(run_dir / f"kpi_resources.csv", index=False)

    print(f"[KPIs] Wrote:")
    print(f"  - kpi_overall_{run_id}.csv")
    print(f"  - kpi_batches_{run_id}.csv")
    print(f"  - kpi_process_{run_id}.csv")
    print(f"  - kpi_resources_{run_id}.csv")
