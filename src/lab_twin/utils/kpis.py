# lab_twin/utils/kpis.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def _read_samples(run_dir: Path) -> pd.DataFrame | None:
    # Support both spellings
    for name in ("sample_report.csv", "samples_report.csv"):
        p = run_dir / name
        if p.exists():
            df = pd.read_csv(p)
            # expected: run_id,batch_id,barcode,source_lab,created_at,tat_minutes
            if "tat_minutes" in df.columns and "batch_id" in df.columns:
                return df
    return None


def _p90(x: pd.Series) -> float:
    return float(np.quantile(x, 0.9)) if len(x) else np.nan

def _sla_pct_from_series(tat: pd.Series, sla_min: float | None,
                         priority: pd.Series | None = None,
                         sla_map: dict[str, float] | None = None) -> float:
    """
    Compute % within SLA. If sla_map and priority are provided, use per-priority thresholds;
    else fall back to single sla_min.
    """
    if tat is None or len(tat) == 0:
        return np.nan
    tat = tat.astype(float)

    if sla_map and priority is not None:
        # vectorized per-row thresholds using map; default to sla_min if missing label
        pri = priority.astype(str).fillna("")
        # build thresholds series
        thr = pri.map(lambda k: sla_map.get(k, sla_min if sla_min is not None else np.inf))
        thr = thr.astype(float)
        return float(((tat <= thr).astype(float).mean()) * 100.0)
    # fallback: single SLA threshold
    if sla_min is None:
        return np.nan
    return float(((tat <= float(sla_min)).astype(float).mean()) * 100.0)


# def write_kpi_summary(run_dir: Path, run_id: str, sla_min: float = 120.0) -> None:
#     run_dir = Path(run_dir)
#     events_path = run_dir / "events_report.csv"
#     if not events_path.exists():
#         raise FileNotFoundError(f"Missing {events_path}")

#     events = pd.read_csv(events_path)
#     # Normalize columns we rely on
#     for col in ["batch_id","process_id","sim_start","sim_end","service_min","wait_min","queue_len_on_arrival","status"]:
#         if col not in events.columns:
#             raise ValueError(f"events_report.csv missing column: {col}")

#     # --- Split decisions vs real processes (decisions are 0-service routers) ---
#     is_decision = events["process_id"].astype(str).str.startswith("decision:")
#     proc_events = events.loc[~is_decision].copy()
#     decision_events = events.loc[is_decision].copy()

#     # ======================
#     # 1) Batch KPIs (events)
#     # ======================
#     batch_span = (
#         proc_events.groupby("batch_id")
#         .agg(start_min=("sim_start","min"),
#              end_min=("sim_end","max"),
#              total_service_min=("service_min","sum"),
#              total_wait_min=("wait_min","sum"),
#              mean_queue_len=("queue_len_on_arrival","mean"))
#         .reset_index()
#     )
#     batch_span["tat_min_events"] = batch_span["end_min"] - batch_span["start_min"]

#     # ==============================
#     # 2) Add per-sample TAT if avail
#     # ==============================
#     samples = _read_samples(run_dir)
#     if samples is not None:
#         tat_agg = (
#             samples.groupby("batch_id")["tat_minutes"]
#             .agg(tat_mean_min="mean",
#                  tat_p90_min=lambda s: s.quantile(0.9),
#                  tat_min_min="min",
#                  tat_max_min="max",
#                  n_samples="count",
#                  sla_pct=lambda s: (s <= sla_min).mean()*100.0)
#             .reset_index()
#         )
#         batch_kpis = batch_span.merge(tat_agg, on="batch_id", how="left")
#     else:
#         # Fallback: use event-derived TAT, no per-sample stats
#         batch_kpis = batch_span.copy()
#         batch_kpis["n_samples"] = np.nan
#         for c in ["tat_mean_min","tat_p90_min","tat_min_min","tat_max_min","sla_pct"]:
#             batch_kpis[c] = np.nan

#     # =================================
#     # 3) Overall KPIs (prefer samples)
#     # =================================
#     makespan_min = proc_events["sim_end"].max()
#     n_batches = batch_kpis["batch_id"].nunique()
#     if samples is not None:
#         n_samples = int(samples.shape[0])
#         overall_avg_tat = samples["tat_minutes"].mean()
#         overall_p90_tat = samples["tat_minutes"].quantile(0.9)
#         overall_sla = (samples["tat_minutes"] <= sla_min).mean()*100.0
#     else:
#         n_samples = int(batch_kpis["n_samples"].fillna(0).sum())
#         overall_avg_tat = batch_kpis["tat_min_events"].mean()
#         overall_p90_tat = batch_kpis["tat_min_events"].quantile(0.9)
#         overall_sla = np.nan

#     # Bottleneck: process with highest avg wait (exclude decisions)
#     proc_wait = (
#         proc_events.groupby("process_id")
#         .agg(avg_wait_min=("wait_min","mean"),
#              avg_service_min=("service_min","mean"))
#         .reset_index()
#         .sort_values("avg_wait_min", ascending=False)
#     )
#     if len(proc_wait):
#         bottleneck_row = proc_wait.iloc[0]
#         bottleneck_process = bottleneck_row["process_id"]
#         bottleneck_avg_wait = bottleneck_row["avg_wait_min"]
#         bottleneck_avg_service = bottleneck_row["avg_service_min"]
#     else:
#         bottleneck_process = ""
#         bottleneck_avg_wait = np.nan
#         bottleneck_avg_service = np.nan

#     max_q = proc_events["queue_len_on_arrival"].max() if len(proc_events) else np.nan

#     overall = pd.DataFrame([{
#         "run_id": run_id,
#         "n_batches": n_batches,
#         "n_samples": n_samples,
#         "makespan_min": makespan_min,
#         "avg_tat_min": overall_avg_tat,
#         "p90_tat_min": overall_p90_tat,
#         "sla_threshold_min": sla_min,
#         "sla_pct": overall_sla,
#         "bottleneck_process": bottleneck_process,
#         "bottleneck_avg_wait_min": bottleneck_avg_wait,
#         "bottleneck_avg_service_min": bottleneck_avg_service,
#         "max_queue_len": max_q,
#     }])

#     # ==============================
#     # 4) Process-level KPIs (no dec)
#     # ==============================
#     def p90(x): return np.quantile(x, 0.9)
#     proc_kpis = (
#         proc_events.groupby("process_id")
#         .agg(n_events=("process_id","count"),
#              avg_service_min=("service_min","mean"),
#              p90_service_min=("service_min", p90),
#              avg_wait_min=("wait_min","mean"),
#              p90_wait_min=("wait_min", p90),
#              avg_queue_len=("queue_len_on_arrival","mean"),
#              max_queue_len=("queue_len_on_arrival","max"))
#         .reset_index()
#         .sort_values(["avg_wait_min","avg_service_min"], ascending=False)
#     )

#     # ======================
#     # 5) Decision audit KPI
#     # ======================
#     routing = pd.DataFrame()
#     if len(decision_events):
#         # status contains "ROUTED:yes" or "ROUTED:no"
#         decision_events = decision_events.copy()
#         decision_events["decision"] = decision_events["process_id"].str.replace("decision:", "", regex=False)
#         decision_events["route"] = decision_events["status"].str.split(":", expand=True).iloc[:, -1].str.lower()
#         routing = (
#             decision_events.groupby(["decision","route"])
#             .size()
#             .reset_index(name="count")
#             .pivot(index="decision", columns="route", values="count")
#             .fillna(0)
#             .astype(int)
#             .reset_index()
#         )

#     # =================
#     # 6) Save CSV files
#     # =================
#     overall.to_csv(run_dir / f"kpi_overall.csv", index=False)
#     batch_kpis.to_csv(run_dir / f"kpi_batches.csv", index=False)
#     proc_kpis.to_csv(run_dir / f"kpi_process.csv", index=False)
#     if len(routing):
#         routing.to_csv(run_dir / f"kpi_routing.csv", index=False)

# ---------- main ----------
def write_kpi_summary(
    run_dir: Path,
    run_id: str,
    sla_min: float | None = 120.0,
    sla_map: dict[str, float] | None = None,   # e.g. {"urgent": 14400, "routine": 43200}
) -> None:
    """
    Writes:
      - kpi_overall.csv
      - kpi_batches.csv
      - kpi_process.csv
      - kpi_routing.csv (if decisions exist)

    SLA handling:
      * If sla_map is provided and 'priority' exists in sample_report, uses per-priority thresholds.
      * Else uses single sla_min for everyone.
    """
    run_dir = Path(run_dir)

    events_path = run_dir / "events_report.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing {events_path}")

    events = pd.read_csv(events_path)
    # Validate required columns
    req = ["batch_id","process_id","sim_start","sim_end",
           "service_min","wait_min","queue_len_on_arrival","status"]
    missing = [c for c in req if c not in events.columns]
    if missing:
        raise ValueError(f"events_report.csv missing columns: {missing}")

    # Separate decisions
    is_decision = events["process_id"].astype(str).str.startswith("decision:")
    proc_events = events.loc[~is_decision].copy()
    decision_events = events.loc[is_decision].copy()

    # 1) Batch KPIs from events
    batch_span = (
        proc_events.groupby("batch_id")
        .agg(start_min=("sim_start","min"),
             end_min=("sim_end","max"),
             total_service_min=("service_min","sum"),
             total_wait_min=("wait_min","sum"),
             mean_queue_len=("queue_len_on_arrival","mean"))
        .reset_index()
    )
    batch_span["tat_min_events"] = batch_span["end_min"] - batch_span["start_min"]

    # 2) Samples (preferred if present)
    samples = _read_samples(run_dir)

    if samples is not None:
        # per-batch aggregation using sample TATs
        has_priority = "priority" in samples.columns
        tat_agg = (
            samples.groupby("batch_id")
            .apply(lambda g: pd.Series({
                "tat_mean_min": float(g["tat_minutes"].mean()),
                "tat_p90_min":  _p90(g["tat_minutes"]),
                "tat_min_min":  float(g["tat_minutes"].min()),
                "tat_max_min":  float(g["tat_minutes"].max()),
                "n_samples":    int(g.shape[0]),
                "sla_pct":      _sla_pct_from_series(
                                    g["tat_minutes"],
                                    sla_min=sla_min,
                                    priority=g["priority"] if has_priority else None,
                                    sla_map=sla_map
                                ),
            }))
            .reset_index()
        )
        batch_kpis = batch_span.merge(tat_agg, on="batch_id", how="left")
    else:
        batch_kpis = batch_span.copy()
        batch_kpis["n_samples"] = np.nan
        for c in ["tat_mean_min","tat_p90_min","tat_min_min","tat_max_min","sla_pct"]:
            batch_kpis[c] = np.nan

    # 3) Overall KPIs
    makespan_min = float(proc_events["sim_end"].max()) if len(proc_events) else np.nan
    n_batches = int(batch_kpis["batch_id"].nunique())

    if samples is not None:
        n_samples = int(samples.shape[0])
        overall_avg_tat = float(samples["tat_minutes"].mean())
        overall_p90_tat = _p90(samples["tat_minutes"])
        overall_sla = _sla_pct_from_series(
            samples["tat_minutes"],
            sla_min=sla_min,
            priority=samples["priority"] if "priority" in samples.columns else None,
            sla_map=sla_map
        )
    else:
        n_samples = int(batch_kpis["n_samples"].fillna(0).sum())
        overall_avg_tat = float(batch_kpis["tat_min_events"].mean())
        overall_p90_tat = _p90(batch_kpis["tat_min_events"])
        overall_sla = np.nan

    # bottleneck by avg wait
    proc_wait = (
        proc_events.groupby("process_id")
        .agg(avg_wait_min=("wait_min","mean"),
             avg_service_min=("service_min","mean"))
        .reset_index()
        .sort_values("avg_wait_min", ascending=False)
    )
    if len(proc_wait):
        bn = proc_wait.iloc[0]
        bottleneck_process = bn["process_id"]
        bottleneck_avg_wait = float(bn["avg_wait_min"])
        bottleneck_avg_service = float(bn["avg_service_min"])
    else:
        bottleneck_process = ""
        bottleneck_avg_wait = np.nan
        bottleneck_avg_service = np.nan

    max_q = float(proc_events["queue_len_on_arrival"].max()) if len(proc_events) else np.nan

    overall = pd.DataFrame([{
        "run_id": run_id,
        "n_batches": n_batches,
        "n_samples": n_samples,
        "makespan_min": makespan_min,
        "avg_tat_min": overall_avg_tat,
        "p90_tat_min": overall_p90_tat,
        # for transparency, store both single and map
        "sla_threshold_min": float(sla_min) if sla_min is not None else np.nan,
        "sla_map_used": str(sla_map) if sla_map else "",
        "sla_pct": overall_sla,
        "bottleneck_process": bottleneck_process,
        "bottleneck_avg_wait_min": bottleneck_avg_wait,
        "bottleneck_avg_service_min": bottleneck_avg_service,
        "max_queue_len": max_q,
    }])

    # 4) Process-level KPIs
    proc_kpis = (
        proc_events.groupby("process_id")
        .agg(n_events=("process_id","count"),
             avg_service_min=("service_min","mean"),
             p90_service_min=("service_min", _p90),
             avg_wait_min=("wait_min","mean"),
             p90_wait_min=("wait_min", _p90),
             avg_queue_len=("queue_len_on_arrival","mean"),
             max_queue_len=("queue_len_on_arrival","max"))
        .reset_index()
        .sort_values(["avg_wait_min","avg_service_min"], ascending=False)
    )

    # 5) Decision audit
    routing = pd.DataFrame()
    if len(decision_events):
        dec = decision_events.copy()
        dec["decision"] = dec["process_id"].str.replace("decision:", "", regex=False)
        dec["route"] = dec["status"].str.split(":", n=1).str[-1].str.lower()
        routing = (
            dec.groupby(["decision","route"])
               .size()
               .reset_index(name="count")
               .pivot(index="decision", columns="route", values="count")
               .fillna(0).astype(int).reset_index()
        )

    # 6) Save
    overall.to_csv(run_dir / "kpi_overall.csv", index=False)
    batch_kpis.to_csv(run_dir / "kpi_batches.csv", index=False)
    proc_kpis.to_csv(run_dir / "kpi_process.csv", index=False)
    if len(routing):
        routing.to_csv(run_dir / "kpi_routing.csv", index=False)
