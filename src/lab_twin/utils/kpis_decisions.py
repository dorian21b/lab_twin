from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

def _read_events(run_dir: Path):
    p = Path(run_dir) / "events_report.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    # Normalize cols
    for c in ["process_id", "status", "batch_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def _read_samples(run_dir: Path):
    p = Path(run_dir) / "sample_report.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def _is_decision_row(s: str) -> bool:
    return s.startswith("decision:")

def write_decision_kpis(run_dir: str | Path, run_id: str):
    run_dir = Path(run_dir)
    events = _read_events(run_dir)

    # ---- Decision splits
    dec = events[events["process_id"].map(_is_decision_row)].copy()
    if dec.empty:
        out = run_dir / f"kpi_decisions_{run_id}.csv"
        pd.DataFrame(columns=["decision","n","yes","no","other","yes_pct","no_pct"]).to_csv(out, index=False)
        return

    # extract outcome from status: "ROUTED:<branch>"
    dec["decision"] = dec["process_id"].str.replace("^decision:", "", regex=True)
    dec["outcome"]  = dec["status"].str.replace("^ROUTED:", "", regex=True)

    piv = dec.pivot_table(index="decision", columns="outcome", values="batch_id", aggfunc="count", fill_value=0)
    piv["n"] = piv.sum(axis=1)
    for col in ["yes","no"]:
        if col not in piv.columns: piv[col] = 0
    piv["yes_pct"] = (piv["yes"] / piv["n"]).round(4)
    piv["no_pct"]  = (piv["no"]  / piv["n"]).round(4)
    piv = piv.reset_index().sort_values("decision")
    piv.to_csv(run_dir / f"kpi_decisions_{run_id}.csv", index=False)

def write_branch_kpis(run_dir: str | Path, run_id: str, tails: dict[str, str] | None = None):
    """
    tails: optional mapping {process_id_of_tail: friendly_branch_name}
           If None, we'll infer "tails" as nodes that never enqueue anything downstream:
           i.e., processes that appear as process_id but not as predecessors. With your current
           line_full build, using the explicit mapping is most reliable.
    """
    run_dir = Path(run_dir)
    events = _read_events(run_dir)
    samples = _read_samples(run_dir)

    # Last event time per (batch) is the batch completion in the sim
    last_evt = (events
                .sort_values("sim_end")
                .groupby("batch_id", as_index=False)
                .tail(1)[["batch_id","process_id","sim_end"]]
               ).rename(columns={"process_id":"tail_process","sim_end":"complete_min"})

    # Merge sample TATs (same per-batch across its samples in your writer)
    tat = (samples[["batch_id","tat_minutes"]]
           .dropna()
           .drop_duplicates("batch_id"))

    out_df = last_evt.merge(tat, on="batch_id", how="left")

    # Friendly branch names
    if tails:
        out_df["branch"] = out_df["tail_process"].map(tails).fillna(out_df["tail_process"])
    else:
        out_df["branch"] = out_df["tail_process"]

    # KPIs per branch
    kpi = (out_df.groupby("branch")
                 .agg(n_batches=("batch_id","nunique"),
                      tat_mean_min=("tat_minutes","mean"),
                      tat_p90_min=("tat_minutes", lambda x: x.quantile(0.90)),
                      tat_min_min=("tat_minutes","min"),
                      tat_max_min=("tat_minutes","max"))
                 .reset_index())
    kpi["tat_mean_min"] = kpi["tat_mean_min"].round(3)
    kpi["tat_p90_min"]  = kpi["tat_p90_min"].round(3)
    kpi["tat_min_min"]  = kpi["tat_min_min"].round(3)
    kpi["tat_max_min"]  = kpi["tat_max_min"].round(3)

    kpi.to_csv(run_dir / f"kpi_branches_{run_id}.csv", index=False)

def write_yield_tree(run_dir: str | Path, run_id: str):
    """
    Compact yield summary for main funnel:
      DNA QC pass %, Library QC pass %, Sequencing success %, Reporting variant rate %
    """
    run_dir = Path(run_dir)
    events = _read_events(run_dir)
    dec = events[events["process_id"].map(_is_decision_row)].copy()
    if dec.empty:
        pd.DataFrame(columns=["stage","pass_pct","n"]).to_csv(run_dir / f"kpi_yield_tree_{run_id}.csv", index=False)
        return

    dec["decision"] = dec["process_id"].str.replace("^decision:", "", regex=True)
    dec["outcome"]  = dec["status"].str.replace("^ROUTED:", "", regex=True)

    def pass_rate(name: str, pass_key: str = "yes"):
        sub = dec[dec["decision"] == name]
        if sub.empty: return {"stage": name, "pass_pct": None, "n": 0}
        n = len(sub)
        p = (sub["outcome"] == pass_key).sum() / n
        return {"stage": name, "pass_pct": round(p,4), "n": n}

    rows = [
        pass_rate("dna_qc_pass", "yes"),
        pass_rate("library_qc_pass", "yes"),
        pass_rate("sequencing_success", "yes"),
        # reporting split is not pass/fail, show variant rate:
        pass_rate("is_variant", "yes"),
    ]
    pd.DataFrame(rows).to_csv(run_dir / f"kpi_yield_tree_{run_id}.csv", index=False)

def write_quality_kpis(run_dir: str | Path, run_id: str, sla_min: float = 120.0):
    """High-level: SLA hit rate, fail counts at each gate, rework exposure."""
    run_dir = Path(run_dir)
    events = _read_events(run_dir)
    samples = _read_samples(run_dir)

    overall = samples.copy()
    overall["hit_sla"] = overall["tat_minutes"] <= sla_min
    sla = overall["hit_sla"].mean() if len(overall) else 0.0

    dec = events[events["process_id"].map(_is_decision_row)].copy()
    dec["decision"] = dec["process_id"].str.replace("^decision:", "", regex=True)
    dec["outcome"]  = dec["status"].str.replace("^ROUTED:", "", regex=True)
    fail_counts = (dec[dec["outcome"].isin(["no","fail"])]
                   .groupby("decision")["batch_id"].nunique()
                   .reset_index(name="n_failed"))

    out = pd.DataFrame({
        "run_id":[samples["run_id"].iloc[0] if "run_id" in samples.columns and len(samples)>0 else run_id],
        "sla_threshold_min":[sla_min],
        "sla_pct":[round(100*sla,2)],
    })
    out.to_csv(run_dir / f"kpi_quality_{run_id}.csv", index=False)

    fail_counts.to_csv(run_dir / f"kpi_fail_counts_{run_id}.csv", index=False)

def write_all_branch_kpis(run_dir: str | Path, run_id: str, tails_map: dict[str,str] | None = None, sla_min: float = 120.0):
    write_decision_kpis(run_dir, run_id)
    write_branch_kpis(run_dir, run_id, tails_map)
    write_yield_tree(run_dir, run_id)
    write_quality_kpis(run_dir, run_id, sla_min)
