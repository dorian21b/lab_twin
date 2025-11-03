from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _plot_gantt(events: pd.DataFrame, outpath: Path):
    d = events[(~events["process_id"].astype(str).str.startswith("decision:")) &
               (pd.to_numeric(events["service_min"], errors="coerce") > 0)].copy()
    d = d.sort_values(["batch_id", "sim_start"])
    y_labels, y_pos = [], []
    for b, sub in d.groupby("batch_id", sort=False):
        for _idx, row in sub.iterrows():
            y_labels.append(f"{b} | {row['process_id']}")
            y_pos.append(len(y_pos))

    fig, ax = plt.subplots(figsize=(12, max(4, len(y_pos)*0.20)))
    for y, row in zip(y_pos, d.itertuples(index=False)):
        ax.barh(y, width=row.sim_end - row.sim_start, left=row.sim_start)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Batch | Process")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_title("End-to-End Timeline (Gantt)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def _plot_process_perf(proc_df: pd.DataFrame, outpath: Path, top_n: int = 10):
    d = proc_df.copy()
    d["avg_service_min"] = pd.to_numeric(d["avg_service_min"], errors="coerce")
    d["avg_wait_min"]    = pd.to_numeric(d["avg_wait_min"], errors="coerce")
    d = d.sort_values("avg_service_min", ascending=False).head(top_n)
    x = range(len(d)); w = 0.4

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - w/2 for i in x], d["avg_service_min"], width=w, label="avg_service_min")
    ax.bar([i + w/2 for i in x], d["avg_wait_min"],    width=w, label="avg_wait_min")
    ax.set_xticks(list(x))
    ax.set_xticklabels(d["process_id"], rotation=45, ha="right")
    ax.set_ylabel("Minutes")
    ax.set_title(f"Top {len(d)} Processes: Service vs Wait")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def _plot_batch_tat(batch_df: pd.DataFrame, outpath: Path):
    d = batch_df.copy()
    if "tat_min_events" in d.columns and d["tat_min_events"].notna().any():
        d["tat"] = pd.to_numeric(d["tat_min_events"], errors="coerce")
    else:
        d["tat"] = pd.to_numeric(d["end_min"], errors="coerce") - pd.to_numeric(d["start_min"], errors="coerce")
    d = d.sort_values("batch_id")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(d["batch_id"].astype(str), d["tat"])
    ax.set_ylabel("Turnaround time (min)")
    ax.set_title("Batch TATs")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def _plot_decisions(events: pd.DataFrame, outpath: Path):
    d = events[events["process_id"].astype(str).str.startswith("decision:")].copy()
    if d.empty:
        return
    d["route"] = d["status"].str.extract(r"ROUTED:(yes|no)", expand=False).fillna("unknown")
    d["decision"] = d["process_id"].str.replace("^decision:", "", regex=True)
    piv = d.pivot_table(index="decision", columns="route", values="batch_id", aggfunc="count", fill_value=0)
    for col in ["yes", "no"]:
        if col not in piv.columns:
            piv[col] = 0
    piv = piv[["yes", "no"]].sort_index()
    x = range(len(piv)); w = 0.4

    fig, ax = plt.subplots(figsize=(8, 4 + 0.2*len(piv)))
    ax.bar([i - w/2 for i in x], piv["yes"], width=w, label="yes")
    ax.bar([i + w/2 for i in x], piv["no"],  width=w, label="no")
    ax.set_xticks(list(x))
    ax.set_xticklabels(piv.index, rotation=30, ha="right")
    ax.set_ylabel("# routed batches")
    ax.set_title("Decision Outcomes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def _save_plots(run_dir: Path):
    # events
    events = pd.read_csv(run_dir / "events_report.csv")
    # proc kpis
    proc_candidates = sorted(run_dir.glob("kpi_process_*.csv")) or [run_dir / "kpi_process.csv"]
    proc_kpis = pd.read_csv(proc_candidates[-1])
    # batch kpis
    batch_candidates = sorted(run_dir.glob("kpi_batches_*.csv")) or [run_dir / "kpi_batches.csv"]
    batch_kpis = pd.read_csv(batch_candidates[-1])

    # ensure numeric
    for c in ["sim_start", "sim_end", "service_min", "wait_min"]:
        if c in events.columns:
            events[c] = pd.to_numeric(events[c], errors="coerce")

    _plot_gantt(events, run_dir / "plot_A_gantt.png")
    _plot_process_perf(proc_kpis, run_dir / "plot_B_process_perf.png")
    _plot_batch_tat(batch_kpis, run_dir / "plot_C_batch_tat.png")
    _plot_decisions(events, run_dir / "plot_D_decisions.png")
