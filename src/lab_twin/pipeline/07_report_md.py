# lab_twin/pipeline/07_report_md.py
from pathlib import Path
import yaml, datetime

def main():
    cfg=yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    out = Path(cfg["outputs_root"])
    latest_baseline = sorted(out.glob("baseline_*"))[-1]
    md = out / "EXEC_SUMMARY.md"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Lab Workflow: Simulation → Monitoring → Optimization  ({ts})",
        "## Baseline monitoring",
        f"- Run: `{latest_baseline.name}`",
        "- Artifacts: `events_report.csv`, `kpi_overall.csv`, `kpi_process.csv`",
        "- Plots: `plot_baseline_timeline.png`, `plot_avg_wait_bar.png`",
        "",
        "## Bottlenecks",
        "- See `plot_queue_1_*.png`, `plot_queue_2_*.png` in the baseline run directory.",
        "",
        "## Scheduling sweeps",
        "- 1D min_gap: `outputs_opt/summary_grid_agg.csv` + CI plots.",
        "- 2D min_gap × WIP: `outputs_opt_2d/heatmap_J.png`, `heatmap_SLA.png`.",
        "",
        "## ML surrogate",
        "- RandomForest fit over sweep results to interpolate policy space.",
        "- Artifacts: `outputs_ml/ml*/feature_importances.csv`, `contour_J.png`, `contour_SLA.png`, `ai_top_policies.csv`",
        "",
        "## Recommendation (current data)",
        "- Operate near **WIP cap = 2** and **min_gap ≈ 70–80 min** (best J; SLA mid-20s to ~30%).",
        "",
        "## Next steps",
        "1) Validate with a real-runs replay (if logs available).",
        "2) Add per-sample release control & partial loads (backlog).",
        "3) Sensitivity to instrument capacity changes (already supported).",
    ]
    md.write_text("\n".join(lines))
    print(f"✅ Report: {md.resolve()}")

if __name__=="__main__":
    main()
