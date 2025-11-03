# scripts/validate_robustness.py
from pathlib import Path
import itertools, pandas as pd
from lab_twin.opt.time_sensitivity import TimeOptConfig, run_one_policy_time_only

out = Path("outputs_validate"); out.mkdir(exist_ok=True, parents=True)

cfg = TimeOptConfig(
    n_batches=6,
    samples_per_batch=96,
    seeds=tuple(range(1, 101)),   # 100 seeds
    sla_min=5*12*60
)

scenarios = [
    ("baseline", {}, {'Quantstudio 5':1,'Novaseq X +':1,'Hamilton Star':1,'Firefly +':1}),
    ("best_found", {'SEQUENCING':0.9}, {'Quantstudio 5':1,'Novaseq X +':1,'Hamilton Star':1,'Firefly +':2}),
    ("qs2", {}, {'Quantstudio 5':2,'Novaseq X +':1,'Hamilton Star':1,'Firefly +':1}),
]

rows = []
for label, dur_scale, caps in scenarios:
    for seed in cfg.seeds:
        # If you have a hook to pass cap overrides, set it in your sim before run
        # e.g., run_one_policy_time_only(..., duration_scale=dur_scale) and ensure resources override is applied globally.
        r = run_one_policy_time_only(out, seed, cfg, min_gap=60.0, duration_scale=dur_scale)
        r["scenario"]=label; r["cap_overrides"]=caps; rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(out/"robustness_raw.csv", index=False)

agg = df.groupby("scenario").agg(
    J_mean=("J","mean"), J_sem=("J","sem"),
    SLA_mean=("sla_pct","mean"), SLA_sem=("sla_pct","sem"),
    avg_tat_mean=("avg_tat_min","mean"),
    p90_tat_mean=("p90_tat_min","mean")
).reset_index()
agg["J_ci"] = 1.96 * agg["J_sem"].fillna(0.0)
agg["SLA_ci"] = 1.96 * agg["SLA_sem"].fillna(0.0)
agg.to_csv(out/"robustness_summary.csv", index=False)
print(agg.sort_values("J_mean"))
