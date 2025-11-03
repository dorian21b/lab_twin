# lab_twin/pipeline/04_policy_sweep.py
from pathlib import Path
import yaml
from lab_twin.opt.runner import grid_search, OptConfig

def main():
    cfg=yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    out="outputs_opt"  # consistent with your runner
    grid_search(
        out_root=out,
        config=OptConfig(
            n_batches=cfg["n_batches"],
            samples_per_batch=cfg["samples_per_batch"],
            seeds=tuple(cfg["seeds"]),
            sla_min=cfg["sla_min"],
            alpha_avg=1.0, beta_p90=0.5, eta_sla=2.0, zeta_wip=0.0
        ),
        min_gap_grid=tuple(cfg["min_gap_grid"])
    )
    print(f" Sweep done → {Path(out).resolve()}")

if __name__=="__main__":
    main()
