# lab_twin/pipeline/run_all.py
"""
Runs the full PoC sequence end-to-end:
  1. Simulate + monitor baseline
  2. Simulate + monitor stress
  3. Optimize stress scenario with 2D sweep

Usage:
  python -m lab_twin.pipeline.run_all
"""

import subprocess
from pathlib import Path

def run(cmd: list[str]):
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def get_latest(prefix: str) -> Path:
    root = Path("outputs_pipeline")
    candidates = sorted(root.glob(f"{prefix}_*"))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}_* folder found in {root}")
    return candidates[-1]

def main():
    # 1️⃣ Baseline
    run(["python", "-m", "lab_twin.pipeline.01_simulate", "baseline"])
    baseline_dir = str(get_latest("baseline"))
    run(["python", "-m", "lab_twin.pipeline.02_monitor", baseline_dir])

    # 2️⃣ Stress
    run(["python", "-m", "lab_twin.pipeline.01_simulate", "stress"])
    stress_dir = str(get_latest("stress"))
    run(["python", "-m", "lab_twin.pipeline.02_monitor", stress_dir])

    # 3️⃣ Optimize stress
    run(["python", "-m", "lab_twin.pipeline.05_opt_2d", stress_dir])

    print("\n✅ All scenarios completed.")
    print(f"- Baseline results:  {baseline_dir}")
    print(f"- Stress results:    {stress_dir}")
    print(f"- Optimization heatmaps: {stress_dir}/opt_2d/")

if __name__ == "__main__":
    main()
