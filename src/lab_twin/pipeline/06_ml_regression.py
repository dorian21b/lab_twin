# lab_twin/pipeline/06_ml_regression.py
from pathlib import Path
import subprocess, sys

def main():
    # Train on dense set if available; else skip contours gracefully.
    roots=[]
    if Path("outputs_opt_2d").exists(): roots.append("outputs_opt_2d")
    if Path("outputs_opt").exists():    roots.append("outputs_opt")

    if not roots:
        print("No optimization outputs found. Run 04_policy_sweep or 05_opt_2d first.")
        sys.exit(1)

    # Call your existing module for each root
    for r in roots:
        print(f"\n=== ML on {r} ===")
        subprocess.run([sys.executable, "-m", "lab_twin.opt.regression", r], check=False)

if __name__=="__main__":
    main()
