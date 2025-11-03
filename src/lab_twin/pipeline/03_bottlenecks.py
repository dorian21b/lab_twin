# lab_twin/pipeline/03_bottlenecks.py
from pathlib import Path
import sys, pandas as pd, matplotlib.pyplot as plt

def main(run_dir: str):
    rd=Path(run_dir)
    ev=pd.read_csv(rd/"events_report.csv")
    kpp=pd.read_csv(rd/"kpi_process.csv").sort_values("avg_wait_min", ascending=False)

    print("\nTop bottlenecks (by avg_wait_min):")
    print(kpp.head(5)[["process_id","n_events","avg_wait_min","p90_wait_min","avg_queue_len","max_queue_len"]])

    # queue length hist for top-2
    top2=kpp.head(2)["process_id"].tolist()
    for i,proc in enumerate(top2,1):
        q=ev.loc[ev["process_id"]==proc, "queue_len_on_arrival"].astype(float)
        plt.figure(figsize=(5,4)); plt.hist(q, bins=range(0, int(q.max()+2)), density=True)
        plt.xlabel("Queue length on arrival"); plt.ylabel("Density")
        plt.title(f"Queue distribution — {proc}")
        plt.tight_layout(); plt.savefig(rd/f"plot_queue_{i}_{proc}.png", dpi=160)

if __name__=="__main__":
    run_dir = sys.argv[1] if len(sys.argv)>1 else sorted(Path("outputs_pipeline").glob("baseline_*"))[-1]
    main(str(run_dir))
