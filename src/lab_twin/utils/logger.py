# import csv
# from datetime import datetime
# from pathlib import Path
# from lab_twin.domain.batch import Batch

# def write_sample_report(batch: Batch, end_time: datetime, run_dir: str | Path = "outputs", run_id: str | None = None) -> None:
#     run_dir = Path(run_dir)
#     run_dir.mkdir(parents=True, exist_ok=True)

#     if run_id is None:
#         run_id = run_dir.name or "run"

#     output_path = run_dir / "sample_report.csv"
#     fieldnames = ["run_id", "batch_id", "barcode", "source_lab", "created_at", "tat_minutes", "priority"]

#     with output_path.open("w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()

#         tat_sim = None
#         if batch.arrival_sim_min is not None and batch.complete_sim_min is not None:
#             tat_sim = round(batch.complete_sim_min - batch.arrival_sim_min, 3)

#         for s in batch.samples:
#             writer.writerow({
#                 "run_id": run_id,
#                 "batch_id": batch.batch_id,
#                 "barcode": s.barcode or "UNASSIGNED",
#                 "source_lab": s.source_lab,
#                 "created_at": s.created_at.isoformat(),
#                 "tat_minutes": tat_sim,
#                 "priority": s.priority,
#             })

#     print(f"CSV report written to {output_path}")

# def write_samples_report(batches: list[Batch], end_time: datetime, run_dir: str|Path="outputs", run_id: str|None=None) -> None:
#     run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
#     if run_id is None: run_id = run_dir.name or "run"
#     output_path = run_dir / "sample_report.csv"
#     fieldnames = ["run_id","batch_id","barcode","source_lab","created_at","tat_minutes"]

#     with output_path.open("w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames); writer.writeheader()
#         for batch in batches:
#             tat_sim = None
#             if batch.arrival_sim_min is not None and batch.complete_sim_min is not None:
#                 tat_sim = round(batch.complete_sim_min - batch.arrival_sim_min, 3)
#             for s in batch.samples:
#                 writer.writerow({
#                     "run_id": run_id,
#                     "batch_id": batch.batch_id,
#                     "barcode": s.barcode or "UNASSIGNED",
#                     "source_lab": s.source_lab,
#                     "created_at": s.created_at.isoformat(),
#                     "tat_minutes": tat_sim,
#                     "priority": getattr(s, "priority", "routine"),
#                 })
#     print(f"CSV report written to {output_path}")

import csv
from datetime import datetime
from pathlib import Path
from lab_twin.domain.batch import Batch

HEADER = ["run_id","batch_id","barcode","source_lab","created_at","tat_minutes","priority"]

def _iso_or_blank(dt):
    try:
        return dt.isoformat()
    except Exception:
        return ""

def write_sample_report(batch: Batch, end_time: datetime, run_dir: str | Path = "outputs", run_id: str | None = None) -> None:
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None: run_id = run_dir.name or "run"

    out = run_dir / "sample_report.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER); w.writeheader()

        # batch-level TAT as fallback for each sample
        tat_sim = None
        if batch.arrival_sim_min is not None and batch.complete_sim_min is not None:
            tat_sim = round(batch.complete_sim_min - batch.arrival_sim_min, 3)

        for s in batch.samples:
            w.writerow({
                "run_id": run_id,
                "batch_id": batch.batch_id,
                "barcode": getattr(s, "barcode", "") or "UNASSIGNED",
                "source_lab": getattr(s, "source_lab", "") or "",
                "created_at": _iso_or_blank(getattr(s, "created_at", None)),
                # prefer per-sample TAT if present; else batch fallback
                "tat_minutes": getattr(s, "tat_minutes", tat_sim),
                "priority": getattr(s, "priority", "routine"),
            })
    print(f"CSV report written to {out}")

def write_samples_report(batches: list[Batch], end_time: datetime, run_dir: str|Path="outputs", run_id: str|None=None) -> None:
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None: run_id = run_dir.name or "run"

    out = run_dir / "sample_report.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER); w.writeheader()

        for batch in batches:
            tat_sim = None
            if batch.arrival_sim_min is not None and batch.complete_sim_min is not None:
                tat_sim = round(batch.complete_sim_min - batch.arrival_sim_min, 3)

            for s in batch.samples:
                w.writerow({
                    "run_id": run_id,
                    "batch_id": batch.batch_id,
                    "barcode": getattr(s, "barcode", "") or "UNASSIGNED",
                    "source_lab": getattr(s, "source_lab", "") or "",
                    "created_at": _iso_or_blank(getattr(s, "created_at", None)),
                    "tat_minutes": getattr(s, "tat_minutes", tat_sim),
                    "priority": getattr(s, "priority", "routine"),
                })
    print(f"CSV report written to {out}")
