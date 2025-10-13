from pathlib import Path
import csv
from dataclasses import dataclass, asdict

@dataclass
class EventRow:
    run_id: str
    batch_id: str
    process_id: str
    sim_start: float
    sim_end: float
    service_min: float
    wait_min: float
    queue_len_on_arrival: int
    status: str            # COMPLETED / FAILED / SKIPPED
    resource_name: str = ""
    note: str = ""

class EventLogger:
    def __init__(self, output_path: Path):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._f = None

    def _ensure_writer(self):
        if self._writer is None:
            self._f = self.path.open("w", newline="")
            headers = [f.name for f in EventRow.__dataclass_fields__.values()]
            self._writer = csv.DictWriter(self._f, fieldnames=headers)
            self._writer.writeheader()

    def write(self, row: EventRow):
        self._ensure_writer()
        self._writer.writerow(asdict(row))

    def close(self):
        if self._f:
            self._f.close()
