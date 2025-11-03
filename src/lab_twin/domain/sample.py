from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class Sample:
    barcode: str | None = None
    priority: str = "routine"
    source_lab: str = "GSTT"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.barcode is not None:
            self.barcode = self.barcode.strip()
            if not self.barcode:
                raise ValueError("barcode must be a non-empty string")

    def set_barcode(self, code: str) -> None:
        if self.barcode is not None:
            raise ValueError("barcode already set")
        code = code.strip()
        if not code:
            raise ValueError("barcode must be a non-empty string")
        self.barcode = code
