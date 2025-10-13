from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, auto
from typing import Callable, Iterable, Optional
from ..domain.batch import Batch


class Stage(Enum):
    GSTT_PREPCR = auto()
    HUB_PREPCR = auto()
    HUB_POSTPCR = auto()
    REPORTING = auto()


class ProcessStatus(Enum):
    PLANNED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()

@dataclass
class Process:
    process_id: str
    name: str
    stage: Stage
    actor_role: str
    resources: tuple[str, ...]
    duration_min_min: int
    duration_max_min: int
    batch_min: int = 56
    batch_max: int = 96
    is_actionable: bool = True
    is_terminal: bool = False  
    run: Optional[Callable[[Batch], None]] = None
    description: str = ""
    #tags: tuple[str, ...] = field(default_factory=tuple)

    def validate(self, batch) -> None:
        if not (self.batch_min <= batch.occupancy <= self.batch_max):
            raise ValueError("Batch size out of range")
        
    @property
    def min_duration(self) -> timedelta:
        return timedelta(minutes=self.duration_min_min)

    @property
    def max_duration(self) -> timedelta:
        return timedelta(minutes=self.duration_max_min)
    

