from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Iterable, Set, Callable, Optional
from lab_twin.domain.sample import Sample

class BatchError(Exception): ...
class DuplicateSampleError(BatchError): ...
class CapacityError(BatchError): ...

@dataclass(eq=True, frozen=False)
class Batch:
    batch_id: str
    samples: List[Sample] = field(default_factory=list)
    min_capacity: int = 56
    max_capacity: int = 96

    arrival_sim_min: Optional[float] = None
    complete_sim_min: Optional[float] = None

    # ---- Derived properties ----
    @property
    def occupancy(self) -> int:
        """Current number of samples in the batch."""
        return len(self.samples)

    @property
    def barcodes(self) -> Set[str]:
        """
        Set of ASSIGNED barcodes (excludes None/empty).
        With Option A, some samples may be unbarcoded until stock retrieval.
        """
        return {s.barcode for s in self.samples if s.barcode}

    # ---- Mutators with guards ----
    def add_sample(self, sample: Sample) -> None:
        """
        Add a single sample.
        - Allows unassigned barcodes (None) pre stock-retrieval.
        - If a barcode is present, enforce uniqueness.
        - Enforce max capacity.
        """
        if sample.barcode and sample.barcode in self.barcodes:
            raise DuplicateSampleError(
                f"Sample with barcode '{sample.barcode}' already in batch {self.batch_id}."
            )
        if self.occupancy + 1 > self.max_capacity:
            raise CapacityError(
                f"Adding sample exceeds max capacity {self.max_capacity} (current {self.occupancy})."
            )
        self.samples.append(sample)

    def add_samples(self, new_samples: Iterable[Sample]) -> None:
        """
        Add many samples at once.
        - Allow unassigned barcodes.
        - Check duplicates ONLY among assigned barcodes.
        - Check capacity.
        """
        new_list = list(new_samples)

        # Duplicates within incoming set (assigned only)
        incoming_assigned = [s.barcode for s in new_list if s.barcode]
        if len(incoming_assigned) != len(set(incoming_assigned)):
            raise DuplicateSampleError("Incoming samples contain duplicate assigned barcodes.")

        # Duplicates vs existing (assigned only)
        incoming_set = set(incoming_assigned)
        dupes = self.barcodes.intersection(incoming_set)
        if dupes:
            raise DuplicateSampleError(f"Duplicate barcodes vs existing batch: {sorted(dupes)}")

        # Capacity
        if self.occupancy + len(new_list) > self.max_capacity:
            raise CapacityError(
                f"Adding {len(new_list)} exceeds max capacity {self.max_capacity} (current {self.occupancy})."
            )

        self.samples.extend(new_list)

    def mark_arrival(self, sim_min: float) -> None:
        if self.arrival_sim_min is None:
            self.arrival_sim_min = sim_min

    def mark_complete(self, sim_min: float) -> None:
        self.complete_sim_min = sim_min

    # ---- Barcode ops for the stock-retrieval step ----
    def assign_missing_barcodes(self, factory: Callable[[], str]) -> int:
        """
        Assign barcodes to any samples missing one.
        Returns how many were assigned.
        """
        assigned = 0
        for s in self.samples:
            if not s.barcode:
                s.set_barcode(factory())
                assigned += 1
        return assigned

    def verify_barcodes_unique(self) -> None:
        """
        Ensure all ASSIGNED barcodes are unique. Unassigned are allowed pre-retrieval.
        """
        assigned_count = sum(1 for s in self.samples if s.barcode)
        if len(self.barcodes) != assigned_count:
            raise DuplicateSampleError("Duplicate barcodes detected among assigned samples.")
