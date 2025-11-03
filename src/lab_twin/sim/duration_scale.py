# lab_twin/sim/duration_scale.py
from __future__ import annotations
from typing import Dict, Iterable
from lab_twin.domain.process import Process

# Global (simple, explicit) registry
_DURATION_SCALE: Dict[str, float] = {}
_KNOWN_STEPS: Iterable[str] | None = None
_LOGGED_ONCE: set[str] = set()

def set_known_steps(step_ids: Iterable[str]) -> None:
    """Optional: tell the scaler what valid steps exist, for early erroring."""
    global _KNOWN_STEPS
    _KNOWN_STEPS = set(step_ids)


def set_duration_scale(scale_map: Dict[str, float], steps: Dict[str, Process], *, verbose: bool = True) -> None:
    """
    Apply per-step duration scaling. Example:
      scale_map = {"DNA_QC_PLATE": 0.90, "SEQUENCING": 0.90}
    """
    if not scale_map:
        return

    known = set(steps.keys())
    unknown = set(scale_map.keys()) - known
    if unknown:
        raise ValueError(f"Duration scale refers to unknown steps: {sorted(unknown)}")

    for k, factor in scale_map.items():
        p = steps[k]
        if verbose:
            print(f"[DURATION_SCALE] step={k} x{factor:.3f} "
                  f"min: {p.duration_min_min:.2f}->{p.duration_min_min*factor:.2f} "
                  f"max: {p.duration_max_min:.2f}->{p.duration_max_min*factor:.2f}")
        p.duration_min_min *= factor
        p.duration_max_min *= factor

def get_duration_scale() -> Dict[str, float]:
    return dict(_DURATION_SCALE)

def scale_service_time(step_id: str, base_time: float, log_fn=None) -> float:
    """Multiply the realized service time for this step if a scale is set."""
    factor = _DURATION_SCALE.get(step_id, 1.0)
    scaled = base_time * factor
    # One-time helpful log per scaled step
    if factor != 1.0 and step_id not in _LOGGED_ONCE:
        if callable(log_fn):
            log_fn(f"[DURATION_SCALE] step={step_id} factor={factor:.3f} "
                   f"base={base_time:.3f} scaled={scaled:.3f}")
        _LOGGED_ONCE.add(step_id)
    return scaled
