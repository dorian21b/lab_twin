# lab_twin/workflow/steps_scaled.py
from __future__ import annotations
from copy import deepcopy
from typing import Dict

from lab_twin.domain.process import Process

def _scale_process(p: Process, f: float) -> Process:
    """
    Return a scaled clone of Process p (min/max durations * f).
    Works whether Process is mutable or frozen.
    """
    q = deepcopy(p)
    # attribute names based on your engine.py usage
    if hasattr(q, "duration_min_min"):
        q.duration_min_min = float(q.duration_min_min) * f
    if hasattr(q, "duration_max_min"):
        q.duration_max_min = float(q.duration_max_min) * f
    return q

def make_steps_scaled(STEPS: Dict[str, Process], duration_scale: Dict[str, float]) -> Dict[str, Process]:
    """
    Return a new dict of {step_key: Process} with durations scaled
    for keys present in duration_scale (others unchanged).
    """
    if not duration_scale:
        return STEPS  # unchanged (safe to reuse original)
    out: Dict[str, Process] = {}
    for k, p in STEPS.items():
        f = float(duration_scale.get(k, 1.0))
        out[k] = _scale_process(p, f) if f != 1.0 else deepcopy(p)
    return out
