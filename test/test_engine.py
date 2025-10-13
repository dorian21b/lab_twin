import random
from lab_twin.domain.sample import Sample
from lab_twin.domain.batch import Batch
from lab_twin.workflow.steps import DAY1_ARRIVAL, WGS_WORKSHEET
from lab_twin.sim.engine import run_sim


def test_engine_run_basic(capsys):
    """Smoke test: engine runs one batch through two steps."""
    random.seed(0)  # make duration deterministic for test

    batch = Batch(batch_id="TEST_001")
    batch.add_samples([Sample(barcode=f"BC{i:03d}") for i in range(56)])
    steps = [DAY1_ARRIVAL, WGS_WORKSHEET]

    run_sim(batch, steps)

    out = capsys.readouterr().out
    assert "Day 1: arrival" in out
    assert "WGS worksheet generated" in out
    assert "TEST_001" in out
