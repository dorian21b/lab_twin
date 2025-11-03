import random

# deterministic hooks: if you set batch.qc_pass = True/False, we’ll use it; else use a probability
def qc_pass(batch) -> bool:
    if hasattr(batch, "qc_pass"):
        return bool(batch.qc_pass)
    return random.random() < 0.85  # tweak

def library_qc_pass(batch) -> bool:
    if hasattr(batch, "lib_qc_pass"):
        return bool(batch.lib_qc_pass)
    return random.random() < 0.90  # tweak

def sequencing_success(batch) -> bool:
    if hasattr(batch, "seq_success"):
        return bool(batch.seq_success)
    return random.random() < 0.95  # tweak

def report_is_variant(batch) -> bool:
    if hasattr(batch, "is_variant"):
        return bool(batch.is_variant)
    return random.random() < 0.30  # ~30% variant, 70% negative
