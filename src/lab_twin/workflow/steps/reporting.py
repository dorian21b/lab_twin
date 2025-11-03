from lab_twin.domain.process import Process, Stage
from lab_twin.domain.barcodes import make_2d_barcode

# Reporting -----------------------------------------------------------------------------

FIRST_CHECK = Process(
    process_id="first_qc_check",
    name="First QC check",
    stage=Stage.REPORTING,
    actor_role="Band 7",
    resources=("TBC_REPORT_WS",),
    duration_min_min=240,
    duration_max_min=240,
    is_actionable=True,
    description="",
) # S/B: sample

SECOND_CHECK = Process(
    process_id="second_qc_check",
    name="First QC check",
    stage=Stage.REPORTING,
    actor_role="Band 7",
    resources=("TBC_REPORT_WS",),
    duration_min_min=90,
    duration_max_min=90,
    is_actionable=True,
    description="",
)  # S/B: sample


"""----------------------------------------------------------------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
"""TODO DECISION TO CODE:  ------ % split neg or variant? ---------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""

"""Negative 70% case"""

WRITE_REPORT_NEG = Process(
    process_id="write_report_neg",
    name="Write report NEGATIVE",
    stage=Stage.REPORTING,
    actor_role="Band 7",
    resources=("TBC_REPORT_WS",),
    duration_min_min=15,
    duration_max_min=15,
    is_actionable=True,
    description="",
)

AUTHORISE_REPORT_NEG = Process(
    process_id="authorise_report_neg",
    name="Authorise report NEGATIVE",
    stage=Stage.REPORTING,
    actor_role="Band 7",
    resources=("TBC_REPORT_WS",),
    duration_min_min=15,
    duration_max_min=15,
    is_actionable=True,
    description="",
)

"""Variant 30% case"""

CONFIRM_VARIANTS = Process(
    process_id="confirm_variants",
    name="Confirm Variants Monogenics subprocess (ONT)",
    stage=Stage.REPORTING,
    actor_role="Band 7",
    resources=("TBC_REPORT_WS",),
    duration_min_min=0,
    duration_max_min=0,
    is_actionable=True,
    description="",
)

WRITE_REPORT_POS = Process(
    process_id="write_report_pos",
    name="Write report NEGATIVE",
    stage=Stage.REPORTING,
    actor_role="Band 7",
    resources=("TBC_REPORT_WS",),
    duration_min_min=60,
    duration_max_min=60,
    is_actionable=True,
    description="",
)

AUTHORISE_REPORT_VAR = Process(
    process_id="authorise_report_var",
    name="Authorise report VARIANT",
    stage=Stage.REPORTING,
    actor_role="Band 8",
    resources=("TBC_REPORT_WS",),
    duration_min_min=30,
    duration_max_min=30,
    is_actionable=True,
    description="",
)
