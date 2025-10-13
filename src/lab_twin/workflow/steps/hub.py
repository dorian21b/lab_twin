from lab_twin.domain.process import Process, Stage
from lab_twin.domain.barcodes import make_2d_barcode

# Hub Lab Process -----------------------------------------------------------------------------

COURIER_DELIVERING_HUB = Process(
    process_id="courier_delivering_hub",
    name="Courier delivers sample to the HUB",
    stage=Stage.HUB_PREPCR,
    actor_role="",         #To Be Confirmed
    resources=(""),        #To Be Confirmed
    duration_min_min=0,    #To Be Confirmed
    duration_max_min=0,    #To Be Confirmed
    is_actionable=True,
    description=""
)

SR_PROCESS_STEP = Process(
    process_id="sr_process_step",
    name="TBC delivery at the Hub Central SR process step",
    stage=Stage.HUB_PREPCR,
    actor_role="",         #To Be Confirmed
    resources=(""),        #To Be Confirmed
    duration_min_min=0,    #To Be Confirmed
    duration_max_min=0,    #To Be Confirmed
    is_actionable=True,
    description=""
)

TRANSPORT_TO_7th_FLOOR = Process(
    process_id="transport_to_7th_floor",
    name="Samples transported to floor 7 at the Hub",
    stage=Stage.HUB_PREPCR,
    actor_role="",         #To Be Confirmed
    resources=("Telelift","Kevin"),        #To Be Confirmed
    duration_min_min=0,    #To Be Confirmed
    duration_max_min=0,    #To Be Confirmed
    is_actionable=True,
    description=""
)

RECEIPT_SCAN_2D = Process(
    process_id="receipt_scan_2d",
    name="Samples receipted, scan 2D plates and tubes",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("LIMS scanner",),
    duration_min_min=5,
    duration_max_min=5,
    is_actionable=True,
    description=""
)

STORE_IF_LATE = Process(
    process_id="receipt_scan_2d",
    name="Store if received after process cutoff point",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("LIMS fridge",),
    duration_min_min=5,
    duration_max_min=5,
    is_actionable=True,
    description=""
)

DAY2 = Process(
    process_id="day2",
    name="Day 2",
    stage=Stage.GSTT_PREPCR,
    actor_role="",
    resources=tuple(),
    duration_min_min=0,
    duration_max_min=0,
    is_actionable=False,
    description="",
)

RETRIEVE_AND_SCAN = Process(
    process_id="retrieve_and_scan",
    name="Retrieve plate, tubes and scan",
    stage=Stage.GSTT_PREPCR,
    actor_role="Band 5",
    resources=("LIMS", "Scanner"),
    duration_min_min=5,
    duration_max_min=5,
    is_actionable=True,
    description="Retrieve stored plates, tubes, and scan into LIMS system."
)