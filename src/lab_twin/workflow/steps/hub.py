from lab_twin.domain.process import Process, Stage
from lab_twin.domain.barcodes import make_2d_barcode

# Hub Lab Process -----------------------------------------------------------------------------

COURIER_DELIVERING_HUB = Process(
    process_id="courier_delivering_hub",
    name="Courier delivers sample to the HUB",
    stage=Stage.HUB_PREPCR,
    actor_role="TBC",         #To Be Confirmed
    resources=("TBC_HUB_COURIER",),        #To Be Confirmed
    duration_min_min=0,    #To Be Confirmed
    duration_max_min=0,    #To Be Confirmed
    is_actionable=True,
    description=""
)

SR_PROCESS_STEP = Process(
    process_id="sr_process_step",
    name="TBC delivery at the Hub Central SR process step",
    stage=Stage.HUB_PREPCR,
    actor_role="TBC",         #To Be Confirmed
    resources=("TBC_HUB_SR_PROCESS",),        #To Be Confirmed
    duration_min_min=0,    #To Be Confirmed
    duration_max_min=0,    #To Be Confirmed
    is_actionable=True,
    description=""
)

TRANSPORT_TO_7TH_FLOOR = Process(
    process_id="transport_to_7th_floor",
    name="Samples transported to floor 7 at the Hub",
    stage=Stage.HUB_PREPCR,
    actor_role="",         #To Be Confirmed
    resources=("Telelift","Kevin",),        #To Be Confirmed
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
    process_id="store_if_late",
    name="Store if received after process cutoff point",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("LIMS fridge",),
    duration_min_min=5,
    duration_max_min=5,
    is_actionable=True,
    description=""
)


"""----------------------------------------------------------------------------------------------------------------------------"""

DAY2 = Process(
    process_id="day2",
    name="Day 2",
    stage=Stage.HUB_PREPCR,
    actor_role="",
    resources=tuple(),
    duration_min_min=0,
    duration_max_min=0,
    is_actionable=False,
    description="Need PC to scan plate but where does results go? Filemaker Pro for example? Need to scan on deck of robot to show what plate is in use firefly?",
)

"""----------------------------------------------------------------------------------------------------------------------------"""

RETRIEVE_AND_SCAN = Process(
    process_id="retrieve_and_scan",
    name="Retrieve plate, tubes and scan",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("LIMS", "Scanner",),
    duration_min_min=5,
    duration_max_min=5,
    is_actionable=True,
    description="Retrieve stored plates, tubes, and scan into LIMS system."
)

"""
In the event of limited DNA availability, a reduced concentration of DNA can be
submitted on the understanding that only a single WGS attempt will be made and
high quality full sequencing coverage may not be achievable.
Suboptimal Pathway: Minimum volume of 60ul at a concentration of 20ng/ul
(preferably higher), along with an A260/280 ratio within the 1.75-2.04 range.
Low Concentration input :Minimum volume of 90ul at a concentration of 12ng/ul
(preferably higher), along with an A260/280 ratio within the 1.75-2.04 range.
"""

SPIN_TUBES = Process(
    process_id="spin_tubes",
    name="Spin 2D tubes in plates",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Plate centrifuge",),
    duration_min_min=1,
    duration_max_min=1,
    is_actionable=True,
    description="Spin 2D tubes in plate centrifuge to ensure all liquid is at the bottom of the tube.",
)

SCAN_PLATE = Process(
    process_id="scan_plate",
    name="Scan plate into LIMS to confirm correct",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("LIMS", "Plate scanner",),
    duration_min_min=5,
    duration_max_min=5,
    is_actionable=True,
    description="Scan plate into LIMS to confirm correct samples are present and in the correct orientation.",
)

SETUP_LUNATIC = Process(
    process_id="setup_lunatic",
    name="Setup Lunatic for DNA quantification",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Lunatic",),
    duration_min_min=15,
    duration_max_min=15,
    is_actionable=True,
    description="Setup Lunatic for DNA quantification.",
)

PREPARE_ROBOT = Process(
    process_id="prepare_robot",
    name="Prepare Hamilton Star for DNA dilution",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Firefly +",),
    duration_min_min=15,
    duration_max_min=15,
    is_actionable=True,
    description="Prepare Hamilton Star for DNA dilution.",
)

START_LUNATIC = Process(
    process_id="start_lunatic",
    name="Start Lunatic run",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Firefly +",),
    duration_min_min=5,
    duration_max_min=5,
    is_actionable=True,
    description="Start Lunatic run for DNA quantification.",
)

TRANSFER_TO_QC_PLATE = Process(
    process_id="transfer_to_qc_plate",
    name="Transfer to QC plate of DNA",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Firefly +",),
    duration_min_min=30,
    duration_max_min=30,
    is_actionable=True,
    description="Transfer DNA from 2D tubes to QC plate using Hamilton Star.",
)

DNA_QC_PLATE = Process(
    process_id="dna_qc_plate",
    name="DNA QC plate",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Lunatic",),
    duration_min_min=30,
    duration_max_min=30,
    is_actionable=True,
    description="Perform DNA QC using Plate reader.",
)

RECORD_QC = Process(
    process_id="record_qc",
    name="Record QC",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("LIMS",),
    duration_min_min=10,
    duration_max_min=10,
    is_actionable=True,
    description="Record QC results into LIMS.",
)

"""----------------------------------------------------------------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
"""TODO DECISION TO CODE:  ------ Pass? ---------------------------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""


TRANSFER_DNA_TO_LIBRARY = Process(
    process_id="transfer_dna_to_library_prep_plate",
    name="Transfer of DNA to DNA library prep plate",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Firefly +",),
    duration_min_min=390,
    duration_max_min=390,
    is_actionable=False,
    description="",
)

END_OF_DAY_ROBOT_MAINTENANCE = Process(
    process_id="end_of_day_robot_maintanance",
    name="End of day robot maintanance",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Firefly +",),
    duration_min_min=30,
    duration_max_min=30,
    is_actionable=False,
    description="",
)

UPDATE_TREND_LOG = Process(
    process_id="update_trend_log",
    name="Update trend log",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("LIMS",),
    duration_min_min=10,
    duration_max_min=10,
    is_actionable=True,
    description="",
)

STORE = Process(
    process_id="store",
    name="Store",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("LIMS fridge",),
    duration_min_min=5,
    duration_max_min=5,
    is_actionable=False,
    description="",
)


"""----------------------------------------------------------------------------------------------------------------------------"""

DAY3 = Process(
    process_id="day3",
    name="Day 3",
    stage=Stage.HUB_PREPCR,
    actor_role="",
    resources=tuple(),
    duration_min_min=0,
    duration_max_min=0,
    is_actionable=False,
    description="",
)

"""----------------------------------------------------------------------------------------------------------------------------"""


PLACE_LIBRARY_PLATE_ON_ROBOT = Process(
    process_id="place_library_plate_on_robot",
    name="Place DNA library prep plate on robot",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Firefly +",),
    duration_min_min=5,
    duration_max_min=5,
    is_actionable=True,
    description="",
)

QC_LIBRARY_PLATE = Process(
    process_id="qc_library_plate",
    name="QC library plate",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Quantstudio 5",),
    duration_min_min=180,
    duration_max_min=180,
    is_actionable=True,
    description="",
)

ROBOT_POOLING = Process(
    process_id="robot_pooling",
    name="Robot pooling",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Firefly +",),
    duration_min_min=45,
    duration_max_min=45,
    is_actionable=True,
    description="",
)

QC_POOLS = Process(
    process_id="qc_pools",
    name="QC pools",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Quantstudio 5",),
    duration_min_min=30,
    duration_max_min=30,
    is_actionable=True,
    description="",
)

"""----------------------------------------------------------------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
"""TODO DECISION TO CODE:  ------ Below 100ng DNA -----------------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""

FINAL_CONC_ADJUSTMENT = Process(
    process_id="final_concentration_adjustment",
    name="Final concentration adjustment",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("TBC_FINAL_ADJUSTMENT",),
    duration_min_min=10,
    duration_max_min=10,
    is_actionable=True,
    description="",
)

QUBIT = Process(
    process_id="qubit",
    name="Qubit",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Qubit",),
    duration_min_min=15,
    duration_max_min=15,
    is_actionable=True,
    description="",
)
###PROBLEM WITH BATCH      60

"""
Load 1 flowcell of 56 samples every M,W,
and F for WGS. Need to load 2nd flowcell at
this point if we are using it

Fill other flow cell with: MSK access?, 10B
96, 16 on 1.5B, 25B?
Costs £176 for 8 on Aviti, 1.5B is £142, 10B
is £87

"""


SEQUENCER_LOADING = Process(
    process_id="sequencer_loading",
    name="Load sequencer",
    stage=Stage.HUB_POSTPCR,
    actor_role="Band 5",
    resources=("Novaseq X +",),
    duration_min_min=30,
    duration_max_min=30,
    is_actionable=True,
    description="",
)   # S/B:Batch 56-96 x2

SEQUENCING = Process(
    process_id="sequencing",
    name="Sequencing",
    stage=Stage.HUB_POSTPCR,
    actor_role="Band 5",
    resources=("Novaseq X +",),
    duration_min_min=2880,
    duration_max_min=2880,
    is_actionable=False,
    description="",
)   # S/B:Batch 56-96 x2    # 48H to do paired end 150 on 25B flow cell

AFTER_RUN_PREPARATION = Process(
    process_id="after_run_preparation",
    name="After run preparation",
    stage=Stage.HUB_POSTPCR,
    actor_role="Band 5",
    resources=("TBC_AFTER_RUN_PREP",),
    duration_min_min=15,
    duration_max_min=15,
    is_actionable=True,
    description="",
)   # S/B:Batch 56-96 x2


"""----------------------------------------------------------------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
"""TODO DECISION TO CODE:  ------ Sequencing successful? ----------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""

"""Scenario fail {"""

FAILED_DNA_QC = Process(
    process_id="failed_dna_qc",
    name="Failed DNA QC",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("Lunatic",),
    duration_min_min=30,
    duration_max_min=30,
    is_actionable=False,
    description="",
)

RECORD_QC_FAIL = Process(
    process_id="record_qc_fail",
    name="Record QC",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("LIMS",),
    duration_min_min=10,
    duration_max_min=10,
    is_actionable=True,
    description="",
)

"""----------------------------------------------------------------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
"""TODO DECISION TO CODE:  ------ Pass? ---------------------------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""


"""YES Repeat sequencing"""

"""No"""

REJECT_UNSUITABLE = Process(
    process_id="reject_unsuitable",
    name="Reject unsuitable samples",
    stage=Stage.HUB_PREPCR,
    actor_role="Band 5",
    resources=("LIMS",),
    duration_min_min=10,
    duration_max_min=10,
    is_actionable=True,
    description="",
)

"""}"""


"""Scenario Success"""


UPLOAD_TO_DRAGEN = Process(
    process_id="upload_to_dragen",
    name="Upload to Dragen for demultiplexing by Bioinformatics",
    stage=Stage.HUB_POSTPCR,
    actor_role="Band 5",
    resources=("LIMS",),
    duration_min_min=60,
    duration_max_min=60,
    is_actionable=True,
    description="",
)

PUSH_TO_GEL = Process(
    process_id="push_to_gel",
    name="Push data to GEL",
    stage=Stage.HUB_POSTPCR,
    actor_role="Band 5",
    resources=("LIMS",),
    duration_min_min=180,
    duration_max_min=300,
    is_actionable=True,
    description="",
)
