from lab_twin.domain.process import Process, Stage
from lab_twin.domain.barcodes import make_2d_barcode

# GSTT Lab Process -----------------------------------------------------------------------------

def do_wgs_worksheet(batch):
    print(f"WGS worksheet generated for batch {batch.batch_id}.")

def do_stock_retrieval(batch):
    assigned = batch.assign_missing_barcodes(make_2d_barcode)
    batch.verify_barcodes_unique()
    print(f"[{batch.batch_id}] stock retrieved: assigned={assigned}, total={batch.occupancy}")

DAY1_ARRIVAL = Process(
    process_id="day1_arrival",
    name="Day 1: arrival",
    stage=Stage.GSTT_PREPCR,
    actor_role="TBC",
    resources=tuple(),
    duration_min_min=0,
    duration_max_min=0,
    is_actionable=False,
    description="Marks the arrival time of the batch at GSTT Lab.",
)

WGS_WORKSHEET = Process(
    process_id="wgs_worksheet_generated",
    name="WGS worksheet generated",
    stage=Stage.GSTT_PREPCR,
    actor_role="Band 7",
    resources=("LIMS:GW",),
    duration_min_min=5,
    duration_max_min=10,
    is_actionable=True,
    run=do_wgs_worksheet,
    description="Generate WGS worksheet in LIMS:GW for the batch.",
)

STOCK_RETRIEVAL = Process(
    process_id="stock_dna_retrieval",
    name="Stock DNA samples retrieved and checked",
    stage=Stage.GSTT_PREPCR,
    actor_role="Band 5",
    resources=("Fridge",),
    duration_min_min=5,
    duration_max_min=5,
    is_actionable=True,
    run=do_stock_retrieval,
    description="Retrieve stock DNA samples from fridge, assign barcodes."
)

UNCAP_TUBES = Process(
    process_id="uncapping_tubes",
    name="Uncap tubes",
    stage=Stage.GSTT_PREPCR,
    actor_role="Band 5",
    resources=("LVL Capper/Decapper",),
    duration_min_min=10,
    duration_max_min=10,
    is_actionable=True,
    description="Uncap tubes using LVL Capper/Decapper."
)

TRANSFER_DNA = Process(
    process_id="transfer_dna",
    name="Transfer DNA to aliquot plate",
    stage=Stage.GSTT_PREPCR,
    actor_role="Band 5",
    resources=("TBC_TRANSFER_DNA",),
    duration_min_min=20,
    duration_max_min=20,
    is_actionable=True,
    description="Transfer DNA from tubes to aliquot plate.",
)

RECAP_TUBES = Process(
    process_id="recapping_tubes",
    name="Recap tubes",
    stage=Stage.GSTT_PREPCR,
    actor_role="Band 5",
    resources=("LVL Capper/Decapper",),
    duration_min_min=10,
    duration_max_min=10,
    is_actionable=True,
    description="Recap tubes using LVL Capper/Decapper.",
)

NUCLEIC_ACID_QC = Process(
    process_id="nucleic_acid_qc",
    name="Nucleic Acid QC",
    stage=Stage.GSTT_PREPCR,
    actor_role="Band 5",
    resources=("Plate reader, Picogreen",),
    duration_min_min=4,
    duration_max_min=4,
    is_actionable=True,
    description="Perform Nucleic Acid QC using Plate reader and Picogreen.",
)

QUANTIFICATION_READING = Process(
    process_id="quantification_reading",
    name="Quantification reading",
    stage=Stage.GSTT_PREPCR,
    actor_role="Band 5",
    resources=("Plate reader",),
    duration_min_min=20,
    duration_max_min=20,
    is_actionable=True,
    description="Read quantification using Plate reader.",
)

CREATE_DILUTION_TUBES = Process(
    process_id="create_dilution_tubes",
    name="Create daughter 2D dilution tubes",
    stage=Stage.GSTT_PREPCR,
    actor_role="Band 5",
    resources=("Hamilton Star",),
    duration_min_min=15,
    duration_max_min=15,
    is_actionable=True,
    description="Create daughter 2D dilution tubes using Hamilton Star.",
)

CREATE_PLATE_FOR_TRANSFER_TO_HUB = Process(
    process_id="create_plate_for_transfer_to_hub",
    name="Create plate of 2d tubes samples for transfer to the Hub",
    stage=Stage.GSTT_PREPCR,
    actor_role="Band 5",
    resources=("Hamilton Star","LVL Capper/Decapper",),
    duration_min_min=20,
    duration_max_min=20,
    is_actionable=True,
    description="Create plate of 2D tube samples for transfer to the Hub using Hamilton Star with capper/decapper and create " \
                "a manifest that specifies what is in plate - preferably electronic " \
                "linked LIMS that populates from scanning barcodes, perhaps only for WGS for simplicity",
)

STOCK_DNA_FREEZER = Process(
    process_id="stock_dna_freezer",
    name="Stock DNA sample back in freezer and plates in the fridge",
    stage=Stage.GSTT_PREPCR,
    actor_role="Band 5",
    resources=("Freezer","Fridge"),
    duration_min_min=10,
    duration_max_min=10,
    is_actionable=True,
    description=""
)

RACK_OF_BARCODED_TUBES = Process(
    process_id="rack_of_barcoded_tubes",
    name="Rack of 2D barcoded tubes not plates",
    stage=Stage.GSTT_PREPCR,
    actor_role="Band 5",
    resources=("TBC_RACK_HANDLER",), #To Be Confirmed
    duration_min_min=0, #To Be Confirmed
    duration_max_min=0, #To Be Confirmed
    is_actionable=True,
    description=""
)
