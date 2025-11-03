# import simpy
# from .nodes import ProcessNode, DecisionNode
# from .decisions import qc_pass, library_qc_pass, sequencing_success, report_is_variant

# def connect_chain(*nodes):
#     for a, b in zip(nodes, nodes[1:]):
#         a.connect(b)
#     return nodes[0], nodes[-1]

# def build_full_graph(env: simpy.Environment, steps: dict, log_event, resources: dict[str, simpy.Resource]):
#     S = steps  # dict: name -> Process

#     # --- simple name normalisation so step.resources can vary a little
#     ALIAS = {
#         "LIMS:GW": "LIMS",
#         "Plate reader, Picogreen": "Plate reader",
#         "Plate reader / Picogreen": "Plate reader",
#         "Seq": "Novaseq X +",
#         "Plate scanner": "Plate scanner",
#         "LIMS scanner": "LIMS scanner",
#     }

#     def pick_server(proc) -> simpy.Resource | None:
#         raw = proc.resources or ()
#         if isinstance(raw, str):            # <-- guard
#             raw = (raw,)
#         for name in raw:
#             name = (name or "").strip()
#             if not name:
#                 continue
#             name = ALIAS.get(name, name)
#             if name in resources:
#                 return resources[name]
#         return None



#     # --- GSTT Day 1
#     n_wgs   = ProcessNode(env, S["WGS_WORKSHEET"],          log_event, resource=pick_server(S["WGS_WORKSHEET"]), is_head=True)
#     n_stock = ProcessNode(env, S["STOCK_RETRIEVAL"],        log_event, resource=pick_server(S["STOCK_RETRIEVAL"]))
#     n_uncap = ProcessNode(env, S["UNCAP_TUBES"],            log_event, resource=pick_server(S["UNCAP_TUBES"]))
#     n_xfer  = ProcessNode(env, S["TRANSFER_DNA"],           log_event, resource=pick_server(S["TRANSFER_DNA"]))
#     n_recap = ProcessNode(env, S["RECAP_TUBES"],            log_event, resource=pick_server(S["RECAP_TUBES"]))
#     n_na_qc = ProcessNode(env, S["NUCLEIC_ACID_QC"],        log_event, resource=pick_server(S["NUCLEIC_ACID_QC"]))
#     n_quant = ProcessNode(env, S["QUANTIFICATION_READING"], log_event, resource=pick_server(S["QUANTIFICATION_READING"]))
#     n_make_dil   = ProcessNode(env, S["CREATE_DILUTION_TUBES"],             log_event, resource=pick_server(S["CREATE_DILUTION_TUBES"]))
#     n_make_plate = ProcessNode(env, S["CREATE_PLATE_FOR_TRANSFER_TO_HUB"],  log_event, resource=pick_server(S["CREATE_PLATE_FOR_TRANSFER_TO_HUB"]))
#     n_restow     = ProcessNode(env, S["STOCK_DNA_FREEZER"], log_event, resource=pick_server(S["STOCK_DNA_FREEZER"]))
#     n_rack       = ProcessNode(env, S["RACK_OF_BARCODED_TUBES"], log_event, resource=pick_server(S["RACK_OF_BARCODED_TUBES"]))

#     connect_chain(n_wgs, n_stock, n_uncap, n_xfer, n_recap, n_na_qc,
#                   n_quant, n_make_dil, n_make_plate, n_restow, n_rack)


#     # --- HUB inbound
#     n_courier = ProcessNode(env, S["COURIER_DELIVERING_HUB"], log_event, resource=pick_server(S["COURIER_DELIVERING_HUB"]))
#     n_sr      = ProcessNode(env, S["SR_PROCESS_STEP"],       log_event, resource=pick_server(S["SR_PROCESS_STEP"]))
#     n_to7     = ProcessNode(env, S["TRANSPORT_TO_7TH_FLOOR"],log_event, resource=pick_server(S["TRANSPORT_TO_7TH_FLOOR"]))
#     n_rcpt    = ProcessNode(env, S["RECEIPT_SCAN_2D"],       log_event, resource=pick_server(S["RECEIPT_SCAN_2D"]))

#     connect_chain(n_rack, n_courier, n_sr, n_to7, n_rcpt)

#     # --- Day 2: Pre-QC
#     n_retrieve     = ProcessNode(env, S["RETRIEVE_AND_SCAN"], log_event, resource=pick_server(S["RETRIEVE_AND_SCAN"]))
#     n_spin         = ProcessNode(env, S["SPIN_TUBES"],        log_event, resource=pick_server(S["SPIN_TUBES"]))
#     n_scan_plate   = ProcessNode(env, S["SCAN_PLATE"],        log_event, resource=pick_server(S["SCAN_PLATE"]))
#     n_setup_luna   = ProcessNode(env, S["SETUP_LUNATIC"],     log_event, resource=pick_server(S["SETUP_LUNATIC"]))
#     n_prep_robot   = ProcessNode(env, S["PREPARE_ROBOT"],     log_event, resource=pick_server(S["PREPARE_ROBOT"]))
#     n_start_luna   = ProcessNode(env, S["START_LUNATIC"],     log_event, resource=pick_server(S["START_LUNATIC"]))
#     n_xfer_qc      = ProcessNode(env, S["TRANSFER_TO_QC_PLATE"], log_event, resource=pick_server(S["TRANSFER_TO_QC_PLATE"]))
#     n_dna_qc       = ProcessNode(env, S["DNA_QC_PLATE"],      log_event, resource=pick_server(S["DNA_QC_PLATE"]))
#     n_record_qc    = ProcessNode(env, S["RECORD_QC"],         log_event, resource=pick_server(S["RECORD_QC"]))

#     d_qc           = DecisionNode(env, "dna_qc_pass", qc_pass, log_event)
#     n_failed_qc    = ProcessNode(env, S["FAILED_DNA_QC"],     log_event, resource=pick_server(S["FAILED_DNA_QC"]))
#     n_record_qc_fail = ProcessNode(env, S["RECORD_QC"],       log_event, resource=pick_server(S["RECORD_QC"]))
#     n_reject_qc    = ProcessNode(env, S["REJECT_UNSUITABLE"], log_event, resource=pick_server(S["REJECT_UNSUITABLE"]))

#     connect_chain(n_rcpt, n_retrieve, n_spin, n_scan_plate, n_setup_luna,
#                   n_prep_robot, n_start_luna, n_xfer_qc, n_dna_qc, d_qc)

#     # QC branch
#     d_qc.connect_yes(n_record_qc)
#     connect_chain(n_failed_qc, n_record_qc_fail, n_reject_qc)
#     d_qc.connect_no(n_failed_qc)

#     # --- Day 3: Library / sequencing
#     n_place_lib = ProcessNode(env, S["PLACE_LIBRARY_PLATE_ON_ROBOT"], log_event, resource=pick_server(S["PLACE_LIBRARY_PLATE_ON_ROBOT"]))
#     n_qc_lib    = ProcessNode(env, S["QC_LIBRARY_PLATE"],             log_event, resource=pick_server(S["QC_LIBRARY_PLATE"]))

#     d_lib       = DecisionNode(env, "library_qc_pass", library_qc_pass, log_event)
#     n_reject_lib= ProcessNode(env, S["REJECT_UNSUITABLE"], log_event, resource=pick_server(S["REJECT_UNSUITABLE"]))

#     n_pool      = ProcessNode(env, S["ROBOT_POOLING"],           log_event, resource=pick_server(S["ROBOT_POOLING"]))
#     n_qc_pools  = ProcessNode(env, S["QC_POOLS"],                log_event, resource=pick_server(S["QC_POOLS"]))
#     n_final_adj = ProcessNode(env, S["FINAL_CONC_ADJUSTMENT"],   log_event, resource=pick_server(S["FINAL_CONC_ADJUSTMENT"]))
#     n_qubit     = ProcessNode(env, S["QUBIT"],                   log_event, resource=pick_server(S["QUBIT"]))
#     n_load      = ProcessNode(env, S["SEQUENCER_LOADING"],       log_event, resource=pick_server(S["SEQUENCER_LOADING"]))
#     n_seq       = ProcessNode(env, S["SEQUENCING"],              log_event, resource=pick_server(S["SEQUENCING"]))

#     d_seq       = DecisionNode(env, "sequencing_success", sequencing_success, log_event)
#     n_upload    = ProcessNode(env, S["UPLOAD_TO_DRAGEN"],        log_event, resource=pick_server(S["UPLOAD_TO_DRAGEN"]))
#     n_push_gel  = ProcessNode(env, S["PUSH_TO_GEL"],             log_event, resource=pick_server(S["PUSH_TO_GEL"]))

#     connect_chain(n_record_qc, n_place_lib, n_qc_lib, d_lib)
#     d_lib.connect_yes(n_pool)
#     d_lib.connect_no(n_reject_lib)

#     connect_chain(n_pool, n_qc_pools, n_final_adj, n_qubit, n_load, n_seq, d_seq)
#     d_seq.connect_yes(n_upload)
#     d_seq.connect_no(n_reject_qc)
#     connect_chain(n_upload, n_push_gel)

#     # --- Reporting
#     n_first   = ProcessNode(env, S["FIRST_CHECK"],  log_event, resource=pick_server(S["FIRST_CHECK"]))
#     n_second  = ProcessNode(env, S["SECOND_CHECK"], log_event, resource=pick_server(S["SECOND_CHECK"]))
#     d_report  = DecisionNode(env, "is_variant", report_is_variant, log_event)

#     n_write_neg = ProcessNode(env, S["WRITE_REPORT_NEG"],    log_event, resource=pick_server(S["WRITE_REPORT_NEG"]))
#     n_auth_neg  = ProcessNode(env, S["AUTHORISE_REPORT_NEG"],log_event, resource=pick_server(S["AUTHORISE_REPORT_NEG"]))

#     n_confirm_var = ProcessNode(env, S["CONFIRM_VARIANTS"],  log_event, resource=pick_server(S["CONFIRM_VARIANTS"]))
#     n_write_pos   = ProcessNode(env, S["WRITE_REPORT_POS"],  log_event, resource=pick_server(S["WRITE_REPORT_POS"]))
#     n_auth_var    = ProcessNode(env, S["AUTHORISE_REPORT_VAR"],log_event, resource=pick_server(S["AUTHORISE_REPORT_VAR"]))

#     connect_chain(n_push_gel, n_first, n_second, d_report)
#     d_report.connect_no(n_write_neg)
#     connect_chain(n_write_neg, n_auth_neg)

#     d_report.connect_yes(n_confirm_var)
#     connect_chain(n_confirm_var, n_write_pos, n_auth_var)

#     head = n_wgs
#     tails = [n_auth_neg, n_auth_var, n_reject_qc, n_reject_lib]

#     for t in tails:
#         t.is_tail = True
#     return head, tails

# src/lab_twin/sim/line_full.py
import simpy
from .nodes import ProcessNode, DecisionNode
from .decisions import qc_pass, library_qc_pass, sequencing_success, report_is_variant


def connect_chain(*nodes):
    for a, b in zip(nodes, nodes[1:]):
        a.connect(b)
    return nodes[0], nodes[-1]


def build_full_graph(
    env: simpy.Environment,
    steps: dict,
    log_event,
    resources: dict[str, simpy.Resource],
):
    """
    Build the full process graph and wire shared resources (servers).
    `steps` is your STEPS map (name -> Process).
    `resources` is the dict returned by make_resources(env, ...).
    """
    S = steps  # alias

    # Normalise a few resource name variants that appear in step definitions
    ALIAS = {
        "LIMS:GW": "LIMS",
        "Plate reader, Picogreen": "Plate reader",
        "Plate reader / Picogreen": "Plate reader",
        "Seq": "Novaseq X +",
        "Plate scanner": "Plate scanner",
        "LIMS scanner": "LIMS scanner",
    }

    def pick_server(proc) -> simpy.Resource | None:
        """Return the first declared resource for this step that exists in `resources`."""
        raw = proc.resources or ()
        if isinstance(raw, str):           # guard against the common 1-tuple mistake
            raw = (raw,)
        for name in raw:
            name = (name or "").strip()
            if not name:
                continue
            name = ALIAS.get(name, name)
            if name in resources:
                return resources[name]
        return None  # step executes without an explicit shared server

    # ---------------------------
    # GSTT Day 1
    # ---------------------------
    n_wgs   = ProcessNode(env, S["WGS_WORKSHEET"],          log_event, resource=pick_server(S["WGS_WORKSHEET"]), is_head=True)
    n_stock = ProcessNode(env, S["STOCK_RETRIEVAL"],        log_event, resource=pick_server(S["STOCK_RETRIEVAL"]))
    n_uncap = ProcessNode(env, S["UNCAP_TUBES"],            log_event, resource=pick_server(S["UNCAP_TUBES"]))
    n_xfer  = ProcessNode(env, S["TRANSFER_DNA"],           log_event, resource=pick_server(S["TRANSFER_DNA"]))
    n_recap = ProcessNode(env, S["RECAP_TUBES"],            log_event, resource=pick_server(S["RECAP_TUBES"]))
    n_na_qc = ProcessNode(env, S["NUCLEIC_ACID_QC"],        log_event, resource=pick_server(S["NUCLEIC_ACID_QC"]))
    n_quant = ProcessNode(env, S["QUANTIFICATION_READING"], log_event, resource=pick_server(S["QUANTIFICATION_READING"]))
    n_make_dil   = ProcessNode(env, S["CREATE_DILUTION_TUBES"],            log_event, resource=pick_server(S["CREATE_DILUTION_TUBES"]))
    n_make_plate = ProcessNode(env, S["CREATE_PLATE_FOR_TRANSFER_TO_HUB"], log_event, resource=pick_server(S["CREATE_PLATE_FOR_TRANSFER_TO_HUB"]))
    n_restow     = ProcessNode(env, S["STOCK_DNA_FREEZER"],  log_event, resource=pick_server(S["STOCK_DNA_FREEZER"]))
    n_rack       = ProcessNode(env, S["RACK_OF_BARCODED_TUBES"], log_event, resource=pick_server(S["RACK_OF_BARCODED_TUBES"]))

    connect_chain(n_wgs, n_stock, n_uncap, n_xfer, n_recap, n_na_qc,
                  n_quant, n_make_dil, n_make_plate, n_restow, n_rack)

    # ---------------------------
    # HUB inbound
    # ---------------------------
    n_courier = ProcessNode(env, S["COURIER_DELIVERING_HUB"], log_event, resource=pick_server(S["COURIER_DELIVERING_HUB"]))
    n_sr      = ProcessNode(env, S["SR_PROCESS_STEP"],        log_event, resource=pick_server(S["SR_PROCESS_STEP"]))
    n_to7     = ProcessNode(env, S["TRANSPORT_TO_7TH_FLOOR"], log_event, resource=pick_server(S["TRANSPORT_TO_7TH_FLOOR"]))
    n_rcpt    = ProcessNode(env, S["RECEIPT_SCAN_2D"],        log_event, resource=pick_server(S["RECEIPT_SCAN_2D"]))

    connect_chain(n_rack, n_courier, n_sr, n_to7, n_rcpt)

    # ---------------------------
    # Day 2: Pre-QC
    # ---------------------------
    n_retrieve   = ProcessNode(env, S["RETRIEVE_AND_SCAN"],    log_event, resource=pick_server(S["RETRIEVE_AND_SCAN"]))
    n_spin       = ProcessNode(env, S["SPIN_TUBES"],           log_event, resource=pick_server(S["SPIN_TUBES"]))
    n_scan_plate = ProcessNode(env, S["SCAN_PLATE"],           log_event, resource=pick_server(S["SCAN_PLATE"]))
    n_setup_luna = ProcessNode(env, S["SETUP_LUNATIC"],        log_event, resource=pick_server(S["SETUP_LUNATIC"]))
    n_prep_robot = ProcessNode(env, S["PREPARE_ROBOT"],        log_event, resource=pick_server(S["PREPARE_ROBOT"]))
    n_start_luna = ProcessNode(env, S["START_LUNATIC"],        log_event, resource=pick_server(S["START_LUNATIC"]))
    n_xfer_qc    = ProcessNode(env, S["TRANSFER_TO_QC_PLATE"], log_event, resource=pick_server(S["TRANSFER_TO_QC_PLATE"]))
    n_dna_qc     = ProcessNode(env, S["DNA_QC_PLATE"],         log_event, resource=pick_server(S["DNA_QC_PLATE"]))
    n_record_qc  = ProcessNode(env, S["RECORD_QC"],            log_event, resource=pick_server(S["RECORD_QC"]))

    d_qc           = DecisionNode(env, "dna_qc_pass", qc_pass, log_event)
    n_failed_qc    = ProcessNode(env, S["FAILED_DNA_QC"],      log_event, resource=pick_server(S["FAILED_DNA_QC"]))
    n_record_qc_fail = ProcessNode(env, S["RECORD_QC"],        log_event, resource=pick_server(S["RECORD_QC"]))
    n_reject_qc    = ProcessNode(env, S["REJECT_UNSUITABLE"],  log_event, resource=pick_server(S["REJECT_UNSUITABLE"]))

    connect_chain(n_rcpt, n_retrieve, n_spin, n_scan_plate, n_setup_luna,
                  n_prep_robot, n_start_luna, n_xfer_qc, n_dna_qc, d_qc)

    # QC branches
    d_qc.connect_yes(n_record_qc)
    connect_chain(n_failed_qc, n_record_qc_fail, n_reject_qc)
    d_qc.connect_no(n_failed_qc)

    # ---------------------------
    # Day 3: Library / Sequencing
    # ---------------------------
    n_place_lib = ProcessNode(env, S["PLACE_LIBRARY_PLATE_ON_ROBOT"], log_event, resource=pick_server(S["PLACE_LIBRARY_PLATE_ON_ROBOT"]))
    n_qc_lib    = ProcessNode(env, S["QC_LIBRARY_PLATE"],             log_event, resource=pick_server(S["QC_LIBRARY_PLATE"]))

    d_lib        = DecisionNode(env, "library_qc_pass", library_qc_pass, log_event)
    n_reject_lib = ProcessNode(env, S["REJECT_UNSUITABLE"], log_event, resource=pick_server(S["REJECT_UNSUITABLE"]))

    n_pool      = ProcessNode(env, S["ROBOT_POOLING"],         log_event, resource=pick_server(S["ROBOT_POOLING"]))
    n_qc_pools  = ProcessNode(env, S["QC_POOLS"],              log_event, resource=pick_server(S["QC_POOLS"]))
    n_final_adj = ProcessNode(env, S["FINAL_CONC_ADJUSTMENT"], log_event, resource=pick_server(S["FINAL_CONC_ADJUSTMENT"]))
    n_qubit     = ProcessNode(env, S["QUBIT"],                 log_event, resource=pick_server(S["QUBIT"]))
    n_load      = ProcessNode(env, S["SEQUENCER_LOADING"],     log_event, resource=pick_server(S["SEQUENCER_LOADING"]))
    n_seq       = ProcessNode(env, S["SEQUENCING"],            log_event, resource=pick_server(S["SEQUENCING"]))

    d_seq      = DecisionNode(env, "sequencing_success", sequencing_success, log_event)
    n_upload   = ProcessNode(env, S["UPLOAD_TO_DRAGEN"], log_event, resource=pick_server(S["UPLOAD_TO_DRAGEN"]))
    n_push_gel = ProcessNode(env, S["PUSH_TO_GEL"],      log_event, resource=pick_server(S["PUSH_TO_GEL"]))

    connect_chain(n_record_qc, n_place_lib, n_qc_lib, d_lib)
    d_lib.connect_yes(n_pool)
    d_lib.connect_no(n_reject_lib)

    connect_chain(n_pool, n_qc_pools, n_final_adj, n_qubit, n_load, n_seq, d_seq)
    d_seq.connect_yes(n_upload)
    d_seq.connect_no(n_reject_qc)  # reuse reject as failure terminal
    connect_chain(n_upload, n_push_gel)

    # ---------------------------
    # Reporting
    # ---------------------------
    n_first  = ProcessNode(env, S["FIRST_CHECK"],  log_event, resource=pick_server(S["FIRST_CHECK"]))
    n_second = ProcessNode(env, S["SECOND_CHECK"], log_event, resource=pick_server(S["SECOND_CHECK"]))
    d_report = DecisionNode(env, "is_variant", report_is_variant, log_event)

    n_write_neg = ProcessNode(env, S["WRITE_REPORT_NEG"],     log_event, resource=pick_server(S["WRITE_REPORT_NEG"]))
    n_auth_neg  = ProcessNode(env, S["AUTHORISE_REPORT_NEG"], log_event, resource=pick_server(S["AUTHORISE_REPORT_NEG"]))

    n_confirm_var = ProcessNode(env, S["CONFIRM_VARIANTS"],   log_event, resource=pick_server(S["CONFIRM_VARIANTS"]))
    n_write_pos   = ProcessNode(env, S["WRITE_REPORT_POS"],   log_event, resource=pick_server(S["WRITE_REPORT_POS"]))
    n_auth_var    = ProcessNode(env, S["AUTHORISE_REPORT_VAR"], log_event, resource=pick_server(S["AUTHORISE_REPORT_VAR"]))

    connect_chain(n_push_gel, n_first, n_second, d_report)
    d_report.connect_no(n_write_neg)          # ~70% negatives
    connect_chain(n_write_neg, n_auth_neg)

    d_report.connect_yes(n_confirm_var)       # ~30% variants
    connect_chain(n_confirm_var, n_write_pos, n_auth_var)

    # ---------------------------
    # Return head/tails
    # ---------------------------
    head = n_wgs
    tails = [n_auth_neg, n_auth_var, n_reject_qc, n_reject_lib]
    for t in tails:
        t.is_tail = True
    return head, tails
