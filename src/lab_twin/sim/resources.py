# src/lab_twin/sim/resources.py
import simpy
from typing import Dict, Optional


def make_resources(env: simpy.Environment, overrides: Optional[Dict[str, int]] = None) -> Dict[str, simpy.Resource]:
    """
    Create shared resources (servers) for GSTT, HUB, and REPORTING labs.
    Add temporary 'TBC_*' placeholders for unknown resources.
    """
    overrides = overrides or {}

    # -------------------------------------------------------------------------
    # ---- HUMAN ROLES --------------------------------------------------------
    # -------------------------------------------------------------------------
    band7 = simpy.Resource(env, capacity=overrides.get("Band 7", 2))
    band5 = simpy.Resource(env, capacity=overrides.get("Band 5", 2))
    band8 = simpy.Resource(env, capacity=overrides.get("Band 8", 1))  # authorisation only

    # -------------------------------------------------------------------------
    # ---- DIGITAL / LIMS / SCANNERS -----------------------------------------
    # -------------------------------------------------------------------------
    lims = simpy.Resource(env, capacity=overrides.get("LIMS", 1))
    lims_gw = lims  # alias for GSTT "LIMS:GW"
    lims_signoff = lims
    scanner = simpy.Resource(env, capacity=overrides.get("Scanner", 1))
    lims_scanner = scanner
    plate_scanner = scanner
    lims_fridge = simpy.Resource(env, capacity=overrides.get("LIMS fridge", 1))

    # -------------------------------------------------------------------------
    # ---- STORAGE / LOGISTICS -----------------------------------------------
    # -------------------------------------------------------------------------
    fridge = simpy.Resource(env, capacity=overrides.get("Fridge", 1))
    freezer = simpy.Resource(env, capacity=overrides.get("Freezer", 1))
    telelift = simpy.Resource(env, capacity=overrides.get("Telelift", 1))
    kevin = simpy.Resource(env, capacity=overrides.get("Kevin", 1))   # human courier

    # -------------------------------------------------------------------------
    # ---- GSTT INSTRUMENTS ---------------------------------------------------
    # -------------------------------------------------------------------------
    capper = simpy.Resource(env, capacity=overrides.get("LVL Capper/Decapper", 1))
    plate_reader = simpy.Resource(env, capacity=overrides.get("Plate reader", 1))
    hamilton = simpy.Resource(env, capacity=overrides.get("Hamilton Star", 1))

    # -------------------------------------------------------------------------
    # ---- HUB INSTRUMENTS ----------------------------------------------------
    # -------------------------------------------------------------------------
    plate_centrifuge = simpy.Resource(env, capacity=overrides.get("Plate centrifuge", 1))
    lunatic = simpy.Resource(env, capacity=overrides.get("Lunatic", 1))
    firefly = simpy.Resource(env, capacity=overrides.get("Firefly +", 1))
    quantstudio = simpy.Resource(env, capacity=overrides.get("Quantstudio 5", 1))
    qubit = simpy.Resource(env, capacity=overrides.get("Qubit", 1))
    novaseq = simpy.Resource(env, capacity=overrides.get("Novaseq X +", 1))
    seq_alias = novaseq  # alias "Seq"

    # -------------------------------------------------------------------------
    # ---- REPORTING RESOURCES ------------------------------------------------
    # -------------------------------------------------------------------------
    reporting_ws = simpy.Resource(env, capacity=overrides.get("Reporting workstation", 2))
    genome_browser = simpy.Resource(env, capacity=overrides.get("Genome browser", 1))

    # -------------------------------------------------------------------------
    # ---- TEMPORARY PLACEHOLDERS (TBC) --------------------------------------
    # -------------------------------------------------------------------------
    tbc_transfer_dna = simpy.Resource(env, capacity=overrides.get("TBC_TRANSFER_DNA", 1))
    tbc_rack_handler = simpy.Resource(env, capacity=overrides.get("TBC_RACK_HANDLER", 1))
    tbc_hub_courier = simpy.Resource(env, capacity=overrides.get("TBC_HUB_COURIER", 1))
    tbc_hub_sr = simpy.Resource(env, capacity=overrides.get("TBC_HUB_SR_PROCESS", 1))
    tbc_final_adj = simpy.Resource(env, capacity=overrides.get("TBC_FINAL_ADJUSTMENT", 1))
    tbc_after_prep = simpy.Resource(env, capacity=overrides.get("TBC_AFTER_RUN_PREP", 1))
    tbc_report_ws = simpy.Resource(env, capacity=overrides.get("TBC_REPORT_WS", 2))

    # -------------------------------------------------------------------------
    # ---- ASSEMBLE DICTIONARY ------------------------------------------------
    # -------------------------------------------------------------------------
    resources = {
        # Humans
        "Band 7": band7,
        "Band 5": band5,
        "Band 8": band8,

        # Digital / LIMS
        "LIMS": lims,
        "LIMS:GW": lims_gw,
        "LIMS sign-off": lims_signoff,
        "LIMS scanner": lims_scanner,
        "Scanner": scanner,
        "Plate scanner": plate_scanner,
        "LIMS fridge": lims_fridge,

        # Storage / logistics
        "Fridge": fridge,
        "Freezer": freezer,
        "Telelift": telelift,
        "Kevin": kevin,

        # GSTT instruments
        "LVL Capper/Decapper": capper,
        "Plate reader": plate_reader,
        "Hamilton Star": hamilton,

        # HUB instruments
        "Plate centrifuge": plate_centrifuge,
        "Lunatic": lunatic,
        "Firefly +": firefly,
        "Quantstudio 5": quantstudio,
        "Qubit": qubit,
        "Novaseq X +": novaseq,
        "Seq": seq_alias,

        # REPORTING
        "Reporting workstation": reporting_ws,
        "Genome browser": genome_browser,

        # Temporary placeholders (TBC)
        "TBC_TRANSFER_DNA": tbc_transfer_dna,
        "TBC_RACK_HANDLER": tbc_rack_handler,
        "TBC_HUB_COURIER": tbc_hub_courier,
        "TBC_HUB_SR_PROCESS": tbc_hub_sr,
        "TBC_FINAL_ADJUSTMENT": tbc_final_adj,
        "TBC_AFTER_RUN_PREP": tbc_after_prep,
        "TBC_REPORT_WS": tbc_report_ws,
    }

    return resources
