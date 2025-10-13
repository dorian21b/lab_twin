# domain/barcodes.py
import uuid


def make_2d_barcode(prefix: str = "2D") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10].upper()}"
