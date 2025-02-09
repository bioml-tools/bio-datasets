__all__ = [
    "ProteinBackboneCompressor",
    "HistogramEncoding",
    "HuffmanEncoding",
    "BinEncoding",
]

import os
import sys
from .compress import ProteinBackboneCompressor, load_bb_histogram_compressor


sys.setrecursionlimit(10000)  # required for hffman


COMPRESSORS = {
    "biotite_afdb": load_bb_histogram_compressor(
        os.path.join(os.path.dirname(__file__), "library/e_coli_delta_full.npz")
    ),
}