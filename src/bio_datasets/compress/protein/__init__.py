__all__ = [
    "ProteinBackboneCompressor",
    "HistogramEncoding",
    "HuffmanEncoding",
    "BinEncoding",
]


from .compress import ProteinBackboneCompressor, load_bb_histogram_compressor


COMPRESSORS = {
    "biotite_afdb": load_bb_histogram_compressor(
        "data/library/e_coli_full.npz"
    ),
}