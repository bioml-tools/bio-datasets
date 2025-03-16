from biotite.structure.io.pdbx import encoding as biotite_encoding

from .encoding import BinEncoding, HistogramEncoding, HuffmanEncoding

biotite_encoding._encoding_classes["Histogram"] = HistogramEncoding
biotite_encoding._encoding_classes_kinds["HistogramEncoding"] = "Histogram"
biotite_encoding._encoding_classes["Huffman"] = HuffmanEncoding
biotite_encoding._encoding_classes_kinds["HuffmanEncoding"] = "Huffman"
biotite_encoding._encoding_classes["Bin"] = BinEncoding
biotite_encoding._encoding_classes_kinds["BinEncoding"] = "Bin"
