"""We have several options for bit packing:

1. Return bools and pack with np.packbits
2. Return bools and pass to pyarrow directly
3. Return packed bytes directly
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from bitarray import bitarray, decodetree
from bitarray.util import vl_encode, vl_decode, huffman_code


import struct
from itertools import islice


def numpy_pack_unaligned_bits(data: np.ndarray) -> bytes:
    num_padding_bits = 8 - len(data) % 8
    return struct.pack("B", num_padding_bits) + np.packbits(data).tobytes()


def numpy_unpack_unaligned_bits(data: bytes) -> np.ndarray:
    num_padding_bits = struct.unpack("B", data[:1])[0]
    return np.unpackbits(data[1:], count=-num_padding_bits if num_padding_bits != 0 else None)


# encode the code itself
# makes sense to do this - alternative would be recomputing frequencies and applying huffman to code them
def encode_code(code):
    res = bytearray(struct.pack("<H", len(code)))
    for sym in sorted(code):
        res.append(sym)
        res.extend(vl_encode(code[sym]))
    return res


def decode_code(stream):
    size = struct.unpack("<H", bytes(islice(stream, 2)))[0]
    code = {}
    for _ in range(size):
        sym = next(stream)
        code[sym] = vl_decode(stream)
    return code


def unaligned_bitarray_to_bytes(data: bitarray) -> bytes:
    num_padding_bits = (8 - len(data) % 8) % 8
    return struct.pack("B", num_padding_bits) + data.tobytes()


def unaligned_bitarray_from_bytes(data: bytes) -> bitarray:
    # TODO: ideally we'd check for some identifier in the first byte
    num_padding_bits = struct.unpack("B", data[:1])[0]
    a = bitarray()
    a.frombytes(data[1:])
    return a[:len(a) - num_padding_bits]


@dataclass
class BinEncoding:
    """Encoding of a continuous variable using bins."""
    bins: list[float]

    @classmethod
    def build(cls, num_bins: int, low: float, high: float):
        bins = np.linspace(low, high, num_bins + 1)
        return cls(bins=list(bins))

    def encode(self, data: np.ndarray) -> bytes:
        data = np.clip(data, self.bins[0]+0.0001, self.bins[-1]-0.0001)
        bins = np.digitize(data, self.bins) - 1  # 0 means data is less than first bin - we assume that this is impossible
        return bins

    def decode(self, data: np.ndarray) -> np.ndarray:
        return np.array(self.bins)[data + 1]  # right edge of bin


@dataclass
class HuffmanEncoding:
    """Encoding of integer array using Huffman coding."""
    probs: Optional[list[float]] = None
    num_bins: Optional[int] = None  # we assume that bin indices are 0, 1, ..., num_bins - 1
    return_bool: bool = False  # if True, return bools, otherwise return bytes

    def __post_init__(self):
        assert len(self.probs) == self.num_bins
        self._code = None
        self._decodetree = None

    @staticmethod
    def compute_bin_probs(data: np.ndarray, num_bins: int) -> list[float]:
        values, counts = np.unique(data, return_counts=True)
        probs = np.zeros(num_bins)
        probs[values] = counts / counts.sum()
        return probs

    @property
    def code(self):
        if self._code is None:
            self._code = huffman_code({i: p for i, p in enumerate(self.probs)})
        return self._code

    @property
    def decodetree(self):
        if self._decodetree is None:
            self._decodetree = decodetree(self.code)
        return self._decodetree

    def encode(self, data: np.ndarray) -> bytes | np.ndarray:
        if self.num_bins is None:
            self.num_bins = data.max() + 1
        if self.probs is None:
            self.probs = self.compute_bin_probs(data, self.num_bins)

        b = bitarray()
        b.encode(self.code, data)
        if self.return_bool:
            # https://github.com/ilanschnell/bitarray/blob/master/examples/ndarray.py
            return np.frombuffer(b.unpack(), dtype=bool)  # unpack creates a byte-aligned repr: 1 byte per bit (-> no padding)
        else:
            return unaligned_bitarray_to_bytes(b)

    def decode(self, data: bytes | np.ndarray) -> np.ndarray:
        if self.return_bool:
            a = bitarray(iter(data.astype(int)))
        else:
            a = unaligned_bitarray_from_bytes(data)
        return np.fromiter(a.decode(self.code), dtype=int)  # should return a list of ints


@dataclass
class HistogramEncoding:
    """Encoding of a continuous variable using histogram-based Huffman coding."""
    bin_encoding: BinEncoding
    huffman_encoding: HuffmanEncoding
    
    def __post_init__(self):
        assert len(self.bin_encoding.bins) == len(self.huffman_encoding.probs) + 1, (
            f"Number of bins ({len(self.bin_encoding.bins)}) + 1 and number of probabilities ({len(self.huffman_encoding.probs)}) must match"
        )

    @classmethod
    def build(
        cls,
        data: np.ndarray,
        num_bins: int,
        low: Optional[float] = None,
        high: Optional[float] = None,
        return_bool: bool = False,
    ):
        # having exact bounds is awkward for digitize - we'll just add a small buffer
        range = (low, high) if low is not None and high is not None else (data.min() - 0.0001, data.max() + 0.0001)
        probs, bins = np.histogram(data, bins=num_bins, range=range, density=True)
        bin_encoding = BinEncoding(list(bins))
        huffman_encoding = HuffmanEncoding(list(probs), len(probs), return_bool=return_bool)
        return cls(bin_encoding, huffman_encoding)

    def encode(self, data: np.ndarray) -> bytes | np.ndarray:
        bin_data = self.bin_encoding.encode(data)
        return self.huffman_encoding.encode(bin_data)

    def decode(self, data: bytes | np.ndarray) -> np.ndarray:
        bin_data = self.huffman_encoding.decode(data)
        return self.bin_encoding.decode(bin_data)


@dataclass
class BitPacking:
    """Pack boolean array into int8 array using np.packbits."""
    def encode(self, data: np.ndarray) -> bytes:
        return np.packbits(data)

    def decode(self, data: bytes) -> np.ndarray:
        return np.unpackbits(data)


@dataclass
class SparseHistogramEncoding:
    """Encoding of a continuous variable using sparse histogram-based Huffman coding.

    We encode the sparsity mask and the masked values separately.
    We can use struct to pack the multiple byte strings into a single bytestring.
    """
    histogram_encoding: HistogramEncoding
    offset: float = 0.0
    zero_threshold: float = 0.01

    @classmethod
    def build(
        cls,
        data: np.ndarray,
        num_bins: int,
        low: Optional[float] = None,
        high: Optional[float] = None,
        zero_threshold: float = 0.01,
        offset: float = 0.0,
    ):
        histogram_encoding = HistogramEncoding.build(data, num_bins, low, high)
        return cls(zero_threshold=zero_threshold, histogram_encoding=histogram_encoding, offset=offset)

    def encode(self, data: np.ndarray) -> bytes:
        raise NotImplementedError("SparseHistogramEncoding is not working atm")
        data = data - self.offset
        sparsity_mask = (np.abs(data) < self.zero_threshold)
        num_padding_bits = 8 - len(sparsity_mask) % 8
        mask_bytes = np.packbits(sparsity_mask).tobytes()  # uint8
        assert num_padding_bits + len(sparsity_mask) == 8 * len(mask_bytes)
        values_bytes = self.histogram_encoding.encode(data[~sparsity_mask])
        mask_length = len(mask_bytes)
        # encode the length of the mask as a uint8 and the number of padding bits as a uint8
        length_bytes = struct.pack("IB", mask_length, num_padding_bits)
        return length_bytes + mask_bytes + values_bytes

    def decode(self, data: bytes) -> np.ndarray:
        mask_length, num_padding_bits = struct.unpack("IB", data[:5])
        mask_bytes = data[5:5 + mask_length]
        values_bytes = data[5 + mask_length:]
        sparsity_mask = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8), count=-num_padding_bits).astype(bool)
        output = np.zeros(len(sparsity_mask))
        decoded_values = self.histogram_encoding.decode(values_bytes)
        output[~sparsity_mask] = decoded_values
        return output + self.offset
