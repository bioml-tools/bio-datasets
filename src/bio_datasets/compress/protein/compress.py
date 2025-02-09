from dataclasses import dataclass
import msgpack
import nerfax
import numpy as np
from biotite.structure.io.pdbx import encoding

from bio_datasets.compress.encoding import HistogramEncoding, SparseHistogramEncoding
from bio_datasets.structure.protein.constants import BACKBONE_BOND_LENGTHS
from bio_datasets.structure.protein.internal_coordinates import get_backbone_internals


@dataclass
class CompressorConfig:
    pass


class Compressor:
    def __init__(self, config: CompressorConfig):
        self.config = config

    def compress(self, data: np.ndarray) -> bytes:
        raise NotImplementedError()

    def decompress(self, data: bytes) -> np.ndarray:
        raise NotImplementedError()


@dataclass
class BackboneCompressorConfig(CompressorConfig):
    pass


class ProteinBackboneCompressor(Compressor):
    """Compressor for N, CA, C coordinates via discretised internal coordinates.

    We can pass an argument to decompress to control whether we return the internals or the atoms.
    TODO: option to return the bins...
    """

    def __init__(
        self,
        bond_length_encoders: list[list[encoding.Encoding]],
        bond_angle_encoders: list[list[encoding.Encoding]],
        dihedral_encoders: list[list[encoding.Encoding]],
    ):
        assert (
            len(bond_length_encoders)
            == len(bond_angle_encoders)
            == len(dihedral_encoders)
            == 3
        )
        self.bond_length_encoders = bond_length_encoders
        self.bond_angle_encoders = bond_angle_encoders
        self.dihedral_encoders = dihedral_encoders

    @classmethod
    def deserialize(cls, serialized_encoders: list[dict]):
        """Similar to biotite.structure.io.pdbx.BinaryCIFData.deserialize"""
        bond_length_encoders = [
            encoding.deserialize_encoding(enc)
            for enc in serialized_encoders["bond_length"]
        ]
        bond_angle_encoders = [
            encoding.deserialize_encoding(enc)
            for enc in serialized_encoders["bond_angle"]
        ]
        dihedral_encoders = [
            encoding.deserialize_encoding(enc)
            for enc in serialized_encoders["dihedral"]
        ]
        return cls(bond_length_encoders, bond_angle_encoders, dihedral_encoders)

    def serialize(self) -> dict[str, list[dict]]:
        return {
            "bond_length": [enc.serialize() for enc in self.bond_length_encoders],
            "bond_angle": [enc.serialize() for enc in self.bond_angle_encoders],
            "dihedral": [enc.serialize() for enc in self.dihedral_encoders],
        }

    def compress_internals(
        self, bond_lengths: np.ndarray, bond_angles: np.ndarray, dihedrals: np.ndarray
    ) -> bytes:
        encoded = {}
        for i in range(3):
            encoded[f"B{i}"] = self.bond_length_encoders[i].encode(bond_lengths[:, i])
            encoded[f"A{i}"] = self.bond_angle_encoders[i].encode(bond_angles[:, i])
            encoded[f"D{i}"] = self.dihedral_encoders[i].encode(dihedrals[:, i])
        return msgpack.packb(encoded, use_bin_type=True)

    def compress_coords(self, data: np.ndarray) -> bytes:
        """Compress an array of N, CA, C coordinates into a byte string."""
        internals = get_backbone_internals(data)
        bond_lengths, bond_angles, dihedrals = internals
        return self.compress_internals(bond_lengths, bond_angles, dihedrals)

    def decompress_internals(
        self, data: bytes
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompress a byte string into internals."""
        decoded = msgpack.unpackb(data, raw=False)
        bond_lengths = np.stack(
            [self.bond_length_encoders[i].decode(decoded[f"B{i}"]) for i in range(3)],
            axis=-1,
        )
        bond_angles = np.stack(
            [self.bond_angle_encoders[i].decode(decoded[f"A{i}"]) for i in range(3)],
            axis=-1,
        )
        dihedrals = np.stack(
            [self.dihedral_encoders[i].decode(decoded[f"D{i}"]) for i in range(3)],
            axis=-1,
        )
        return bond_lengths, bond_angles, dihedrals

    def decompress_coords(self, data: bytes) -> np.ndarray:
        """Decompress a byte string into an array of N, CA, C coordinates."""
        bond_lengths, bond_angles, dihedrals = self.decompress_internals(data)
        xyz_reconstructed = nerfax.reconstruct.reconstruct_from_internal_coordinates(
            bond_lengths,
            bond_angles,
            dihedrals,
            mode="fully_sequential",  # i think this mode is cpu-optimised
        )
        # TODO: is this a jax or numpy array?
        # N, CA, C
        return xyz_reconstructed


def load_bb_histogram_compressor(
    path_to_lib,
    sparse_bond_lengths: bool = False,
    sparse_bond_angles: bool = False,
    encode_deltas: bool = False,
    zero_threshold: float = 0.0001
):
    """Library should be a dict of arrays saved in npz format, e.g. the output of scripts/build_foldcomp_histogram_library.py."""
    lib = np.load(path_to_lib)
    if sparse_bond_lengths:
        bond_length_encoders = [
            SparseHistogramEncoding.from_library(
                lib[f"bond_lengths_{i}"], lib[f"bond_length_edges_{i}"], zero_threshold=zero_threshold, offset=BACKBONE_BOND_LENGTHS[i]
            )
            for i in range(3)
        ]
    else:
        bond_length_encoders = [
            HistogramEncoding.from_library(
                lib[f"bond_lengths_{i}"], lib[f"bond_length_edges_{i}"]
            )
            for i in range(3)
        ]
    bond_angle_encoders = [
        HistogramEncoding.from_library(
            lib[f"bond_angles_{i}"], lib[f"bond_angle_edges_{i}"]
        )
        for i in range(3)
    ]
    dihedral_encoders = [
        HistogramEncoding.from_library(
            lib[f"dihedrals_{i}"], lib[f"dihedral_edges_{i}"]
        )
        for i in range(3)
    ]
    return ProteinBackboneCompressor(
        bond_length_encoders=bond_length_encoders,
        bond_angle_encoders=bond_angle_encoders,
        dihedral_encoders=dihedral_encoders,
    )


# class ProteinStructureCompressor:
#     def __init__(
#         self,
#         backbone_compressor: ProteinBackboneCompressor,
#         sidechain_compressor: ProteinSidechainCompressor,
#     ):
#         self.backbone_compressor = backbone_compressor
#         self.sidechain_compressor = sidechain_compressor

#     def compress(self, data: bs.AtomArray | str | bytes) -> bytes:
#         """Compress a protein structure into a byte string.

#         Input can be an AtomArray, or a string / byte string of PDB file contents.
#         """
#         if isinstance(data, str):
#             data = io.StringIO(data)
#             atoms = load_structure(data)
#         elif isinstance(data, bytes):
#             data = io.BytesIO(data)
#             atoms = load_structure(data)
#         elif isinstance(data, bs.AtomArray):
#             atoms = data
#         else:
#             raise ValueError(f"Invalid input type: {type(data)}")

#         n_atoms = atoms[atoms.atom_name == "N"]
#         ca_atoms = atoms[atoms.atom_name == "CA"]
#         c_atoms = atoms[atoms.atom_name == "C"]
#         bb_coords = np.stack([n_atoms.coord, ca_atoms.coord, c_atoms.coord], axis=1)
#         bb_compressed = self.backbone_compressor.compress(bb_coords)
#         # sidechains = self.sidechain_compressor.compress(atoms)  How? atom14 perhaps?
#         return msgpack.packb(
#             {
#                 "bb": bb_compressed,
#                 "sc": sidechains,
#             },
#             use_bin_type=True,
#         )

#     def decompress(self, data: bytes) -> bs.AtomArray:
#         decompressed = msgpack.unpackb(data, raw=False)
#         bb_compressed = decompressed["bb"]
#         sc_compressed = decompressed["sc"]
#         bb_coords = self.backbone_compressor.decompress(bb_compressed)
#         sc_atoms = self.sidechain_compressor.decompress(sc_compressed)
#         return np.concatenate([bb_atoms, sc_atoms])


@dataclass
class NullEncoding:
    def encode(self, arr):
        return arr

    def decode(self, arr):
        return arr
