"""For bond lengths and bond angles, we infer the range of allowed values from the dataset.

For dihedrals, we use -pi to pi as the allowable range.
"""

import argparse
import io
import itertools
import os
from typing import Optional

import biotite.structure as bs
import foldcomp
import numpy as np
import tqdm

from bio_datasets import load_dataset
from bio_datasets.structure.parsing import load_structure
from bio_datasets.structure.protein import ProteinChain, ProteinComplex, ProteinDictionary
from bio_datasets.structure.protein.internal_coordinates import (
    get_backbone_internals_from_atoms, get_full_internals
)
from bio_datasets.structure.protein.constants import af2 as af2_constants


class InternalCoordinateLibrary:
    """Store statistics for the distribution of bond lengths, bond angles, and dihedrals.
    
    For both backbone and sidechain we store histograms (and means of bond lengths and angles).
    For sidechain downstream applications it's safe to use the mean values.
    """
    def __init__(self, bond_length_bits: int, bond_angle_bits: int, dihedral_bits: int, sidechain_torsion_bits: int):
        self.bond_length_bits = bond_length_bits
        self.bond_angle_bits = bond_angle_bits
        self.dihedral_bits = dihedral_bits
        self.sidechain_torsion_bits = sidechain_torsion_bits
        self.all_bond_lengths = []  # backbone bond lengths [(l1, 3), (l2, 3), ...]
        self.all_bond_angles = []  # backbone bond angles [(l1, 3), (l2, 3), ...]
        self.all_dihedrals = []  # backbone dihedrals [(l1, 3), (l2, 3), ...]
        self.all_sc_bond_lengths = {aa: [] for aa in af2_constants.restypes}  # sidechain bond lengths [(l1, 14), (l2, 14), ...]
        self.all_sc_bond_angles = {aa: [] for aa in af2_constants.restypes}  # sidechain bond angles [(l1, 14), (l2, 14), ...]
        self.all_sc_dihedrals = {aa: [] for aa in af2_constants.restypes}  # sidechain dihedrals [(l1, 14), (l2, 14), ...]
        self.sc_dict = ProteinDictionary.from_preset("sidechainnet")

    def add_example(self, atoms: bs.AtomArray, delta: bool = False):
        bond_lengths, bond_angles, dihedrals = get_full_internals(atoms)
        bb_bond_lengths, bb_bond_angles, bb_dihedrals = bond_lengths[:, :3], bond_angles[:, :3], dihedrals[:, :3]
        # Each L, 11. The length is the length between the atom and its parent,
        # the angle the angle between the atom's grandparent, parent and itself,
        # and the dihedral the angle between the great grandparent, grand parent, parent and self.
        # nerfax.mpnerf_constants.make_idx_mask(aa_letter) gives the indices off the great grandparent, grandparent and parent atoms
        sc_bond_lengths, sc_bond_angles, sc_dihedrals = bond_lengths[:, 3:], bond_angles[:, 3:], dihedrals[:, 3:]
        assert bb_dihedrals[0, 0] == 0.0 and bb_bond_angles[0, 0] == 1.0
        # remove fixed values which are not informative
        bb_dihedrals[0, 0] = np.nan
        bb_bond_angles[0, 0] = np.nan
        if delta:
            bb_bond_lengths = bb_bond_lengths[1:] - bb_bond_lengths[:-1]
            bb_bond_angles = bb_bond_angles[1:] - bb_bond_angles[:-1]
            bb_dihedrals = bb_dihedrals[1:] - bb_dihedrals[:-1]

        self.all_bond_lengths.append(bb_bond_lengths)
        self.all_bond_angles.append(bb_bond_angles)
        self.all_dihedrals.append(bb_dihedrals)
        for residue_code, (sc_bond_lengths, sc_bond_angles, sc_dihedrals) in zip(bs.get_residues(atoms)[1], zip(sc_bond_lengths, sc_bond_angles, sc_dihedrals)):
            residue_letter = af2_constants.restype_3to1.get(residue_code)
            if residue_letter is None:
                continue
            residue_atoms = self.sc_dict.residue_atoms[residue_code]
            atoms_mask = np.zeros(11, dtype=bool)
            atoms_mask[len(residue_atoms) - 3] = True  # only consider atoms other than N,CA,C (atoms in sc_bond_lengths,etc.)

            # for positions not corresponding to atoms for the given residue, the values are all null so are safely ignored
            assert np.all(sc_bond_lengths[~atoms_mask] == 0.0)
            assert np.all(sc_bond_angles[~atoms_mask] == 0.0)
            assert np.all(sc_dihedrals[~atoms_mask] == 0.0)
            self.all_sc_bond_lengths[residue_letter].append(sc_bond_lengths[atoms_mask])
            self.all_sc_bond_angles[residue_letter].append(sc_bond_angles[atoms_mask])
            self.all_sc_dihedrals[residue_letter].append(sc_dihedrals[atoms_mask])

    @property
    def library(self):
        return {
            "bond_lengths": np.concatenate(self.all_bond_lengths, axis=0),
            "bond_angles": np.concatenate(self.all_bond_angles, axis=0),
            "dihedrals": np.concatenate(self.all_dihedrals, axis=0),
            "sc_bond_lengths": {aa: np.concatenate(self.all_sc_bond_lengths[aa], axis=0) for aa in self.sc_dict.residue_letters},
            "sc_bond_angles": {aa: np.concatenate(self.all_sc_bond_angles[aa], axis=0) for aa in self.sc_dict.residue_letters},
            "sc_dihedrals": {aa: np.concatenate(self.all_sc_dihedrals[aa], axis=0) for aa in self.sc_dict.residue_letters},
        }

    def reference_values(self):
        return {k: np.mean(v, axis=0) for k, v in self.library.items() if "dihedrals" not in k}

    def to_histogram_library(self) -> dict:
        histogram_library = {"sidechain": {"mean_values": {}}, "backbone": {"mean_values": {}}}
        lib = self.library
        for i in range(3):
            (
                histogram_library["backbone"]["bond_lengths"][i],
                histogram_library["backbone"]["bond_length_edges"][i],
            ) = np.histogram(
                lib["bond_lengths"][:, i], bins=2**self.bond_length_bits, density=False
            )
            
            (
                histogram_library["backbone"]["bond_angles"][i],
                histogram_library["backbone"]["bond_angle_edges"][i],
            ) = np.histogram(
                lib["bond_angles"][:, i][~np.isnan(lib["bond_angles"][:, i])],
                bins=2**self.bond_angle_bits,
                density=False,
            )
            (
                histogram_library["backbone"]["dihedrals"][i],
                histogram_library["backbone"]["dihedral_edges"][i],
            ) = np.histogram(
                lib["dihedrals"][:, i][~np.isnan(lib["dihedrals"][:, i])],
                bins=2**self.dihedral_bits,
                range=(-np.pi, np.pi),
                density=False,
            )

            histogram_library["backbone"]["mean_values"][f"bond_lengths_{i}"] = np.mean(lib["bond_lengths"][:, i])
            histogram_library["backbone"]["mean_values"][f"bond_angles_{i}"] = np.mean(lib["bond_angles"][:, i][~np.isnan(lib["bond_angles"][:, i])])
            # i thought it would be interesting to store pairs, but then naive discretization
            # is too expensive. a simple conditional probabilistic model rather than a standard
            # histogram would be better - we can do this with these counts! but then the decoder
            # is less straightforward.
            histogram_library["ramachandran_counts"] = np.histogram2d(
                self.all_dihedrals[:, 0],
                self.all_dihedrals[:, 2],
                bins=[
                    histogram_library["backbone"]["dihedral_edges_0"],
                    histogram_library["backbone"]["dihedral_edges_2"],
                ],
            )[0]
        for residue_letter in self.sc_dict.residue_letters:
            for i in range(len(self.sc_dict.residue_atoms[residue_letter])-3):
                (
                    histogram_library["sidechain"][residue_letter][f"sc_bond_lengths_{i}"],
                    histogram_library["sidechain"][residue_letter][f"sc_bond_length_edges_{i}"],
                ) = np.histogram(
                    lib["sc_bond_lengths"][residue_letter][:, i], bins=2**self.bond_length_bits, density=False
                )
                (
                    histogram_library["sidechain"][residue_letter][f"sc_bond_angles_{i}"],
                    histogram_library["sidechain"][residue_letter][f"sc_bond_angle_edges_{i}"],
                ) = np.histogram(
                    lib["sc_bond_angles"][residue_letter][:, i], bins=2**self.bond_angle_bits, density=False
                )
                (
                    histogram_library["sidechain"][residue_letter][f"sc_dihedrals_{i}"],
                    histogram_library["sidechain"][residue_letter][f"sc_dihedral_edges_{i}"],
                ) = np.histogram(
                    lib["sc_dihedrals"][residue_letter][:, i], bins=2**self.dihedral_bits, density=False
                )
                histogram_library["sidechain"][residue_letter]["mean_values"][f"sc_bond_lengths_{i}"] = np.mean(lib["sc_bond_lengths"][residue_letter][:, i])
                histogram_library["sidechain"][residue_letter]["mean_values"][f"sc_bond_angles_{i}"] = np.mean(lib["sc_bond_angles"][residue_letter][:, i])

        return histogram_library


def build_foldcomp_library(
    db_file, max_examples: Optional[int] = None, delta: bool = False
):
    assert os.path.exists(db_file)
    library = InternalCoordinateLibrary()
    with foldcomp.open(db_file, decompress=True) as db:
        for (name, pdb_str) in tqdm.tqdm(itertools.islice(db, max_examples)):
            # if we opened with decompress False, we wouldn't get name
            atoms = load_structure(
                io.StringIO(pdb_str), file_type="pdb", extra_fields=["b_factor"]
            )
            library.add_example(atoms, delta=delta)
    return library


def build_biodataset_library(
    dataset_name: str,
    max_examples: Optional[int] = None,
    delta: bool = False,
):
    dataset = load_dataset(dataset_name)
    library = InternalCoordinateLibrary()
    for example in itertools.islice(dataset, max_examples):
        if isinstance(example["structure"], ProteinChain):
            atoms = example["structure"].atoms
        elif isinstance(example["structure"], ProteinComplex):
            atoms = example["structure"].atoms
        elif isinstance(example["structure"], bs.AtomArray):
            atoms = example["structure"]
        else:
            raise ValueError(f"Unknown structure type: {type(example['structure'])}")
        library.add_example(atoms, delta=delta)
    return library


def main(args):
    if args.dataset_type == "foldcomp":
        library = build_foldcomp_library(
            args.dataset_name, args.max_examples, delta=False
        )
    elif args.dataset_type == "biodataset":
        library = build_biodataset_library(
            args.dataset_name, args.max_examples, delta=False
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    histogram_library = library.to_histogram_library()
    # I guess we should save as a numpy array
    np.savez(args.output_file, **histogram_library)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument(
        "--dataset_type", choices=["foldcomp", "biodataset"], default="foldcomp"
    )
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--bond_length_bits", type=int, default=10)
    parser.add_argument("--bond_angle_bits", type=int, default=13)
    parser.add_argument("--dihedral_bits", type=int, default=14)
    parser.add_argument("--sidechain_torsion_bits", type=int, default=None)  # maybe 10 is reasonable, since there is no real issue with accumulation of errors

    args = parser.parse_args()
    main(args)
