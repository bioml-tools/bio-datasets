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
from bio_datasets.structure.protein import ProteinChain, ProteinComplex
from bio_datasets.structure.protein.internal_coordinates import (
    get_backbone_internals_from_atoms,
)


def build_foldcomp_library(
    db_file, max_examples: Optional[int] = None, delta: bool = False
):
    assert os.path.exists(db_file)
    all_bond_lengths = []
    all_bond_angles = []
    all_dihedrals = []
    with foldcomp.open(db_file, decompress=True) as db:
        for (name, pdb_str) in tqdm.tqdm(itertools.islice(db, max_examples)):
            # if we opened with decompress False, we wouldn't get name
            atoms = load_structure(
                io.StringIO(pdb_str), file_type="pdb", extra_fields=["b_factor"]
            )
            bond_lengths, bond_angles, dihedrals = get_backbone_internals_from_atoms(
                atoms
            )
            assert dihedrals[0, 0] == 0.0 and bond_angles[0, 0] == 1.0
            # remove fixed values which are not informative
            dihedrals[0, 0] = np.nan
            bond_angles[0, 0] = np.nan
            if delta:
                bond_lengths = bond_lengths[1:] - bond_lengths[:-1]
                bond_angles = bond_angles[1:] - bond_angles[:-1]
                dihedrals = dihedrals[1:] - dihedrals[:-1]
            all_bond_lengths.append(bond_lengths)
            all_bond_angles.append(bond_angles)
            all_dihedrals.append(dihedrals)
    return (
        np.concatenate(all_bond_lengths, axis=0),
        np.concatenate(all_bond_angles, axis=0),
        np.concatenate(all_dihedrals, axis=0),
    )


def build_biodataset_library(
    dataset_name: str,
    max_examples: Optional[int] = None,
    delta: bool = False,
):
    dataset = load_dataset(dataset_name)
    all_bond_lengths = []
    all_bond_angles = []
    all_dihedrals = []
    for example in itertools.islice(dataset, max_examples):
        if isinstance(example["structure"], ProteinChain):
            atoms = example["structure"].atoms
        elif isinstance(example["structure"], ProteinComplex):
            atoms = example["structure"].atoms
        elif isinstance(example["structure"], bs.AtomArray):
            atoms = example["structure"]
        else:
            raise ValueError(f"Unknown structure type: {type(example['structure'])}")
        bond_lengths, bond_angles, dihedrals = get_backbone_internals_from_atoms(
            atoms, delta=delta
        )

        # [L, 3], [L, 3], [L, 3]
        all_bond_lengths.append(bond_lengths)
        all_bond_angles.append(bond_angles)
        all_dihedrals.append(dihedrals)
    return (
        np.concatenate(all_bond_lengths, axis=0),
        np.concatenate(all_bond_angles, axis=0),
        np.concatenate(all_dihedrals, axis=0),
    )


def main(args):
    if args.dataset_type == "foldcomp":
        all_bond_lengths, all_bond_angles, all_dihedrals = build_foldcomp_library(
            args.dataset_name, args.max_examples, delta=False
        )
    elif args.dataset_type == "biodataset":
        all_bond_lengths, all_bond_angles, all_dihedrals = build_biodataset_library(
            args.dataset_name, args.max_examples, delta=False
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    histogram_library = {}
    for i in range(3):
        (
            histogram_library[f"bond_lengths_{i}"],
            histogram_library[f"bond_length_edges_{i}"],
        ) = np.histogram(
            all_bond_lengths[:, i], bins=2**args.bond_length_bits, density=False
        )
        (
            histogram_library[f"bond_angles_{i}"],
            histogram_library[f"bond_angle_edges_{i}"],
        ) = np.histogram(
            all_bond_angles[:, i][~np.isnan(all_bond_angles[:, i])],
            bins=2**args.bond_angle_bits,
            density=False,
        )
        (
            histogram_library[f"dihedrals_{i}"],
            histogram_library[f"dihedral_edges_{i}"],
        ) = np.histogram(
            all_dihedrals[:, i][~np.isnan(all_dihedrals[:, i])],
            bins=2**args.dihedral_bits,
            range=(-np.pi, np.pi),
            density=False,
        )

        # TODO: write a test to make sure I understand how to invert this stuff.
        # bins range from 1 to len(bins) - 1 inclusive - i.e. 1 to 2**num_bits inclusive
        bond_length_bins = np.digitize(
            all_bond_lengths[:, i], histogram_library[f"bond_length_edges_{i}"]
        )  # 0 gets assigned to values before the left edge
        bond_length_deltas = bond_length_bins[1:] - bond_length_bins[:-1]
        bond_angle_bins = np.digitize(
            all_bond_angles[:, i], histogram_library[f"bond_angle_edges_{i}"]
        )
        bond_angle_deltas = bond_angle_bins[1:] - bond_angle_bins[:-1]
        dihedral_bins = np.digitize(
            all_dihedrals[:, i], histogram_library[f"dihedral_edges_{i}"]
        )
        dihedral_deltas = dihedral_bins[1:] - dihedral_bins[:-1]
        # bincount requires non-negative integers, so we add the max negative delta to all deltas.
        # i.e.. bin i = 1 , bin i + 1 = 1024. Are we safe excluding 0 / right including 1024? yes see comment above.
        bond_length_delta_counts = np.bincount(
            bond_length_deltas + (2**args.bond_length_bits - 1),
            minlength=2**args.bond_length_bits * 2 - 1,
        )  # times two for negative, -1 for zero
        bond_angle_delta_counts = np.bincount(
            bond_angle_deltas + (2**args.bond_angle_bits - 1),
            minlength=2**args.bond_angle_bits * 2 - 1,
        )  # times two for negative, -1 for zero
        dihedral_delta_counts = np.bincount(
            dihedral_deltas + (2**args.dihedral_bits - 1),
            minlength=2**args.dihedral_bits * 2 - 1,
        )  # times two for negative, -1 for zero
        histogram_library[f"bond_length_deltas_{i}"] = bond_length_delta_counts
        # TODO: check exactly how edges are defined (what are start and end.)
        # bins defines a monotonically increasing array of bin edges
        # including the rightmost edge [and leftmost presumably].
        # leftmost edge needs to be -2**args.bond_length_bits-1
        # rightmost edge needs to be 2**args.bond_length_bits - 1
        histogram_library["bond_length_delta_edges"] = np.arange(
            -(2**args.bond_length_bits) - 1, 2**args.bond_length_bits
        )
        histogram_library[f"bond_angle_deltas_{i}"] = bond_angle_delta_counts
        histogram_library["bond_angle_delta_edges"] = np.arange(
            -(2**args.bond_angle_bits) - 1, 2**args.bond_angle_bits
        )
        histogram_library[f"dihedral_deltas_{i}"] = dihedral_delta_counts
        histogram_library["dihedral_delta_edges"] = np.arange(
            -(2**args.dihedral_bits) - 1, 2**args.dihedral_bits
        )

    histogram_library["ramachandran_counts"] = np.histogram2d(
        all_dihedrals[:, 0],
        all_dihedrals[:, 2],
        bins=[
            histogram_library["dihedral_edges_0"],
            histogram_library["dihedral_edges_2"],
        ],
    )[0]
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

    args = parser.parse_args()
    main(args)
