import argparse
import io
import itertools
import os
from typing import Optional

import biotite.structure as bs
import foldcomp
import tqdm
import numpy as np
from bio_datasets import load_dataset
from bio_datasets.structure.protein import ProteinChain, ProteinComplex
from bio_datasets.structure.parsing import load_structure
from bio_datasets.structure.protein.internal_coordinates import get_backbone_internals_from_atoms


def build_foldcomp_library(
    db_file, max_examples: Optional[int] = None
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
            bond_lengths, bond_angles, dihedrals = get_backbone_internals_from_atoms(atoms)
            all_bond_lengths.append(bond_lengths)
            all_bond_angles.append(bond_angles)
            all_dihedrals.append(dihedrals)
    return all_bond_lengths, all_bond_angles, all_dihedrals


def build_biodataset_library(
    dataset_name: str,
    max_examples: Optional[int] = None,
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
        bond_lengths, bond_angles, dihedrals = get_backbone_internals_from_atoms(atoms)
        all_bond_lengths.append(bond_lengths)
        all_bond_angles.append(bond_angles)
        all_dihedrals.append(dihedrals)
    return all_bond_lengths, all_bond_angles, all_dihedrals


def main(args):
    if args.dataset_type == "foldcomp":
        all_bond_lengths, all_bond_angles, all_dihedrals = build_foldcomp_library(args.dataset_name, args.max_examples)
    elif args.dataset_type == "biodataset":
        all_bond_lengths, all_bond_angles, all_dihedrals = build_biodataset_library(args.dataset_name, args.max_examples)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    histogram_library = {}
    for i in range(3):
        histogram_library[f"bond_lengths_{i}"] = np.histogram(all_bond_lengths[i], bins=2**args.bond_length_bits)
        histogram_library[f"bond_angles_{i}"] = np.histogram(all_bond_angles[i], bins=2**args.bond_angle_bits)
        histogram_library[f"dihedrals_{i}"] = np.histogram(all_dihedrals[i], bins=2**args.dihedral_bits)
    # I guess we should save as a numpy array
    np.savez(args.output_file, **histogram_library)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--dataset_type", choices=["foldcomp", "biodataset"], default="foldcomp")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--bond_length_bits", type=int, default=10)
    parser.add_argument("--bond_angle_bits", type=int, default=13)
    parser.add_argument("--dihedral_bits", type=int, default=14)
    args = parser.parse_args()
    main(args)
