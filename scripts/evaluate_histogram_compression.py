"""For bond lengths and bond angles, we infer the range of allowed values from the dataset.

For dihedrals, we use -pi to pi as the allowable range.

Foldcomp has the advantage of not bothering with bond lengths and angles.
This allows it to get to 8 bytes per residue.

To maximise efficiency, we'll have to write some kind of parallel implementation.
"""

import argparse
import io
import itertools
import os
import sys
import numpy as np
import time
from typing import Optional

sys.setrecursionlimit(10000)

import foldcomp
import tqdm

from bio_datasets.structure.parsing import load_structure
from bio_datasets.structure.protein.utils import load_backbone_coord_array
from bio_datasets.compress.protein.compress import load_bb_histogram_compressor
from bio_datasets.compress.utils import compute_aligned_rmsd


def foldcomp_examples_generator(
    db_file, max_examples: Optional[int] = None
):
    assert os.path.exists(db_file)
    with foldcomp.open(db_file, decompress=True) as db:
        for (name, pdb_str) in tqdm.tqdm(itertools.islice(db, max_examples)):
            # if we opened with decompress False, we wouldn't get name
            atoms = load_structure(
                io.StringIO(pdb_str), file_type="pdb", extra_fields=["b_factor"]
            )
            yield atoms


def main(args):
    if args.dataset_type == "foldcomp":
        examples_generator = foldcomp_examples_generator(args.dataset_name, args.max_examples)
    elif args.dataset_type == "biodataset":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    all_sizes = []
    all_lengths = []
    all_rmsds = []
    # TODO: make sure we aren't needlessly recomputing the counts or the code or anything
    compression_times = []
    decompression_times = []
    for atoms in examples_generator:
        backbone_coords = load_backbone_coord_array(atoms)
        compressor = load_bb_histogram_compressor(args.histogram_lib)
        t0 = time.time()
        compressed = compressor.compress(backbone_coords)
        t1 = time.time()
        decompressed = compressor.decompress(compressed)
        t2 = time.time()
        all_sizes.append(len(compressed))
        all_lengths.append(backbone_coords.shape[0])
        rmsd = compute_aligned_rmsd(decompressed, backbone_coords.reshape((-1,3)))
        print(f"Bytes per residue: {len(compressed)/backbone_coords.shape[0]:.2f}, rmsd: {rmsd:.2f}")
        all_rmsds.append(rmsd)
        compression_times.append(t1-t0)
        decompression_times.append(t2-t1)
        
    total_size = sum(all_sizes)
    total_length = sum(all_lengths)
    print(f"Bytes per residue: {total_size/total_length:.2f}")
    print(f"Mean RMSD: {np.mean(all_rmsds):.2f}")
    print(f"Compression time: {np.mean(compression_times):.2f}, decompression time: {np.mean(decompression_times):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("histogram_lib", type=str, help="Path to saved histogram library to use for compression.")
    parser.add_argument("--dataset_type", choices=["foldcomp", "biodataset"], default="foldcomp")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()
    main(args)
