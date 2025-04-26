"""Use utils from nerfax parser to convert protein into internal coordinates.

for calculation of backbone torsions, see ramachandran plot example
phi, psi, omega = struc.dihedral_backbone(chain)

n.b. backbone dihedrals involve previous residue atoms
"""
import jax
import nerfax
import numpy as np
from biotite import structure as bs

from bio_datasets.structure.protein import ProteinChain, ProteinDictionary
from bio_datasets.structure.protein.utils import load_backbone_coord_array
from nerfax import parser as nerfax_parser
from nerfax.plugin import protein_fold


def get_backbone_internals(backbone_coords: np.ndarray):
    """Get backbone internals for a protein structure.

    Args:
        backbone_coords: (l, 3, 3) [N, CA, C]
    Returns:
        bond_lengths: (l, 3)
        angles: (l, 3)
        dihedrals: (l, 3)
    """
    lengths, angles, dihedrals = nerfax.parser.xyz_to_internal_coords(
        nerfax.parser.insert_zero(backbone_coords).reshape((-1, 3))
    )
    angles = angles.at[0, 0].set(1.0)  # (Dummy non-zero angle required)
    lengths = lengths.at[0, 0].set(
        lengths.at[1, 0].get()
    )  # (Dummy initial bond length)
    lengths = np.array(jax.device_get(lengths))
    angles = np.array(jax.device_get(angles))
    dihedrals = np.array(jax.device_get(dihedrals))
    return lengths, angles, dihedrals


def get_backbone_internals_from_atoms(structure: bs.AtomArray):
    # https://github.com/PeptoneLtd/nerfax/blob/2dd1ea019197cd0e273a8d5b920cc850c6b03460/nerfax/mpnerf_constants.py#L590
    xyz = load_backbone_coord_array(structure)
    return get_backbone_internals(xyz)


def get_full_internals(structure: bs.AtomArray):
    """Get full internals for a protein structure.

    The first 3 columns are the backbone bond lengths, angles, and dihedrals.
    The remaining 11 columns are the sidechain bond lengths, angles, and dihedrals.

    Returns:
        bond_lengths: (l, 14)
        angles: (l, 14)
        dihedrals: (l, 14)
    """
    sc_dict = ProteinDictionary.from_preset("sidechainnet")
    prot = ProteinChain(structure, residue_dictionary=sc_dict)
    coords = prot.reduced_atom_coords()
    point_ref, _ = nerfax_parser.get_point_ref_and_cloud_mask(prot.sequence)
    bond_lengths, angles_mask = nerfax_parser.get_data_masks(coords, point_ref)  # bond mask is lengths
    angles, dihedrals = angles_mask
    return np.array(jax.device_get(bond_lengths)), np.array(jax.device_get(angles)), np.array(jax.device_get(dihedrals))


def reconstruct_atoms_from_full_internals(bond_lengths, angles, dihedrals, sequence):
    point_ref, cloud_mask = nerfax_parser.get_point_ref_and_cloud_mask(sequence)
    angles_mask = np.stack([angles, dihedrals])
    coords = protein_fold(cloud_mask, point_ref[:3], angles_mask, bond_lengths)
    return ProteinChain.from_reduced_atom_coords(coords, sequence, ProteinDictionary.from_preset("sidechainnet"))


def reference_sidechain_values():
    """
    Get reference sidechain values for a protein structure.

    Returns:
        bond_lengths: (l, 11)
        angles: (l, 11)
    """
    raise NotImplementedError()


# TODO: implement converter between full internals and torsions to be compressed.
# ## Pull out the lengths, angles and dihedrals
# this is going to be something like the relevant reference coordinates for placement of each individual atom.
# ref_coords = vmap(lambda x,y: x[y], in_axes=(0,1))(coords, point_ref_mod).swapaxes(1,2)
# data_sc = vmap(decompose_quad)(ref_coords) # lengths, angles, dihedrals
