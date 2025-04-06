"""Use utils from nerfax parser to convert protein into internal coordinates.

TODO: figure out appropriate bond lengths / angles to measure with biopython
to avoid needing to use nerfax at all for structure -> internals. (we still need
nerfax for internals -> structure, although this could be a cool thing to contribute
to biotite.)
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
    lengths, angles, torsions = nerfax.parser.xyz_to_internal_coords(
        nerfax.parser.insert_zero(backbone_coords).reshape((-1, 3))
    )
    angles = angles.at[0, 0].set(1.0)  # (Dummy non-zero angle required)
    lengths = lengths.at[0, 0].set(
        lengths.at[1, 0].get()
    )  # (Dummy initial bond length)
    lengths = np.array(jax.device_get(lengths))
    angles = np.array(jax.device_get(angles))
    torsions = np.array(jax.device_get(torsions))
    return lengths, angles, torsions


def get_backbone_internals_from_atoms(structure: bs.AtomArray):
    # https://github.com/PeptoneLtd/nerfax/blob/2dd1ea019197cd0e273a8d5b920cc850c6b03460/nerfax/mpnerf_constants.py#L590
    xyz = load_backbone_coord_array(structure)
    return get_backbone_internals(xyz)


def get_full_internals(structure: bs.AtomArray):
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
    return bs.AtomArray.from_coord_array(coords)


# from bio_datasets.structure.protein import constants as protein_constants
# from biotite.structure.geometry import dihedral
# from biotite.structure.util import coord_for_atom_name_per_residue
# def get_sidechain_internals_list(structure: bs.AtomArray) -> list:
#     # one option would be to return l, 4 array of values (+ l, 4 mask)
#     residue_torsions = []
#     # more efficient approach will be to iterate over residue types - because dihedral can operate on arrays (l,3)
#     # if we implement this, we have to also implement ordering returned dihedrals in original order.
#     for res in bs.residue_iter(structure):
#         chi_atoms_list = protein_constants.chi_angles_atoms[res.res_name[0]]
#         chis = [
#             dihedral(*coord_for_atom_name_per_residue(res, atom_list))[0]
#             for atom_list in chi_atoms_list
#         ]
#         residue_torsions.append(chis)
#     return residue_torsions


# def get_sidechain_internals(structure: bs.AtomArray) -> np.ndarray:
#     res_ids, res_names = bs.get_residues(structure)
#     res_names = np.array(res_names)
#     unique_res_names = np.unique(res_names)
#     chi_array = np.zeros((len(res_names), 4))
#     chi_mask = np.zeros((len(res_names), 4), dtype=bool)
#     for res_name in unique_res_names:
#         res_mask = res_names == res_name
#         chi_ids = np.argwhere(
#             protein_constants.chi_angles_mask[
#                 protein_constants.restype_order[
#                     protein_constants.restype_3to1[res_name]
#                 ]
#             ]
#         ).flatten()
#         assert len(chi_ids) == len(protein_constants.chi_angles_atoms[res_name])
#         for chi_id, chi_atom_list in zip(
#             chi_ids, protein_constants.chi_angles_atoms[res_name]
#         ):
#             chi_array[res_mask, chi_id] = dihedral(
#                 *coord_for_atom_name_per_residue(
#                     structure[structure.res_name == res_name], chi_atom_list
#                 )
#             )
#             chi_mask[res_mask, chi_id] = True
#     return chi_array, chi_mask
