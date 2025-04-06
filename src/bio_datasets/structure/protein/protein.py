"""Defines protein objects that are lightweight wrappers around Biotite's AtomArray and AtomArrayStack.

This library is not intended to be a general-purpose library for protein structure analysis.
We simply wrap Biotite's AtomArray and AtomArrayStack to offer a few convenience methods
for dealing with protein structures in an ML context; specifically exposing residue-level properties,
including coordinates and distances.
"""
import copy
from dataclasses import dataclass
from typing import List, Optional, Union

import biotite.structure as bs
from biotite.structure.residues import get_residue_starts
import numpy as np

from bio_datasets.structure.biomolecule import BaseBiomoleculeComplex, BiomoleculeChain
from bio_datasets.structure.protein.constants import af2 as af2_constants
from bio_datasets.structure.protein.constants import scnet as scnet_constants
from bio_datasets.structure.residue import (
    ResidueDictionary,
    create_single_chain_atom_array_from_restype_index,
    get_all_residue_names,
    get_residue_starts_mask,
    register_preset_res_dict,
)

register_preset_res_dict(
    "protein",
    residue_names=copy.deepcopy(af2_constants.resnames),
    atom_types=copy.deepcopy(af2_constants.atom_types[:-1]),  # remove OXT
    backbone_atoms=["N", "CA", "C", "O"],
    unknown_residue_name="UNK",
    conversions=[
        {
            "residue": "MSE",
            "to_residue": "MET",
            "atom_swaps": [("SE", "SD")],
            "element_swaps": [("SE", "S")],
        },
        {
            "residue": "SEC",
            "to_residue": "CYS",
            "atom_swaps": [("SE", "SG")],
            "element_swaps": [("SE", "S")],
        },
    ],
)


# all residues
register_preset_res_dict(
    "protein_all",
    residue_names=get_all_residue_names("protein"),
    backbone_atoms=["N", "CA", "C", "O"],
    unknown_residue_name="UNK",
)


# TODO: decide whether UNK handling is satisfactory.
register_preset_res_dict(
    "alphafold",
    residue_names=af2_constants.resnames,
    residue_atoms={**af2_constants.residue_atoms, "UNK": ["N", "CA", "C", "O"]},
    atom_types=af2_constants.atom_types,
    backbone_atoms=["N", "CA", "C", "O"],
    unknown_residue_name="UNK",
)

# TODO: check this is consistent with more recent sidechainnet api, which doesn't use SC_BUILD_INFO
register_preset_res_dict(
    "sidechainnet",
    residue_names=list(sorted(scnet_constants.SC_BUILD_INFO.keys())) + ["UNK"],
    residue_atoms={
        **{k: ["N", "CA", "C", "O"] + v["atom-names"] for k, v in scnet_constants.SC_BUILD_INFO.items()},
        "UNK": ["N", "CA", "C", "O"],
    },
    # atom_types=... only required if we want to provide a custom atom37 type ordering (default is alphabetical)
    backbone_atoms=["N", "CA", "C", "O"],
    unknown_residue_name="UNK",
)


@dataclass
class ProteinDictionary(ResidueDictionary):
    """Defaults configure a dictionary with just the 20 standard amino acids.

    Supports OXT atoms, which generic residue dictionary does not currently.
    """

    keep_oxt: bool = False

    def _check_atom14_compatible(self):
        return all(len(res_ats) <= 14 for res_ats in self.residue_atoms.values())

    def _check_atom37_compatible(self):
        assert self.atom_types is not None
        return all(
            at in af2_constants.atom_types
            for res_ats in self.residue_atoms.values()
            for at in res_ats
        )

    def __post_init__(self):
        self._atom37_compatible = self._check_atom37_compatible()
        self._atom14_compatible = self._check_atom14_compatible()
        return super().__post_init__()

    @property
    def atom37_compatible(self):
        return self._atom37_compatible

    @property
    def atom14_compatible(self):
        return self._atom14_compatible

    def get_residue_sizes(self, restype_index, chain_id: Union[str, np.ndarray]):
        # should only be called with single chain
        if isinstance(chain_id, np.ndarray):
            assert len(np.unique(chain_id)) == 1
        residue_sizes = self.residue_sizes[restype_index].copy()
        if self.keep_oxt:
            residue_sizes[-1] += 1  # add oxt
        return residue_sizes

    def get_expected_relative_atom_indices(self, restype_index, atomtype_index):
        assert self.atom_types is not None
        if self.keep_oxt:
            expected_relative_atom_indices = np.zeros(restype_index.shape[0]).astype(
                int
            )
            oxt_id = self.atom_types.index("OXT")
            oxt_mask = atomtype_index == oxt_id
            residues_with_oxt_sizes = self.residue_sizes[restype_index[oxt_mask]]
            expected_relative_atom_indices[
                ~oxt_mask
            ] = super().get_expected_relative_atom_indices(
                restype_index[~oxt_mask], atomtype_index[~oxt_mask]
            )
            expected_relative_atom_indices[oxt_mask] = residues_with_oxt_sizes
            return expected_relative_atom_indices
        else:
            return super().get_expected_relative_atom_indices(
                restype_index, atomtype_index
            )

    def get_atom_names(
        self,
        restype_index: np.ndarray,
        relative_atom_index: np.ndarray,
        chain_id: np.ndarray,
    ):
        if self.keep_oxt:
            assert len(np.unique(chain_id)) == 1
            final_residue_mask = restype_index == restype_index[-1]
            oxt_mask = final_residue_mask & (
                relative_atom_index == self.residue_sizes[restype_index]
            )
            atom_names = np.full((len(restype_index)), "", dtype="U6")
            atom_names[~oxt_mask] = self.standard_atoms_by_residue()[
                restype_index[~oxt_mask],
                relative_atom_index[~oxt_mask],
            ]
            atom_names[oxt_mask] = "OXT"
            return atom_names
        else:
            return super().get_atom_names(restype_index, relative_atom_index, chain_id)

    def get_elements(self, restype_index, relative_atom_index, chain_id):
        if self.keep_oxt:
            assert len(np.unique(chain_id)) == 1
            final_residue_mask = restype_index == restype_index[-1]
            oxt_mask = final_residue_mask & (
                relative_atom_index == self.residue_sizes[restype_index]
            )
            elements = np.full((len(restype_index)), "", dtype="U6")
            elements[~oxt_mask] = self.standard_elements_by_residue()[
                restype_index[~oxt_mask],
                relative_atom_index[~oxt_mask],
            ]
            elements[oxt_mask] = "O"
            return elements
        else:
            return super().get_elements(restype_index, relative_atom_index, chain_id)


def filter_backbone(array, residue_dictionary):
    """
    Filter all peptide backbone atoms of one array.

    N, CA, C and O

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where an atom
        is a part of the peptide backbone.
    """

    return np.isin(array.atom_name, residue_dictionary.backbone_atoms) & np.isin(
        array.res_name, residue_dictionary.residue_names
    )


def set_annotation_at_masked_atoms(
    atoms: bs.AtomArray, annot_name: str, new_annot: np.ndarray
):
    assert "mask" in atoms._annot
    atoms.add_annotation(annot_name, dtype=new_annot.dtype)
    if len(new_annot) != len(atoms):
        assert len(new_annot) == np.sum(atoms.mask)
        getattr(atoms, annot_name)[atoms.mask] = new_annot
    else:
        getattr(atoms, annot_name)[atoms.mask] = new_annot[atoms.mask]


# TODO: add support for batched application of these functions (i.e. to multiple proteins at once)
class ProteinMixin:
    def to_complex(self):
        return ProteinComplex.from_atoms(self.atoms)

    def beta_carbon_coords(self) -> np.ndarray:
        # TODO: check for nan cb or missing cb
        has_beta_carbon = self.atoms.res_name != "GLY"
        beta_carbon_coords = np.zeros((self.num_residues, 3), dtype=np.float32)
        beta_carbon_coords[has_beta_carbon[self._residue_starts]] = self.atoms.coord[
            self._residue_starts[has_beta_carbon] + 4
        ]
        beta_carbon_coords[~has_beta_carbon[self._residue_starts]] = self.atoms.coord[
            self._residue_starts[~has_beta_carbon] + 1
        ]  # ca for gly
        return beta_carbon_coords

    def backbone_coords(self, atom_names: Optional[List[str]] = None) -> np.ndarray:
        assert all(
            atom in self.residue_dictionary.backbone_atoms + ["CB"]
            for atom in atom_names
        ), f"Invalid entries in atom names: {atom_names}"
        coords = super().backbone_coords([at for at in atom_names if at != "CB"])
        if "CB" in atom_names:
            cb_index = atom_names.index("CB")
            coords_with_cb = np.zeros(
                (len(coords), len(atom_names), 3), dtype=np.float32
            )
            coords_with_cb[:, cb_index] = self.beta_carbon_coords()
            non_cb_indices = [atom_names.index(at) for at in atom_names if at != "CB"]
            coords_with_cb[:, non_cb_indices] = coords
            return coords_with_cb
        return coords

    def contacts(self, atom_name: str = "CA", threshold: float = 8.0) -> np.ndarray:
        return super().contacts(atom_name=atom_name, threshold=threshold)

    def reduced_atom_coords(self) -> np.ndarray:
        """Reduced atom coordinate representation: each residue is represented by max(atoms_per_residue) atoms.

        max(atoms_per_residue) is determined by the dictionary. e.g. for the alphafold preset dictionary,
        we have at most 14 atoms per residue, and reduced_atom_coords corresponds to `atom14`
        Returns:
            np.ndarray: (num_residues, max(atoms_per_residue), 3)
        """
        num_atoms_per_residue = max(self.residue_dictionary.residue_sizes)
        atom_coords = np.full(
            (self.num_residues, num_atoms_per_residue, 3), np.nan
        )
        atom_index = self.residue_dictionary.atomtype_index_full_to_reduced()[
            self.atoms.restype_index, self.atoms.atomtype_index
        ]
        atom_coords[self.atoms.res_index, atom_index] = self.atoms.coord
        return atom_coords

    def full_atom_coords(self) -> np.ndarray:
        """Full atom coordinate representation: each residue is represented by all atoms in the dictionary.

        For the alphafold preset dictionary, this corresponds to `atom37`
        Returns:
            np.ndarray: (num_residues, num_atom_types_in_dictionary, 3)
        """
        full_atom_coords = np.full(
            (self.num_residues, len(self.atom_types), 3), np.nan
        )
        full_atom_coords[
            self.atoms.res_index, self.atoms.atomtype_index
        ] = self.atoms.coord
        return full_atom_coords


class ProteinChain(ProteinMixin, BiomoleculeChain):

    """A single protein chain."""

    def __init__(
        self,
        atoms: bs.AtomArray,
        residue_dictionary: ResidueDictionary,
        verbose: bool = False,
        backbone_only: bool = False,
        keep_hydrogens: bool = False,
        replace_unexpected_with_unknown: bool = False,
        raise_error_on_unexpected: bool = False,
    ):
        assert residue_dictionary is not None
        super().__init__(
            atoms,
            residue_dictionary=residue_dictionary,
            verbose=verbose,
            backbone_only=backbone_only,
            keep_hydrogens=keep_hydrogens,
            replace_unexpected_with_unknown=replace_unexpected_with_unknown,
            raise_error_on_unexpected=raise_error_on_unexpected,
        )

    @classmethod
    def from_reduced_atom_coords(cls, reduced_atom_coords, sequence, residue_dictionary):
        restype_index = residue_dictionary.sequence_to_restype_index(sequence)
        new_atom_array = create_single_chain_atom_array_from_restype_index(
            restype_index, residue_dictionary=residue_dictionary, chain_id="A"
        )
        residue_starts = get_residue_starts(new_atom_array)
        residue_index = (
            np.cumsum(get_residue_starts_mask(new_atom_array, residue_starts)) - 1
        )
        relative_atom_index = np.arange(len(new_atom_array)) - residue_starts[residue_index]
        indices = np.stack([residue_index, relative_atom_index], axis=-1)
        new_atom_array.coord = reduced_atom_coords[indices]
        return cls(new_atom_array, residue_dictionary)

class ProteinComplex(ProteinMixin, BaseBiomoleculeComplex):
    """A protein complex."""

    def __init__(self, chains: List[ProteinChain]):
        super().__init__(chains)
        self.residue_dictionary = chains[0].residue_dictionary

    @staticmethod
    def default_residue_dictionary():
        return ProteinDictionary.from_preset("protein", keep_oxt=False)
