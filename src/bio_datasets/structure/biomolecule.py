import io
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from biotite import structure as bs
from biotite.structure.io.pdb import PDBFile
from biotite.structure.residues import get_residue_starts

from bio_datasets.np_utils import map_categories_to_indices

from .chemical.chemical import Molecule, T
from .residue import ResidueDictionary, get_residue_starts_mask

ALL_EXTRA_FIELDS = ["occupancy", "b_factor", "atom_id", "charge"]


def create_complete_atom_array_from_restype_index(
    restype_index: np.ndarray,
    residue_dictionary: ResidueDictionary,
    chain_id: Tuple[str, np.ndarray],
    extra_fields: Optional[List[str]] = None,
    backbone_only: bool = False,
):
    """
    Populate annotations from restype_index, assuming all atoms are present.

    Assumes single chain for now
    """
    if isinstance(chain_id, np.ndarray):
        unique_chain_ids = np.unique(chain_id)
        chain_atom_arrays = []
        chain_residue_starts = []
        residue_starts_offset = 0
        for chain_id in unique_chain_ids:
            (
                atom_array,
                residue_starts,
                full_annot_names,
            ) = create_complete_atom_array_from_restype_index(
                restype_index=restype_index,
                residue_dictionary=residue_dictionary,
                chain_id=chain_id,
                extra_fields=extra_fields,
                backbone_only=backbone_only,
            )
            chain_atom_arrays.append(atom_array)
            chain_residue_starts.append(residue_starts + residue_starts_offset)
            residue_starts_offset += len(atom_array)
        concatenated_array = sum(chain_atom_arrays, bs.AtomArray(length=0))
        for key in atom_array._annot.keys():
            if key not in concatenated_array._annot:
                concatenated_array.set_annotation(
                    key,
                    np.concatenate([atoms._annot[key] for atoms in chain_atom_arrays]),
                )
        return (
            concatenated_array,
            np.concatenate(chain_residue_starts),
            full_annot_names,
        )
    else:
        if backbone_only:
            residue_sizes = len(residue_dictionary.backbone_atoms) * len(restype_index)
        else:
            residue_sizes = residue_dictionary.get_residue_sizes(
                restype_index, chain_id
            )
            # (n_residues,) NOT (n_atoms,)

        residue_starts = np.concatenate(
            [[0], np.cumsum(residue_sizes)[:-1]]
        )  # (n_residues,)
        new_atom_array = bs.AtomArray(length=np.sum(residue_sizes))
        chain_id = np.full(len(new_atom_array), chain_id, dtype="U4")
        new_atom_array.set_annotation(
            "chain_id",
            chain_id,
        )
        full_annot_names = [
            "chain_id",
        ]
        residue_index = (
            np.cumsum(get_residue_starts_mask(new_atom_array, residue_starts)) - 1
        )

        relative_atom_index = (
            np.arange(len(new_atom_array)) - residue_starts[residue_index]
        )
        atom_names = new_atom_array.atom_name
        new_atom_array.set_annotation("restype_index", restype_index[residue_index])
        atom_names = residue_dictionary.get_atom_names(
            new_atom_array.restype_index, relative_atom_index, chain_id
        )
        new_atom_array.set_annotation("atom_name", atom_names)
        new_atom_array.set_annotation(
            "res_name",
            np.array(residue_dictionary.residue_names)[new_atom_array.restype_index],
        )
        new_atom_array.set_annotation("res_index", residue_index)
        new_atom_array.set_annotation("res_id", residue_index + 1)
        new_atom_array.set_annotation(
            "element", np.char.array(new_atom_array.atom_name).astype("U1")
        )
        full_annot_names += [
            "atom_name",
            "restype_index",
            "res_name",
            "res_index",
            "res_id",
        ]
        if extra_fields is not None:
            for f in extra_fields:
                new_atom_array.add_annotation(
                    f, dtype=float if f in ["occupancy", "b_factor"] else int
                )
        return new_atom_array, residue_starts, full_annot_names


class Biomolecule(Molecule):
    """Base class for biomolecule objects.

    Biomolecules (DNA, RNA and Proteins) are chains of residues.

    Biomolecule and modality-specific subclasses provide convenience
    methods for interacting with residue-level properties.

    n.b. as well as proteins, dna and rna, the PDB also contains hybrid dna/rna molecules.
    other classes of biopolymers are polysaccharides and peptidoglycans.
    """

    def __init__(
        self,
        atoms: bs.AtomArray,
        residue_dictionary: ResidueDictionary,
        verbose: bool = False,
        backbone_only: bool = False,
        raise_error_on_unexpected_residue: bool = False,
    ):
        self.residue_dictionary = residue_dictionary
        self.backbone_only = backbone_only
        self.raise_error_on_unexpected_residue = raise_error_on_unexpected_residue
        atoms = self.convert_residues(atoms, self.residue_dictionary)
        atoms = self.filter_atoms(
            atoms,
            self.residue_dictionary,
            raise_error_on_unexpected=self.raise_error_on_unexpected_residue,
        )  # e.g. check for standard residues.
        self.atoms = self.standardise_atoms(
            atoms,
            residue_dictionary=self.residue_dictionary,
            verbose=verbose,
            backbone_only=self.backbone_only,
        )
        self._standardised = True

    @property
    def is_standardised(self):
        return self._standardised

    @staticmethod
    def convert_residues(atoms: bs.AtomArray, residue_dictionary: ResidueDictionary):
        for conversion_dict in residue_dictionary.conversions or []:
            atom_swaps = conversion_dict["atom_swaps"]
            from_mask = (atoms.res_name == conversion_dict["residue"]).astype(bool)
            for swap in atom_swaps:
                atoms.atom_name[
                    from_mask & (atoms.atom_name == swap[0]).astype(bool)
                ] = swap[1]
            atoms.res_name[from_mask] = conversion_dict["to_residue"]
        return atoms

    @staticmethod
    def filter_atoms(
        atoms, residue_dictionary, raise_error_on_unexpected: bool = False
    ):
        expected_residue_mask = np.isin(
            atoms.res_name, residue_dictionary.residue_names
        )
        if raise_error_on_unexpected and ~expected_residue_mask.any():
            unexpected_residues = np.unique(atoms[~expected_residue_mask].res_name)
            raise ValueError(
                f"Found unexpected residues: {unexpected_residues} in atom array"
            )
        return atoms[expected_residue_mask]

    @staticmethod
    def standardise_atoms(
        atoms,
        residue_dictionary,
        verbose: bool = False,
        backbone_only: bool = False,
    ):
        assert (
            "element" in atoms._annot
        ), "Elements must be present to exclude hydrogens"
        atoms = atoms[~np.isin(atoms.element, ["H", "D"])]
        residue_starts = get_residue_starts(atoms)
        if "atomtype_index" not in atoms._annot:
            atoms.set_annotation(
                "atomtype_index",
                map_categories_to_indices(
                    atoms.atom_name, residue_dictionary.atom_types
                ),
            )
        if "restype_index" not in atoms._annot:
            atoms.set_annotation(
                "restype_index", residue_dictionary.resname_to_index(atoms.res_name)
            )
        atoms.set_annotation(
            "res_index",
            np.cumsum(get_residue_starts_mask(atoms, residue_starts)) - 1,
        )

        (
            new_atom_array,
            full_residue_starts,
            full_annot_names,
        ) = create_complete_atom_array_from_restype_index(
            atoms.restype_index[residue_starts],
            residue_dictionary=residue_dictionary,
            chain_id=atoms.chain_id[residue_starts],
            extra_fields=[f for f in ALL_EXTRA_FIELDS if f in atoms._annot],
        )
        # first we get an array of atom indices for each residue (i.e. a mapping from atom type index to expected index
        # then we index into this array to get the expected relative index for each atom
        expected_relative_atom_indices = (
            residue_dictionary.get_expected_relative_atom_indices(
                atoms.restype_index, atoms.atomtype_index
            )
        )

        unexpected_atom_mask = expected_relative_atom_indices == -100
        if np.any(
            unexpected_atom_mask
            & (atoms.res_name != residue_dictionary.unknown_residue_name)
        ):
            unexpected_atoms = atoms.atom_name[unexpected_atom_mask]
            unexpected_residues = atoms.res_name[unexpected_atom_mask]
            unexpected_str = "\n".join(
                [
                    f"{res_name} {res_id} {atom_name}"
                    for res_name, res_id, atom_name in zip(
                        unexpected_residues,
                        atoms.res_id[unexpected_atom_mask],
                        unexpected_atoms,
                    )
                ]
            )
            raise ValueError(
                f"At least one unexpected atom detected in a residue: {unexpected_str}.\n"
                f"HETATMs are not supported."
            )

        # for unk residues, we just drop any e.g. sidechain atoms without raising an exception
        unexpected_unk_atom_mask = unexpected_atom_mask & (
            atoms.res_name == residue_dictionary.unknown_residue_name
        )
        atoms = atoms[~unexpected_unk_atom_mask]
        expected_relative_atom_indices = expected_relative_atom_indices[
            ~unexpected_unk_atom_mask
        ]
        residue_starts = get_residue_starts(atoms)

        assert len(full_residue_starts) == len(
            residue_starts
        ), f"Full residue starts: {full_residue_starts} and residue starts: {residue_starts} do not match"

        existing_atom_indices_in_full_array = (
            full_residue_starts[atoms.res_index] + expected_relative_atom_indices
        )

        for annot_name, annot in atoms._annot.items():
            if (
                annot_name in ["atomtype_index", "mask"]
                or annot_name in full_annot_names
            ):
                continue
            getattr(new_atom_array, annot_name)[
                existing_atom_indices_in_full_array
            ] = annot.astype(new_atom_array._annot[annot_name].dtype)

        # set_annotation vs setattr: set_annotation adds to annot and verifies size
        new_atom_array.coord[existing_atom_indices_in_full_array] = atoms.coord
        # if we can create a res start index for each atom, we can assign the value based on that...

        assert (
            np.unique(new_atom_array.res_index) == np.unique(atoms.res_index)
        ).all(), "We need this to agree to use residue indexing for filling annotations"
        new_atom_array.set_annotation(
            "res_id",
            atoms.res_id[residue_starts][new_atom_array.res_index].astype(
                new_atom_array.res_id.dtype
            ),
        )  # override with auth res id
        new_atom_array.set_annotation(
            "chain_id",
            atoms.chain_id[residue_starts][new_atom_array.res_index].astype(
                new_atom_array.chain_id.dtype
            ),
        )
        new_atom_array.set_annotation(
            "ins_code",
            atoms.ins_code[residue_starts][new_atom_array.res_index].astype(
                new_atom_array.ins_code.dtype
            ),
        )

        new_atom_array.set_annotation(
            "atomtype_index",
            map_categories_to_indices(
                new_atom_array.atom_name, residue_dictionary.atom_types
            ),
        )
        assert np.all(
            new_atom_array.atom_name != ""
        ), "All atoms must be assigned a name"
        mask = np.zeros(len(new_atom_array), dtype=bool)
        mask[existing_atom_indices_in_full_array] = True
        missing_atoms_strings = [
            f"{res_name} {res_id} {atom_name}"
            for res_name, res_id, atom_name in zip(
                new_atom_array.res_name[~mask],
                new_atom_array.res_id[~mask],
                new_atom_array.atom_name[~mask],
            )
        ]
        if verbose:
            print("Filled in missing atoms:\n", "\n".join(missing_atoms_strings))
        new_atom_array.set_annotation("mask", mask)
        if backbone_only:
            # TODO: more efficient backbone only
            new_atom_array = new_atom_array[
                np.isin(new_atom_array.atom_name, residue_dictionary.backbone_atomss)
            ]
            full_residue_starts = get_residue_starts(new_atom_array)
        return new_atom_array

    @classmethod
    def from_pdb(cls, pdb_path: str):
        pdbf = PDBFile.read(pdb_path)
        atoms = pdbf.get_structure()
        return cls(atoms)

    def to_pdb(self, pdb_path: str):
        # to write to pdb file, we have to drop nan coords
        atoms = self.atoms[~self.nan_mask]
        pdbf = PDBFile()
        pdbf.set_structure(atoms)
        pdbf.write(pdb_path)

    def to_pdb_string(self):
        with io.StringIO() as f:
            self.to_pdb(f)
            return f.getvalue()

    @property
    def nan_mask(self):
        return np.isnan(self.atoms.coord).any(axis=-1)

    @property
    def residue_index(self):
        return self.atoms["residue_index"][self._residue_starts]

    @property
    def restype_index(self):
        # TODO: parameterise this via a name e.g. 'aa'
        return self.atoms["restype_index"][self._residue_starts]

    @property
    def sequence(self) -> str:
        return "".join(
            self.residue_dictionary.residue_types[
                self.atoms.restype_index[self._residue_starts]
            ]
        )

    @property
    def num_residues(self) -> int:
        return len(self._residue_starts)

    @property
    def backbone_mask(self):
        return np.isin(self.atoms.atom_name, self.backbone_atoms)

    def __len__(self):
        return self.num_residues  # n.b. -- not equal to len(self.atoms)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.atoms[key]
        else:
            return self.__class__(self.atoms[key])

    def backbone_coords(self, atom_names: Optional[List[str]] = None) -> np.ndarray:
        assert all(
            [atom in self.backbone_atoms for atom in atom_names]
        ), f"Invalid entries in atom names: {atom_names}"
        assert self._standardised, "Atoms must be in standard order"
        backbone_coords = self.atoms.coord[self.backbone_mask].reshape(
            -1, len(self.backbone_atoms), 3
        )
        if atom_names is None:
            return backbone_coords
        else:
            backbone_atom_indices = [
                self.backbone_atoms.index(atom) for atom in atom_names if atom != "CB"
            ]
            selected_coords = np.zeros(
                (len(backbone_coords), len(atom_names), 3), dtype=np.float32
            )
            selected_backbone_indices = [
                atom_names.index(atom) for atom in atom_names if atom != "CB"
            ]
            selected_coords[:, selected_backbone_indices] = backbone_coords[
                :, backbone_atom_indices
            ]
            return selected_coords

    def residue_separations(self):
        return np.abs(
            self.atoms.residue_index[:, None] - self.atoms.residue_index[None, :]
        )

    def distances(
        self,
        atom_names: Union[str, List[str]],
        residue_mask_from: Optional[np.ndarray] = None,
        residue_mask_to: Optional[np.ndarray] = None,
        nan_fill: Optional[Union[float, str]] = None,
        multi_atom_calc_type: str = "min",
    ) -> np.ndarray:
        # TODO: handle nans
        # TODO: allow non-backbone atoms
        # TODO: handle atoms belonging to wrong modality with nan masking.
        # TODO: are these masks the right way round?
        if residue_mask_from is None:
            residue_mask_from = np.ones(self.num_residues, dtype=bool)
        if residue_mask_to is None:
            residue_mask_to = np.ones(self.num_residues, dtype=bool)
        backbone_coords = self.backbone_coords(atom_names)  # L, n_atoms, 3
        raise NotImplementedError("check from / to masks")
        if isinstance(atom_names, str):
            assert (
                backbone_coords.shape[1] == 1
            ), "Expected single atom distance calculation"
            backbone_coords = np.squeeze(backbone_coords, dim=1)
            dists = np.sqrt(
                np.sum(
                    (
                        backbone_coords[None, residue_mask_from]
                        - backbone_coords[residue_mask_to, None]
                    )
                    ** 2,
                    axis=-1,
                )
            )  # L_i, L_j
        else:
            multi_atom_dists = np.sqrt(
                np.sum(
                    (
                        backbone_coords[None, residue_mask_from, None, :]
                        - backbone_coords[residue_mask_to, None, :None]
                    )
                    ** 2,
                    axis=-1,
                )
            ).reshape(
                (
                    residue_mask_to.sum(),
                    residue_mask_from.sum(),
                    len(atom_names) * len(atom_names),
                )
            )  # L_i, L_j, n_atoms, n_atoms
            if multi_atom_calc_type == "min":
                dists = np.min(multi_atom_dists, axis=-1)
            elif multi_atom_calc_type == "max":
                dists = np.max(multi_atom_dists, axis=-1)
            elif multi_atom_calc_type == "all":
                dists = multi_atom_dists
        if nan_fill is not None:
            if isinstance(nan_fill, float) or isinstance(nan_fill, int):
                dists = np.nan_to_num(dists, nan=nan_fill)
            elif nan_fill == "max":
                if dists.ndim == 2:
                    max_dist = np.nanmax(dists, axis=-1)
                    dists = np.nan_to_num(dists, nan=max_dist)
                elif dists.ndim == 3:
                    max_dist = np.nanmax(dists, axis=(-1, -2))
                    dists = np.nan_to_num(dists, nan=max_dist)
                else:
                    raise ValueError(
                        f"Invalid dists shapel: {dists.shape}. Expected 2 or 3 dims."
                    )
            else:
                raise ValueError(
                    f"Invalid nan_fill: {nan_fill}. Please specify a float or int."
                )
        return dists

    def contacts(self, atom_name: str, threshold: float) -> np.ndarray:
        return self.distances(atom_name, nan_fill="max") < threshold

    def backbone(self) -> T:
        return self.__class__(self.atoms[self.backbone_mask])

    def get_chain(self) -> "BiomoleculeChain":
        raise NotImplementedError()


class BiomoleculeChain(Biomolecule):
    def __init__(
        self,
        atoms: bs.AtomArray,
        residue_dictionary: ResidueDictionary,
        verbose: bool = False,
        backbone_only: bool = False,
    ):
        assert (
            len(np.unique(atoms.chain_id)) == 1
        ), f"Expected single chain, found chain ids {np.unique(atoms.chain_id)}"
        super().__init__(
            atoms=atoms,
            residue_dictionary=residue_dictionary,
            verbose=verbose,
            backbone_only=backbone_only,
        )


class BiomoleculeComplex(Biomolecule):
    def __init__(self, chains: List[BiomoleculeChain]):
        self._chain_ids = [mol.chain_id for mol in chains]
        self._chains_lookup = {mol.chain_id: mol for mol in chains}

    @property
    def atoms(self):
        return sum(
            [prot.atoms for prot in self._proteins_lookup.values()],
            bs.AtomArray(length=0),
        )

    @property
    def chain_ids(self):
        return self._chain_ids

    @property
    def chains(self):
        return [(chain_id, self.get_chain(chain_id)) for chain_id in self.chain_ids]

    def get_chain(self, chain_id: str) -> "BiomoleculeChain":
        return self._chains_lookup[chain_id]

    def interface(
        self,
        atom_names: Union[str, List[str]] = "CA",
        chain_pair: Optional[Tuple[str, str]] = None,
        threshold: float = 10.0,
        nan_fill: Optional[Union[float, str]] = None,
    ) -> T:
        distances = self.interface_distances(
            atom_names=atom_names, chain_pair=chain_pair, nan_fill=nan_fill
        )
        interface_mask = distances < threshold
        return self.__class__.from_atoms(self.atoms[interface_mask])

    def interface_distances(
        self,
        atom_names: Union[str, List[str]] = "CA",
        chain_pair: Optional[Tuple[str, str]] = None,
        nan_fill: Optional[Union[float, str]] = None,
    ) -> np.ndarray:
        if chain_pair is None:
            if len(self._chain_ids) != 2:
                raise ValueError(
                    "chain_pair must be specified for non-binary complexes"
                )
            chain_pair = (self._chain_ids[0], self._chain_ids[1])
        residue_mask_from = self.atoms.chain_id == chain_pair[0]
        residue_mask_to = self.atoms.chain_id == chain_pair[1]
        return self.distances(
            atom_names=atom_names,
            residue_mask_from=residue_mask_from,
            residue_mask_to=residue_mask_to,
            nan_fill=nan_fill,
        )
