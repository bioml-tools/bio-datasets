"""Use utils from nerfax parser to convert protein into internal coordinates."""
import jax
import nerfax
import numpy as np
from biotite import structure as bs


def load_backbone_coord_array(structure: bs.AtomArray):
    xyz = np.stack(
        [structure[structure.atom_name == at].coord for at in ["N", "CA", "C"]],
        axis=1,
    )  # L, 3, 3 -> Lx3, 3
    return xyz


def get_backbone_internals(backbone_coords: np.ndarray):
    lengths, angles, torsions = nerfax.parser.xyz_to_internal_coords(
        nerfax.parser.insert_zero(backbone_coords).reshape((-1, 3))
    )
    angles = angles.at[0, 0].set(1.0)  # (Dummy non-zero angle required)
    lengths = lengths.at[0, 0].set(lengths.at[1, 0].get())  # (Dummy initial bond length)
    return jax.device_get(lengths), jax.device_get(angles), jax.device_get(torsions)


def get_backbone_internals_from_atoms(structure: bs.AtomArray):
    # https://github.com/PeptoneLtd/nerfax/blob/2dd1ea019197cd0e273a8d5b920cc850c6b03460/nerfax/mpnerf_constants.py#L590
    xyz = load_backbone_coord_array(structure)
    return get_backbone_internals(xyz)


def get_sidechain_internals(structure: bs.AtomArray):
    pass
