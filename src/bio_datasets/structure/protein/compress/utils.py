import nerfax
from nerfax.utils import get_align_rigid_bodies_fn


def compute_aligned_rmsd(decompressed, original):
    decompressed = get_align_rigid_bodies_fn(decompressed, original.reshape((-1, 3)))(
        decompressed
    )
    return nerfax.foldcomp_tests.compute_rmsd(decompressed, original)
