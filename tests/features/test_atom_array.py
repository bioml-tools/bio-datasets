import numpy as np
from biotite.structure.sequence import to_sequence

from bio_datasets.features.atom_array import ProteinAtomArrayFeature
from bio_datasets.structure.protein import ProteinDictionary


def test_encode_decode_atom_array(afdb_atom_array):
    """Round-trip encoding of AtomArray.
    We encode residue-level annotations separately to the atom coords so
    important to check that they get decoded back to the atom array correctly.
    """
    prot_dict = ProteinDictionary.from_preset("protein", keep_oxt=True)
    feat = ProteinAtomArrayFeature(residue_dictionary=prot_dict)
    encoded = feat.encode_example(afdb_atom_array)
    bs_sequences, _ = to_sequence(afdb_atom_array)
    assert prot_dict.decode_restype_index(encoded["restype_index"]) == str(
        bs_sequences[0]
    )
    decoded = feat.decode_example(encoded).atoms
    # TODO: fix these tests - issue is that atoms get reordered
    # assert np.isclose(decoded.coord, afdb_atom_array.coord).all()
    # assert np.all(decoded.atom_name == afdb_atom_array.atom_name)
    assert np.all(decoded.res_name == afdb_atom_array.res_name)
    assert np.all(decoded.res_id == afdb_atom_array.res_id)
    assert np.all(decoded.chain_id == afdb_atom_array.chain_id)


def test_encode_decode_atom_array_without_residue_dictionary(afdb_atom_array):
    """Round-trip encoding of AtomArray.
    We encode residue-level annotations separately to the atom coords so
    important to check that they get decoded back to the atom array correctly.
    """
    feat = ProteinAtomArrayFeature(residue_dictionary=None)
    encoded = feat.encode_example(afdb_atom_array)
    decoded = feat.decode_example(encoded).atoms
    # this time we will drop the OXT atom
    print(decoded.coord[:10], afdb_atom_array.coord[:10])
    # assert np.isclose(decoded.coord, afdb_atom_array.coord[:-1]).all()
    # assert np.all(decoded.atom_name == afdb_atom_array.atom_name[:-1])
    assert np.all(decoded.res_name == afdb_atom_array.res_name[:-1])
    assert np.all(decoded.res_id == afdb_atom_array.res_id[:-1])
    assert np.all(decoded.chain_id == afdb_atom_array.chain_id[:-1])
