import os

import pytest
from biotite.structure.io.pdb import PDBFile


@pytest.fixture(scope="session")
def afdb_atom_array():
    return PDBFile.read(
        os.path.join(os.path.dirname(__file__), "AF-V9HVX0-F1-model_v4.pdb")
    ).get_structure(model=1)


def atoms_top7():
    return PDBFile.read(
        os.path.join(os.path.dirname(__file__), "1qys.pdb")
    ).get_structure(model=1)


@pytest.fixture(scope="session")
def pdb_atoms_top7():
    return atoms_top7()


@pytest.fixture(scope="session")
def cif_file_1aq1():
    return os.path.join(os.path.dirname(__file__), "1aq1.cif")
