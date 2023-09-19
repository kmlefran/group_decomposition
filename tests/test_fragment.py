import sys

sys.path.append(sys.path[0].replace("/tests", ""))
import pytest

from group_decomposition.fragfunctions import (
    count_groups_in_set,
    count_uniques,
    identify_connected_fragments,
)

# import unittest

# class TestFragmenting(unittest.TestCase):
def test_two_ethyls_in_smile():
    """We should find two ethyls in this molecule"""
    smi = "c1ccc(c(c1)c2ccc(o2)C(=O)N3C[C@H](C4(C3)CC[NH2+]CC4)C(=O)NCCOCCO)F"
    atframe = identify_connected_fragments(smi)
    uniqueframe = count_uniques(atframe)
    numEt = list(uniqueframe[uniqueframe["Smiles"] == "*CC*"]["count"])[0]
    assert numEt == 2


def test_invalid_smiles_raises_ValueError():
    """Given an invalid smiles, we should see a ValueError"""
    with pytest.raises(ValueError):
        identify_connected_fragments("CBx")


def test_phenyl_in_smi():
    """Test that we find a phenyl ring in the molecule"""
    frag_frame = identify_connected_fragments(
        "c1ccc(cc1)C[C@H]2CC[C@@]3(C2)[C@@H](CCCN3Br)O"
    )
    unique_frame = count_uniques(frag_frame)
    frag_smiles = list(unique_frame["Smiles"])
    phenyl_in_frag = "*c1ccccc1" in frag_smiles
    assert phenyl_in_frag == True


def test_ring_frag_contains_atoms_doublebonded_to_ring():
    """Test that """
    frag_frame = identify_connected_fragments(
        "c1c(cnc(c1C(=O)N[C@@H]2CCS(=O)(=O)C2)NN)Br"
    )
    unique_frame = count_uniques(frag_frame)
    frag_smiles = list(unique_frame["Smiles"])
    double_bonded_ring_in_frag = "*[C@@H]1CCS(=O)(=O)C1" in frag_smiles
    assert double_bonded_ring_in_frag == True


def test_acyclic_molecule_is_fragmented():
    frag_frame = identify_connected_fragments("CCC=O")
    num_frags = len(frag_frame.index)
    assert num_frags == 2


def test_two_phenyl_counted_same_if_drop_attachements():
    frag_frame = identify_connected_fragments("c1cc(ccc1O)Oc2c(ccc(c2F)F)[N+](=O)[O-]")
    unique_frame = count_uniques(frag_frame, drop_attachments=True)
    num_phenyl = list(unique_frame[unique_frame["Smiles"] == "c1ccccc1"]["count"])[0]
    assert num_phenyl == 2
