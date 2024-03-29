"""Tests that fragmenting is as expected"""
import sys

import pytest

from group_decomposition.fragfunctions import (
    count_uniques,
    identify_connected_fragments,
)

sys.path.append(sys.path[0].replace("/tests", ""))


# import unittest

# class TestFragmenting(unittest.TestCase):
def test_ether_rejoined():
    """The that the monoatomic oxygen in the chain gets rejoined to the fragments"""
    smi = "c1ccc(c(c1)c2ccc(o2)C(=O)N3C[C@H](C4(C3)CC[NH2+]CC4)C(=O)NCCOCCO)F"
    atframe = identify_connected_fragments(
        smi, bb_patt="[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0;!#9;!#17;!#35;!#1]):2]"
    )
    uniqueframe = count_uniques(atframe)
    assert "*CCOCC*" in list(uniqueframe["Smiles"])


def test_invalid_smiles_raises_ValueError():
    """Given an invalid smiles, we should see a ValueError"""
    # a
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
    assert phenyl_in_frag is True


def test_ring_frag_contains_atoms_doublebonded_to_ring():
    """Test that"""
    frag_frame = identify_connected_fragments(
        "c1c(cnc(c1C(=O)N[C@@H]2CCS(=O)(=O)C2)NN)Br"
    )
    unique_frame = count_uniques(frag_frame)
    frag_smiles = list(unique_frame["Smiles"])
    double_bonded_ring_in_frag = "*[C@@H]1CCS(=O)(=O)C1" in frag_smiles
    assert double_bonded_ring_in_frag is True


def test_acyclic_molecule_is_fragmented():
    """Test that fragments are generated for acyclic molecules"""
    frag_frame = identify_connected_fragments("CCC=O")
    num_frags = len(frag_frame.index)
    assert num_frags == 2


def test_two_phenyl_counted_same_if_drop_attachements():
    """Test that multiple phenyls are counted with different attachment points"""
    frag_frame = identify_connected_fragments("c1cc(ccc1O)Oc2c(ccc(c2F)F)[N+](=O)[O-]")
    unique_frame = count_uniques(frag_frame, drop_attachments=True)
    num_phenyl = list(unique_frame[unique_frame["Smiles"] == "c1ccccc1"]["count"])[0]
    assert num_phenyl == 2
