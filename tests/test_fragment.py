import sys
sys.path.append("/Users/chemlab/Documents/Retrievium Work/Scaffolding/Coding Workspace/scripts")
import fragfinder
from fragfinder import identify_connected_fragments, count_uniques
import rdkit
import unittest

class TestFragmenting(unittest.TestCase):
    def test_count_unique(self):
        smi='c1ccc(c(c1)c2ccc(o2)C(=O)N3C[C@H](C4(C3)CC[NH2+]CC4)C(=O)NCCOCCO)F'
        atframe = identify_connected_fragments(smi)
        uniqueframe = count_uniques(atframe)
        numEt = uniqueframe[uniqueframe['Smiles']=='*CC*']['count'][0]
        self.assertEqual(numEt,2,'should be 2 ethyl')

if __name__ == '__main__':
    unittest.main()