import sys
sys.path.append(sys.path[0].replace('/src',''))
import rdkit
import re
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem #used for 3d coordinates
from rdkit.Chem.Scaffolds import rdScaffoldNetwork # scaffolding
import pandas as pd #lots of work with data frames
from rdkit.Chem import PandasTools # Smiles and molecule  integration with Pandas
from rdkit.Chem import rdqueries # search for rdScaffoldAttachment points * to remove
import numpy as np #for arrays in fragment identification

def find_smallest_rings(nodemolecules):
    """Given get_scaffold_vertices list of molecules, remove non-smallest nodes(those with non-ring atoms or non-ring bonds)."""
    no_nonring_atoms = eliminate_nonring_atoms(nodemolecules)
    no_nonring_atoms_or_bonds = eliminate_nonring_bonds(no_nonring_atoms)
    return no_nonring_atoms_or_bonds

def get_scaffold_vertices(molecule):
    """given rdkit Chem.molecule object return list of molecules of fragments generated by scaffolding."""
    scaffold_params = set_scaffold_params()
    scaffold_network = rdScaffoldNetwork.CreateScaffoldNetwork([molecule],scaffold_params)
    node_molecules = [Chem.MolFromSmiles(x) for x in scaffold_network.nodes]
    return node_molecules

def set_scaffold_params():
    """Defines rdScaffoldNetwork parameters."""
    scafnetparams = rdScaffoldNetwork.ScaffoldNetworkParams() #use default bond breaking (break non-ring - ring single bonds, see paper for reaction SMARTS)
    scafnetparams.includeScaffoldsWithoutAttachments = False #maintain attachments in scaffolds
    scafnetparams.includeGenericScaffolds = False #don't include scaffolds without atom labels
    scafnetparams.keepOnlyFirstFragment = False #keep all generated fragments - some were discarded messing with code if True
    return scafnetparams

def get_molecules_atomicnum(molecule):
    """Given molecule object, get list of atomic numbers."""
    atomNumList = []
    for atom in molecule.GetAtoms():
        atomNumList.append(atom.GetAtomicNum())
    return atomNumList    

def get_molecules_atomsinrings(molecule):
    """Given molecule object, get Boolean list of if atoms are in a ring."""
    isInRingList = []
    for atom in molecule.GetAtoms():
        isInRingList.append(atom.IsInRing())
    return isInRingList

def trim_placeholders(rwmol):
    """Given Chem.RWmol, remove atoms with atomic number 0."""
    qa = rdqueries.AtomNumEqualsQueryAtom(0) #define query for atomic number 0
    if len(rwmol.GetAtomsMatchingQuery(qa)) > 0: #if there are matches
        queryMatch = rwmol.GetAtomsMatchingQuery(qa)
        rmAtIdx = []
        for atom in queryMatch: #identify atoms to be removed
            rmAtIdx.append(atom.GetIdx())
        rmAtIdxSort = sorted(rmAtIdx,reverse=True) #remove starting from highest number so upcoming indices not affected
        #e.g. need to remove 3 and 5, if don't do this and you remove 3, then the 5 you want to remove is now 4, and you'll remove wrong atom
        for idx in rmAtIdxSort: #remove atoms
            rwmol.RemoveAtom(idx)
    return rwmol

def get_canonical_molecule(smile: str):
    """Ensures that molecule numbering is consistent with creating molecule from canonical SMILES for consistency."""
    mol = Chem.MolFromSmiles(smile) 
    molsmi = Chem.MolToSmiles(mol) #molsmi is canonical SMILES
    return Chem.MolFromSmiles(molsmi) #create canonical molecule numbering from canonical SMILES

def copy_molecule(molecule):
    molSmi = Chem.MolToSmiles(molecule)
    return Chem.MolFromSmiles(molSmi)

def clean_smile(trimSmi):
    """"""
    trimSmi = trimSmi.replace('[*H]','*')
    trimSmi = trimSmi.replace('[*H3]','*')
    trimSmi = trimSmi.replace('[*H2]','*')
    trimSmi = trimSmi.replace('[*H+]','*')  
    trimSmi = trimSmi.replace('[*H3+]','*')
    trimSmi = trimSmi.replace('[*H2+]','*')
    trimSmi = trimSmi.replace('[*H-]','*')  
    trimSmi = trimSmi.replace('[*H3-]','*')
    trimSmi = trimSmi.replace('[*H2-]','*')
    return trimSmi

def eliminate_nonring_bonds(nodemolecules):
    """Given list of molecules of eliminate_nonring_atoms output, remove molecules that contain bonds that are not ring bonds or double bonded to ring."""
    #mainly removes ring-other ring single bonds, as in biphenyl
    ringFrags=[]
    for frag in nodemolecules:
        flag=1
        for bond in frag.GetBonds():
            if not bond.IsInRing():
                if bond.GetBondType() != Chem.rdchem.BondType.DOUBLE and bond.GetBeginAtom().GetAtomicNum() != 0 and bond.GetEndAtom().GetAtomicNum():
                    flag=0
                    break
        if flag == 1:
            ringFrags.append(frag)
    return ringFrags

def eliminate_nonring_atoms(nodemolecules):
    """given list of molecules of utils.get_scaffold_vertices output, removes molecules that contain atoms that are not in ring or not double bonded to ring."""
    firstParse = []
    for fragMol in nodemolecules:
        flag=1
        for idx,atom in enumerate(fragMol.GetAtoms()):
            nonRingDouble=0
            if not atom.IsInRing(): #if atom is not in ring, check if it is double bonded to a ring
                for neigh in atom.GetNeighbors():
                    bondType = fragMol.GetBondBetweenAtoms(idx,neigh.GetIdx()).GetBondType()
                    #print(bondType)
                    if fragMol.GetAtomWithIdx(neigh.GetIdx()).IsInRing() and bondType ==Chem.rdchem.BondType.DOUBLE:
                        print('I passed the if')
                        nonRingDouble=1
            if atom.GetAtomicNum() != 0: #if not attachment (atomic number 0 used as attachments by rdScaffoldNetwork)
                if not atom.IsInRing(): #if atom is not in ring
                    if nonRingDouble==0: #if atom is not double bonded to ring
                        flag=0 #all the above true, don't remove molecule from output
                        break #will remove from output if a single atom in the node fails the tests
        if flag == 1: #if pass all tests for all atoms, add to list to be returned
            firstParse.append(fragMol)
    return firstParse   