import sys
import rdkit
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem #used for 3d coordinates
from rdkit.Chem import Draw,rdDepictor
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from rdkit.Chem.Scaffolds import rdScaffoldNetwork # scaffolding
from rdkit import RDPaths
from rdkit.Chem.Draw import rdMolDraw2D
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import pyvis
from pyvis.network import Network
import inspect
from IPython import display
from IPython.display import SVG
import pandas as pd #lots of work with data frames
from rdkit.Chem import PandasTools # Smiles and molecule  integration with Pandas
from rdkit.Chem import rdqueries # search for rdScaffoldAttachment points * to remove
import numpy as np #for arrays in fragment identification
ifg_path = os.path.join(RDPaths.RDContribDir, 'IFG') #identifying functional groups via Ertl
sys.path.append(ifg_path)
import ifg

def get_canonical_molecule(smile: str):
    mol = Chem.MolFromSmiles(smile) 
    molsmi = Chem.MolToSmiles(mol)
    return Chem.MolFromSmiles(molsmi)

def set_scaffold_params():
    scafnetparams = rdScaffoldNetwork.ScaffoldNetworkParams()
    scafnetparams.includeScaffoldsWithoutAttachments = False
    scafnetparams.includeGenericScaffolds = False
    scafnetparams.keepOnlyFirstFragment = False
    return scafnetparams

def get_scaffold_vertices(molecule):
    scaffold_params = set_scaffold_params()
    scaffold_network = rdScaffoldNetwork.CreateScaffoldNetwork([molecule],scaffold_params)
    node_molecules = [Chem.MolFromSmiles(x) for x in scaffold_network.nodes]
    return node_molecules

def eliminate_nonring_atoms(nodemolecules):
    firstParse = []
    print(len(nodemolecules))
    for fragMol in nodemolecules:
        flag=1
        for idx,atom in enumerate(fragMol.GetAtoms()):
            nonRingDouble=0
            if not atom.IsInRing():
                for neigh in atom.GetNeighbors():
                    bondType = fragMol.GetBondBetweenAtoms(idx,neigh.GetIdx()).GetBondType()
                    #print(bondType)
                    if fragMol.GetAtomWithIdx(neigh.GetIdx()).IsInRing() and bondType ==Chem.rdchem.BondType.DOUBLE:
                        print('I passed the if')
                        nonRingDouble=1
            if atom.GetAtomicNum() != 0:
                if not atom.IsInRing():
                    if nonRingDouble==0:
                        flag=0
                        break
        if flag == 1:
            firstParse.append(fragMol)
    return firstParse    

def eliminate_nonring_bonds(nodemolecules):
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
    
def find_smallest_rings(nodemolecules):
    #print(len(nodemolecules))
    no_nonring_atoms = eliminate_nonring_atoms(nodemolecules)
    #print(len(no_nonring_atoms))
    no_nonring_atoms_or_bonds = eliminate_nonring_bonds(no_nonring_atoms)
    #print(len(no_nonring_atoms_or_bonds))
    return no_nonring_atoms_or_bonds

def get_molecules_atomicnum(molecule):
    atomNumList = []
    for atom in molecule.GetAtoms():
        atomNumList.append(atom.GetAtomicNum())
    return atomNumList    

def get_molecules_atomsinrings(molecule):
    isInRingList = []
    for atom in molecule.GetAtoms():
        isInRingList.append(atom.IsInRing())
    return isInRingList    

def initialize_molecule_frame(molecule):
    atomic_numbers = get_molecules_atomicnum(molecule)
    atoms_in_rings = get_molecules_atomsinrings(molecule)
    initialization_data = {'atomNum': atomic_numbers, 'inRing': atoms_in_rings, 'molPart': ['Unknown'] * molecule.GetNumAtoms()}
    return pd.DataFrame(initialization_data)

def trim_placeholders(rwmol):
    qa = rdqueries.AtomNumEqualsQueryAtom(0)
    if len(rwmol.GetAtomsMatchingQuery(qa)) > 0:
        queryMatch = rwmol.GetAtomsMatchingQuery(qa)
        #print(queryMatch)
        rmAtIdx = []
        for atom in queryMatch:
            rmAtIdx.append(atom.GetIdx())
        rmAtIdxSort = sorted(rmAtIdx,reverse=True)    
        for idx in rmAtIdxSort:
            rwmol.RemoveAtom(idx)
    return rwmol       

def identify_ring_atom_index(molecule,ringFrags):
    #print(ringFrags)
    listOfRings = []
    for ringIdx in ringFrags:
        ring = Chem.RWMol(ringIdx)
        rings_no_placeholders = trim_placeholders(ring)
        matches = molecule.GetSubstructMatches(rings_no_placeholders)
        for match in matches:
            listOfRings.append(match)
    #print(listOfRings)        
    return listOfRings        

def remove_subset_rings(indices):
    uniqueNotSubsetList = []
    for ringIdex in indices:
        flag=0
        for ringIdex2 in indices:
            if len(ringIdex) < len(ringIdex2):
                if(all(x in ringIdex2 for x in ringIdex)):
                    flag=1
        if flag == 0:
            uniqueNotSubsetList.append(ringIdex)
    #print(uniqueNotSubsetList)        
    return uniqueNotSubsetList        

def assign_rings_to_molframe(indices,molFrame):
    ringCount = 1
    for ringIdx in indices:
        molFrame.loc[ringIdx,['molPart']] = 'Ring {ring}'.format(ring=ringCount)
        ringCount += 1
    return molFrame

def set_double_bonded_in_ring(molFrame,molecule):
    inRing = get_molecules_atomsinrings(molecule)
    notinring = []
    inring = []
    for atom in range(len(inRing)):
        if inRing[atom] == False:
            notinring.append(atom)
        else:
            inring.append(atom)
    # print(molFrame)
    molFrameSubset = molFrame.loc[molFrame['inRing']==False,:]
    idxToUpdate = list(molFrameSubset.loc[molFrameSubset['molPart'] != 'Unknown',:].index)
    # print(idxToUpdate)
    molFrame.loc[idxToUpdate,['inRing']] = True
    return molFrame

def assign_side_and_linkers(molFrame,molecule):
    inRing = get_molecules_atomsinrings(molecule)
    notinring = []
    inring = []
    for atom in range(len(inRing)):
        if inRing[atom] == False:
            notinring.append(atom)
        else:
            inring.append(atom)
    inringPY = np.array(inring)        
    checkAtoms = np.array(notinring)
    fgs=[]
    fgType = []
    linkercount = 1
    periphcount=1
    while checkAtoms.size > 0:
        grp = np.array([checkAtoms[0]])
        a = molecule.GetAtomWithIdx(int(checkAtoms[0]))
        aneigh = a.GetNeighbors()
        aneighnum = []
        for n in aneigh:
            aneighnum.append(n.GetIdx())
        aneighnumPY = np.array(aneighnum)    
        ringAtCount=0
        #print(aneighnum)
        newNeighbors=True
        while newNeighbors:
            newNeighbors = False
            for n in np.nditer(aneighnumPY):
                #if molFrame.loc[int(n),'inRing'] == True:
                #    ringAtCount = ringAtCount + 1
                if n in notinring: #molFrame.loc[int(n),'inRing'] != True:           
                    #print(grp)
                    if n not in grp:
                        grp = np.append(grp,n)
                    nneigh = molecule.GetAtomWithIdx(int(n)).GetNeighbors()
                    nneighnum = []
                    for nn in nneigh:
                        nneighnum.append(nn.GetIdx())
                    #print(nneighnum)
                    nneighnumPY = np.array(nneighnum)
                    nneighnumPY = np.setdiff1d(nneighnumPY,inringPY)
            #        it = np.nditer(nneighnumPY,flags=['f_index'])
        #         idxToDe = []
        #          for c in it:
        #               if c not in notinringPY:
        #                    print(c.index)
                    #print(idxToDe)        
                    notInNeigh = np.setdiff1d(nneighnumPY,aneighnumPY)
                    notInNeighGr = np.setdiff1d(notInNeigh,grp)
                    #print(notInNeighGr)
                    if notInNeighGr.size > 0:
                        newNeighbors = True
                    notInGr = np.setdiff1d(nneighnumPY,grp)
                    #print(notInGr)
                    aneighnumPY = np.append(aneighnumPY,notInNeighGr)
                    #print(notInGr in notinringPY)
                    grp = np.append(grp,notInGr)
                    #print(notInNeigh)
                #print(set(nneighnum).difference(aneighnum))
                #aneighnum.append(list(set(nneighnum).difference(aneighnum)))
        checkAtoms=checkAtoms[~np.isin(checkAtoms,grp)]
        atRing=0
        for idx in grp:
            neigh = molecule.GetAtomWithIdx(int(idx)).GetNeighbors()
            for n in neigh:
                if n.GetIdx() in inring:
                    atRing+=1
        #print(atRing)
        if atRing == 1:
            fgType.append("Peripheral {count}".format(count=periphcount))
            periphcount+=1     
        elif atRing > 1:
            fgType.append('Linker {count}'.format(count = linkercount))
            linkercount+=1
        fgs.append(list(grp))

    #Add molecule part label to molecule frame
    i=0
    #print(fgs)
    while i < len(fgs):
        #print(fgType[i])
        for idx in fgs[i]:
    #         print('{index}, {molPart}'.format(index=idx, molPart=molFrame.loc[idx,'molPart']))
            if molFrame.loc[idx,'molPart'] == 'Unknown':
                molFrame.loc[idx,'molPart'] = fgType[i]
        i+=1     
    return molFrame

def generate_part_smiles(molFrame,molecule):
    fragments = molFrame['molPart'].unique()
    fragSmi = []
    tempSmi = Chem.MolToSmiles(molecule)
    for frag in fragments:
        fragDict = {}
        attachN = 0
        #print('Fragment is {frag}'.format(frag=frag))
        atomsInFrag = list(molFrame[molFrame['molPart'] == frag].index)
        #print(molFrame[molFrame['molPart'] == frag])
        atomsInFragAndAtt = []
        for at in atomsInFrag:
            atomsInFragAndAtt.append(at)
        
        #print(atomsInFrag)
        tempMol = Chem.MolFromSmiles(tempSmi)
        #print(atomsInFrag)
        for at in atomsInFrag:
            #print('Frag atom {at}'.format(at=molecule.GetAtomWithIdx(at).GetIdx()))
            atNeigh = molecule.GetAtomWithIdx(at).GetNeighbors()
            for an in atNeigh:
                anidx = an.GetIdx()
                #print('Neighbour atom {an}'.format(an=anidx))
                if anidx not in atomsInFrag:
                    #print('not in fragment, updating to 0')
                    atomsInFragAndAtt.append(anidx)
                    #print(tempMol.GetAtomWithIdx(anidx).GetAtomicNum())
                    tempMol.GetAtomWithIdx(anidx).SetAtomicNum(0)
                    #print(tempMol.GetAtomWithIdx(anidx).GetAtomicNum())
        outSmi = Chem.MolFragmentToSmiles(tempMol,atomsInFragAndAtt)
        outSmi = outSmi.replace('[*H]','*')
        outSmi = outSmi.replace('[*H3]','*')
        outSmi = outSmi.replace('[*H2]','*')
        outSmi = outSmi.replace('[*H+]','*')  
        outSmi = outSmi.replace('[*H3+]','*')
        outSmi = outSmi.replace('[*H2+]','*')
        outSmi = outSmi.replace('[*H-]','*')  
        outSmi = outSmi.replace('[*H3-]','*')
        outSmi = outSmi.replace('[*H2-]','*')
        fragSmi.append(outSmi)
    return fragSmi

def find_alkyl_groups(molFrame,fragSmi,molecule):
    fragments = molFrame['molPart'].unique()
    alkyl_break_params = rdScaffoldNetwork.ScaffoldNetworkParams(['[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0]):2]>>[*]-[*:1].[*]-[*:2]'])
    alkyl_break_params.includeGenericScaffolds=False
    alkyl_break_params.keepOnlyFirstFragment=False
    alkyl_break_params.includeScaffoldsWithoutAttachments=False
    alkyl_break_params.pruneBeforeFragmenting = False
    maybe_alkyl_smi=[]
    for al in fragments: #going over molecule parts again, this time looking for alkyl
        if 'Ring' not in al: #don't do this for ring systems
            atomsInAl = list(molFrame[molFrame['molPart'] == al].index)
            maybe_alkyl_smi.append(Chem.MolFragmentToSmiles(molecule,atomsInAl,canonical=True))
    maybe_alkyl_mol = [Chem.MolFromSmiles(x) for x in maybe_alkyl_smi]
    maybe_alkyl_net = rdScaffoldNetwork.CreateScaffoldNetwork(maybe_alkyl_mol,alkyl_break_params)
    maybe_alkyl_nodes = [Chem.MolFromSmiles(x) for x in maybe_alkyl_net.nodes]
    for node in maybe_alkyl_nodes:
        isAlkyl=1
        for atom in node.GetAtoms():
            if isAlkyl ==0:
                break
            elif atom.GetAtomicNum() not in [0,6]:
                isAlkyl = 0
                continue
        if isAlkyl == 1 and '*' in Chem.MolToSmiles(node):
            fragSmi.append(Chem.MolToSmiles(node))
    return fragSmi

def generate_fragment_frame(fragmentSmiles):
    fragFrame = pd.DataFrame(set(fragmentSmiles),columns=['Smiles']) #here is where we subset to unique
    PandasTools.AddMoleculeColumnToFrame(fragFrame,'Smiles','Molecule',includeFingerprints=True)
    fragFrame.drop(fragFrame.index[fragFrame['Smiles'] == "*"].tolist())
    return fragFrame

def add_ertl_functional_groups(fragFrame):
    for molecule in fragFrame['Molecule']:
        if molecule.GetRingInfo().NumRings() == 0:
            fgs = ifg.identify_functional_groups(molecule)
            for fg in fgs:
                if len(fg) != 0 and fg.atoms != '*':
                    if len(fg.atomIds)==1 and molecule.GetAtomWithIdx(fg.atomIds[0]).IsInRing():
                        print("Skipping")
                        print(fg)
                    else:
                        #print("Adding")
                        #print(fg)
                        justGrp = Chem.MolFromSmarts(fg.atoms)
                        #print(fg.atoms, fg.type)
                        grpAndAtt = Chem.MolFromSmiles(fg.type)
                        if grpAndAtt is not None:
                            matchpatt = grpAndAtt.GetSubstructMatch(justGrp)
                            #print(matchpatt)
                            if len(matchpatt) >0:
                                for atom in grpAndAtt.GetAtoms():
                                    aidx = atom.GetIdx()
                                    if aidx not in list(matchpatt):
                                        grpAndAtt.GetAtomWithIdx(aidx).SetAtomicNum(0)
                                fgSmile = Chem.MolToSmiles(grpAndAtt)
                                flag=0
                                for atom in grpAndAtt.GetAtoms():
                                    if atom.GetAtomicNum() == 0 and atom.IsInRing():
                                        flag = 1
                                if fgSmile not in fragFrame['Smiles'].unique():                    
                                    if fgSmile != len(fgSmile) * "*" and flag == 0:
    #                                     if fgSmile == '**C':
    #                                         fgSmile = 'C(*)*'
                                        fragFrame.loc[len(fragFrame)] = [fgSmile, Chem.MolFromSmiles(fgSmile)]
    return fragFrame     

def add_xyz_coords(fragFrame):
    xyzBlockList = []
    qa = rdqueries.AtomNumEqualsQueryAtom(0)
    for mol in fragFrame['Molecule']:
        hmolrw = Chem.RWMol(mol) # Change type of molecule object
        hmolrw = Chem.AddHs(hmolrw) # Add hydrogens
        zeroAt = hmolrw.GetAtomsMatchingQuery(qa)   #Replace placeholder * with At
        for atom in zeroAt:
            hmolrw.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(85)
        AllChem.EmbedMolecule(hmolrw)

        AllChem.MMFFOptimizeMolecule(hmolrw) #Optimize with MMFF94
        xyzBlockList.append(AllChem.rdmolfiles.MolToXYZBlock(hmolrw)) #Store xyz coordinates
    fragFrame['xyz'] = xyzBlockList  
    return fragFrame

def add_number_attachements(fragFrame):
    attachList = []
    for molecule in fragFrame['Molecule']:
        attach=0
        atoms = molecule.GetAtoms()
        for atom in atoms:
            if atom.GetAtomicNum() == 0:
                attach += 1
        attachList.append(attach)
    fragFrame['numAttachments'] = attachList
    return fragFrame

def identify_fragments(smile: str):
    mol1 = get_canonical_molecule(smile)
    mol1nodemols = get_scaffold_vertices(mol1)
    ringFrags = find_smallest_rings(mol1nodemols)
    molFrame = initialize_molecule_frame(mol1)
    ring_atom_indices = identify_ring_atom_index(mol1,ringFrags)
    ring_indices_nosubset = remove_subset_rings(ring_atom_indices)
    molFrame = assign_rings_to_molframe(ring_indices_nosubset,molFrame)
    molFrame = set_double_bonded_in_ring(molFrame,mol1)
    molFrame = assign_side_and_linkers(molFrame,mol1)
    #print(molFrame)
    fragSmi = generate_part_smiles(molFrame,molecule=mol1)
    fragSmi = find_alkyl_groups(molFrame,fragSmi,mol1)
    #print(fragSmi)
    fragFrame = generate_fragment_frame(fragSmi)
    fragFrame = add_ertl_functional_groups(fragFrame)
    fragFrame = add_xyz_coords(fragFrame)
    fragFrame = add_number_attachements(fragFrame)
    #print(fragFrame)
    return fragFrame