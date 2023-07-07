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
from group_decomposition import ifg
from group_decomposition import utils

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


def initialize_molecule_frame(molecule):
    """Given a molecule, create an initial data frame for identifying main parts of molecule (Rings/side chains/linkers)"""
    atomic_numbers = utils.get_molecules_atomicnum(molecule)
    atoms_in_rings = utils.get_molecules_atomsinrings(molecule)
    initialization_data = {'atomNum': atomic_numbers, 'inRing': atoms_in_rings, 'molPart': ['Unknown'] * molecule.GetNumAtoms()}
    return pd.DataFrame(initialization_data)
   

def identify_ring_atom_index(molecule,ringFrags):
    """Given molecules and list of rings(from utils.find_smallest_rings), return list of lists of indices of the ring atoms. Each element in list is a ring, each element in that list is the index of an atom in ring."""
    listOfRings = []
    for ringIdx in ringFrags:
        ring = Chem.RWMol(ringIdx)
        rings_no_placeholders = utils.trim_placeholders(ring)
        matches = molecule.GetSubstructMatches(rings_no_placeholders)
        for match in matches:
            listOfRings.append(match)
     
    return listOfRings        

def remove_subset_rings(indices):
    """Given identify_ring_atom_index, remove lists from the index that are subset of the other lists."""
    #For example, if a molecule had a phenyl ring and napthalene ring scaffolds, the phenyl would also show up as a structure match for the napthalene.
    #Here, we check: phenyl: [1,2,3,4,5,6], phenyl2:[7,8,9,10,11,12], napthalene: [7,8,9,10,11,12,13,14,15,16], remove phenyl2 from list since all its atoms are in napthalene
    uniqueNotSubsetList = []
    for ringIdex in indices:
        flag=0 #will change to 1 if it is a subset
        for ringIdex2 in indices:
            if len(ringIdex) < len(ringIdex2): #only check if ringIdex smaller than ringIdex2(otherwise can't be subset)
                if(all(x in ringIdex2 for x in ringIdex)):
                    flag=1 #ringIdex is a subset, do not add it
        if flag == 0:
            uniqueNotSubsetList.append(ringIdex)  
    return uniqueNotSubsetList        

def assign_rings_to_molframe(indices,molFrame):
    """Given list of indices, update molFrame so that the atoms are assigned to rings."""
    ringCount = 1 #parts are labeled Ring 1, Ring 2 .... Ring R
    for ringIdx in indices:
        molFrame.loc[ringIdx,['molPart']] = 'Ring {ring}'.format(ring=ringCount) #find index, update to correct ring
        ringCount += 1 #next ring is Ring 2
    return molFrame

def set_double_bonded_in_ring(molFrame,molecule):
    """Given molFrame updated by assign_rings_to_molframe, and the parent molecule, ensure that atoms double bonded to the ring are counted as inRing in the Boolean column."""
    # inRing = utils.get_molecules_atomsinrings(molecule)
    # notinring = []
    # inring = []
    # for atom in range(len(inRing)):
    #     if inRing[atom] == False:
    #         notinring.append(atom)
    #     else:
    #         inring.append(atom)
    molFrameSubset = molFrame.loc[molFrame['inRing']==False,:]
    idxToUpdate = list(molFrameSubset.loc[molFrameSubset['molPart'] != 'Unknown',:].index) #find the subset that have a label e.g. Ring 1, but not labeled inRing
    molFrame.loc[idxToUpdate,['inRing']] = True #update the needed atoms to 
    return molFrame

def assign_side_and_linkers(molFrame,molecule):
    """Given a molFrame updated with Rings and the parent molecule, assign the remaining atoms to side chains or linkers."""
    inRing = utils.get_molecules_atomsinrings(molecule)
    notinring = []
    inring = []
    for atom in range(len(inRing)):
        if inRing[atom] == False:
            notinring.append(atom)
        else:
            inring.append(atom)
    inringPY = np.array(inring)  #Generate list of atoms in rings
    checkAtoms = np.array(notinring) #We will check over the atoms not in rings
    fgs=[]
    fgType = []
    linkercount = 1 #For labeling Linker 1, Linker 2...start at 1
    periphcount=1 #see line above, but for side chains
    while checkAtoms.size > 0: #while there are atoms left to check
        grp = np.array([checkAtoms[0]]) #initialize atoms in this group, starting with the first atom
        a = molecule.GetAtomWithIdx(int(checkAtoms[0])) #atom a is the first atom remaining in checkAtoms by index
        aneigh = a.GetNeighbors()
        #get the indices of the neighbours of ay in np array
        aneighnum = []
        for n in aneigh:
            aneighnum.append(n.GetIdx()) 
        aneighnumPY = np.array(aneighnum)    
        ringAtCount=0
        #Initialize newNeighbours to True, will be set to false in loop if an iteration fails to look at an atom we ahve already seen
        newNeighbors=True
        while newNeighbors:
            newNeighbors = False #will be set to true if we find a new neighbour, and loop will continue to next iteration to check the neighbours of that atom
            for n in np.nditer(aneighnumPY): #iterate over the neighbours of atom a
                if n in notinring: #only checking non-ring atoms as ring atoms won't be a side chain or a linker
                    if n not in grp:
                        grp = np.append(grp,n) #if neighbour n is not in a ring, and not yet added to the group, add it to the group
                    nneigh = molecule.GetAtomWithIdx(int(n)).GetNeighbors()
                    nneighnum = []
                    for nn in nneigh:
                        nneighnum.append(nn.GetIdx())
                    nneighnumPY = np.array(nneighnum) #find list of neighbours of n
                    nneighnumPY = np.setdiff1d(nneighnumPY,inringPY)    #neighbours not in ring
                    notInNeigh = np.setdiff1d(nneighnumPY,aneighnumPY) #not in ring and not in neighbours of a
                    notInNeighGr = np.setdiff1d(notInNeigh,grp) #not in ring, not in neighbours of a and not in grp already
                    if notInNeighGr.size > 0: #if any remaining, we have new neighbours and continue iteration
                        newNeighbors = True
                    notInGr = np.setdiff1d(nneighnumPY,grp) #find those not in group or ring to add
                    aneighnumPY = np.append(aneighnumPY,notInNeighGr) #add atoms to list we are iterating over so we check them too
                    grp = np.append(grp,notInGr) #add the neighbours not in ring or not already in the group to the group
        checkAtoms=checkAtoms[~np.isin(checkAtoms,grp)] #remove atoms in grp from checkAtoms
        #at this point, we are done iterating over the set of connected atoms comprising the linker/peripheral
        #the next iteration, we will start at another atomt that we have not checked, generate its connectivity and group, remove those etc.
        atRing=0
        for idx in grp: #counter number of atoms in the group that are bonded to rings, if ==1, it is side chain, if ==2, it is linker
            neigh = molecule.GetAtomWithIdx(int(idx)).GetNeighbors()
            for n in neigh:
                if n.GetIdx() in inring:
                    atRing+=1
        if atRing == 1:
            fgType.append("Peripheral {count}".format(count=periphcount))
            periphcount+=1     
        elif atRing > 1:
            fgType.append('Linker {count}'.format(count = linkercount))
            linkercount+=1
        fgs.append(list(grp)) 
    i=0
    while i < len(fgs): #update molFrame with group parts
        for idx in fgs[i]:
            if molFrame.loc[idx,'molPart'] == 'Unknown':
                molFrame.loc[idx,'molPart'] = fgType[i]
        i+=1     
    return molFrame

def generate_part_smiles(molFrame,molecule):
    """Given complete molFrame and molecule, generate SMILES for each part(Ring/linker/peripheral), return as list."""
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
    """Given molFrame(complete), list of molecule parts as SMILES, and a molecule object, add all continuous alkyl groups to list of SMILES."""
    fragments = molFrame['molPart'].unique()
    #Custom bond breaking - break single (-) bonds between carbons with 4 total attachments(i.e. sp3) that are not in a ring ([C;X4;!R])
    #and atoms that are (ring atoms or not sp3 carbons) AND not atomic number 0 ([R,!$([C;X4]);!#0])
    #fragment 1 is the alkyl, fragment 2 is the remainder [*]-[*:1].[*]-[*:2]
    alkyl_break_params = rdScaffoldNetwork.ScaffoldNetworkParams(['[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0]):2]>>[*]-[*:1].[*]-[*:2]'])
    alkyl_break_params.includeGenericScaffolds=False
    alkyl_break_params.keepOnlyFirstFragment=False
    alkyl_break_params.includeScaffoldsWithoutAttachments=False
    alkyl_break_params.pruneBeforeFragmenting = False
    maybe_alkyl_smi=[]
    #for all molecule parts but rings, perform the fragmentation and generate smiles
    for al in fragments: #going over molecule parts again, this time looking for alkyl
        if 'Ring' not in al: #don't do this for ring systems
            atomsInAl = list(molFrame[molFrame['molPart'] == al].index)
            maybe_alkyl_smi.append(Chem.MolFragmentToSmiles(molecule,atomsInAl,canonical=True))        
    maybe_alkyl_mol = [Chem.MolFromSmiles(x) for x in maybe_alkyl_smi]
    maybe_alkyl_net = rdScaffoldNetwork.CreateScaffoldNetwork(maybe_alkyl_mol,alkyl_break_params)
    maybe_alkyl_nodes = [Chem.MolFromSmiles(x) for x in maybe_alkyl_net.nodes]
    #check each node, and if all the atoms are placeholder or carbon, add it to fragments
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
    """Given list of SMILES Generate output frame with SMILES codes and molecules for unique fragments."""
    fragFrame = pd.DataFrame(set(fragmentSmiles),columns=['Smiles']) #here is where we subset to unique
    PandasTools.AddMoleculeColumnToFrame(fragFrame,'Smiles','Molecule',includeFingerprints=True)
    fragFrame.drop(fragFrame.index[fragFrame['Smiles'] == "*"].tolist())
    return fragFrame

def add_ertl_functional_groups(fragFrame):
    """For each SMILES in fragFrame, identify heavy/double bonded atom functional groups and update the frame with them."""
    for molecule in fragFrame['Molecule']:
        if molecule.GetRingInfo().NumRings() == 0: #don't do this for rings
            fgs = ifg.identify_functional_groups(molecule) 
            for fg in fgs: #iterate over identified functional groups
                if len(fg) != 0 and fg.atoms != '*':
                    if len(fg.atomIds)==1 and molecule.GetAtomWithIdx(fg.atomIds[0]).IsInRing(): #if one atom and it is a ring atom, skip
                        print("Skipping")
                        print(fg)
                    else:
                        justGrp = Chem.MolFromSmarts(fg.atoms) #generate molecule of functional group
                        grpAndAtt = Chem.MolFromSmiles(fg.type) #generate molecule of functional group environment
                        if grpAndAtt is not None: #if initialized successfully
                            matchpatt = grpAndAtt.GetSubstructMatch(justGrp) #find the match
                            if len(matchpatt) >0:
                                for atom in grpAndAtt.GetAtoms():
                                    aidx = atom.GetIdx()
                                    if aidx not in list(matchpatt): #set the atoms in grpAndAtt to 0 if they are not in the group
                                        grpAndAtt.GetAtomWithIdx(aidx).SetAtomicNum(0)
                                fgSmile = Chem.MolToSmiles(grpAndAtt) 
                                flag=0
                                for atom in grpAndAtt.GetAtoms():
                                    if atom.GetAtomicNum() == 0 and atom.IsInRing():
                                        flag = 1
                                if fgSmile not in fragFrame['Smiles'].unique(): #append to fragFrame if placeholders not in ring                    
                                    if fgSmile != len(fgSmile) * "*" and flag == 0:
                                        fragFrame.loc[len(fragFrame)] = [fgSmile, Chem.MolFromSmiles(fgSmile)]
    return fragFrame     

def add_xyz_coords(fragFrame):
    """Given fragFrame with molecules, add xyz coordinates form MM94 optimization to it."""
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
    """Add number of attachments column to fragFrame, counting number of *."""
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

def generate_full_molFrame(mol1):
    """Generate data frame for molecule assigning all atoms to rings/linkers/peripherals."""
    mol1nodemols = utils.get_scaffold_vertices(mol1)
    ringFrags = utils.find_smallest_rings(mol1nodemols)
    molFrame = initialize_molecule_frame(mol1)
    ring_atom_indices = identify_ring_atom_index(mol1,ringFrags)
    ring_indices_nosubset = remove_subset_rings(ring_atom_indices)
    molFrame = assign_rings_to_molframe(ring_indices_nosubset,molFrame)
    molFrame = set_double_bonded_in_ring(molFrame,mol1)
    molFrame = assign_side_and_linkers(molFrame,mol1)
    return molFrame


def identify_fragments(smile: str):
    """Perform the full fragmentation.
    
    This will break down molecule into unique rings, linkers, functional groups, peripherals, but not maintain connectivity.
    Atoms may be assigned to multiple groups. eg. C(=O)C, C=O, C may all be SMILES included from an acetyl peripheral
    """
    mol = utils.get_canonical_molecule(smile)
    molFrame = generate_full_molFrame(mol)
    #print(molFrame)
    fragSmi = generate_part_smiles(molFrame,molecule=mol)
    fragSmi = find_alkyl_groups(molFrame,fragSmi,mol)
    #print(fragSmi)
    fragFrame = generate_fragment_frame(fragSmi)
    fragFrame = add_ertl_functional_groups(fragFrame)
    fragFrame = add_xyz_coords(fragFrame)
    fragFrame = add_number_attachements(fragFrame)
    #print(fragFrame)
    return fragFrame


def trim_molpart(molFrame,molPartList,molecule):
    """Given molFrame, and unique parts in molFrame, and the molecule, break molecule into the unique parts."""
    #will return with connections set to count:*, not labeled
    count=1
    bonds = []
    labels = []
    for molPart in molPartList:
        atomsInMolPart = molFrame.loc[molFrame['molPart'] == molPart,:].index
        for atom in atomsInMolPart:
            aidx = molecule.GetAtomWithIdx(atom).GetIdx()
            for neigh in molecule.GetAtomWithIdx(atom).GetNeighbors():
                nidx = neigh.GetIdx()
                if nidx not in atomsInMolPart:
                    b = molecule.GetBondBetweenAtoms(aidx,nidx)
                    if b.GetIdx() not in bonds:
                        bonds.append(b.GetIdx())
                        labels.append([count,count])
                        count  = count+1
    print(bonds)
    print(labels)
    fragMol = Chem.FragmentOnBonds(molecule,bondIndices=bonds,dummyLabels=labels)
    splitSmiles  = Chem.MolToSmiles(fragMol).split('.')#remove molecule from fragframe, add fragments
    newSmi=[]
    for split in splitSmiles:
        newSmi.append(Chem.MolToSmiles(Chem.MolFromSmiles(split)))
    return {'smiles':newSmi, 'count':count}


def break_molparts(molPartSmi,count):
    """For a given list of Smiles of the molecule parts, break non-ring groups into Ertl functional groups and alkyl groups."""
    elToRM = []
    newSmi=[]
    for i,partsmi in enumerate(molPartSmi):
        molecule = Chem.MolFromSmiles(partsmi)
        if molecule.GetRingInfo().NumRings() == 0: #don't do this for rings
            bondsToBreak = molecule.GetSubstructMatches(Chem.MolFromSmarts('[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0]):2]')) #break sp3 carbon to ring/heteroatom bonds
            bonds = []
            labels = []
            for bond in bondsToBreak: #iterate over matches, storing bond index, and the dummy atom labels to be used in a list
                b = molecule.GetBondBetweenAtoms(bond[0],bond[1])
                bonds.append(b.GetIdx())
                labels.append([count,count])
                count  = count+1 #all dummy atom labels will be different for different bond breaking
            if bonds:
                elToRM.append(i)
                fragMol = Chem.FragmentOnBonds(molecule,bondIndices=bonds,dummyLabels=labels)
                splitSmiles  = Chem.MolToSmiles(fragMol).split('.')#FragmentOnBonds returns SMILES with fragments separated by ., get each fragment in its own string in a list
                for split in splitSmiles:
                    newSmi.append(Chem.MolToSmiles(Chem.MolFromSmiles(split))) #store canonical fragment smiles in new list
    elToRM = sorted(elToRM,reverse=True)
    for el in elToRM:
        del molPartSmi[el] #if the molPart was broken apart, remove it from the output so each atom is uniquely assigned
    outSmi  = molPartSmi + newSmi #this is the Smiles of all fragments in the molecule
    return outSmi        




def identify_connected_fragments(smile: str):
    """
    Given Smiles string, identify fragments in the molecule as follows:
    Break all ring-non-ring atom single bonds
    Store the resulting fragments
    For non-ring fragments, separate those into alkyl chains and hetero/double bonded atoms (similar to Ertl functional groups)
    Each bond breaking, connectivity is maintained through dummy atom labels.
    e.g. C-N -> C-[1*] N-[1*] - reattaching via the matching labels would reassemble the molecule
    
    Args:
        smile: a string containing smiles for a given molecule, does not need to be canonical
    Returns:
        pandas data frame with columms 'Smiles', 'Molecule', 'numAttachments' and 'xyz'
        Containing, fragment smiles, fragment Chem.Molecule object, number of * placeholders, and rough xyz coordinates for the fragment is * were At
    Notes: currently will break apart a functional group if contains a ring-non-ring single bond. e.g. ring N-nonring S=O -> ring N-[1*] nonring S=O-[1*]    
    """
    #ensure smiles is canonical so writing and reading the smiles will result in same number ordering of atoms
    mol = utils.get_canonical_molecule(smile)
    #assign molecule into parts (Rings, side chains, peripherals)
    molFrame = generate_full_molFrame(mol)
    #break molecule into fragments defined by the unique parts in molFrame (Ring 1, Peripheral 1, Linker 1, Linker 2, etc.)
    fragmentSmiles = trim_molpart(molFrame,molFrame['molPart'].unique(),mol)
    #break side chains and linkers into Ertl functional groups and alkyl chains
    fullSmi = break_molparts(fragmentSmiles['smiles'],fragmentSmiles['count'])
    #initialize the output data frame
    fragFrame = generate_fragment_frame(fullSmi)
    #add hydrogens and xyz coordinates resulting from MMFF94 opt, changing placeholders to At
    fragFrame = add_xyz_coords(fragFrame)
    #count number of placeholders in each fragment - it is the number of places it is attached
    fragFrame = add_number_attachements(fragFrame)
    return fragFrame

def count_uniques(fragFrame,dropAttachments=False):
    """
    Given fragframe resulting from identify_connected_fragments, remove dummy atom labels(and placeholders entirely if dropAttachments=True)
    Then, compare the Smiles to count the unique fragments, and return a version of fragFrame that only includes unique fragments
    and the number of times each unique fragment occurs.

    Args:
        fragFrame: frame resulting from identify_connected_fragments typically, or any similar frame with a list of SMILES codes in column ['Smiles']
        dropAttachments: boolean, if False, retains placeholder * at points of attachment, if True, removes * for fragments with more than one atom

    Returns:
        pandas data frame with columns 'Smiles', 'count' and 'Molecule', containing the Smiles string, the number of times the Smiles was in fragFrame, and rdkit.Chem.Molecule object    

    Notes: if dropAttachments=False, similar fragments with different number/positions of attachments will not count as being the same.
    e.g. ortho-attached aromatics would not match with meta or para attached aromatics        
    """
    smileList = fragFrame['Smiles']
    noConnectSmile=[]
    for smile in smileList:
        if dropAttachments:
            mol = Chem.MolFromSmiles(smile)
            nonZeroAtoms=0
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() != 0:
                    nonZeroAtoms += 1
            if nonZeroAtoms > 1:
                temp= re.sub('\[[0-9]+\*\]', '', smile)
                tm = Chem.MolFromSmiles(re.sub('\(\)', '', temp))
                if tm is None:
                    noConnectSmile.append(Chem.MolToSmiles(Chem.MolFromSmiles(re.sub('\[[0-9]+\*\]', '*', smile))))
                    Warning('Could not construct molecule after dropping attachments, maintaining attachments for {smile}'.format(smile=smile))
                else:
                    noConnectSmile.append(Chem.MolToSmiles(tm))
            else:
                noConnectSmile.append(Chem.MolToSmiles(Chem.MolFromSmiles(re.sub('\[[0-9]+\*\]', '*', smile))))
        else:
            noConnectSmile.append(Chem.MolToSmiles(Chem.MolFromSmiles(re.sub('\[[0-9]+\*\]', '*', smile))))
    uniqueSmiles=[]
    uniqueSmilesCounts=[]
    for smile in noConnectSmile:
        if smile not in uniqueSmiles:
            uniqueSmiles.append(smile)
            uniqueSmilesCounts.append(1)
        else:
            print(uniqueSmiles)
            ix = uniqueSmiles.index(smile)
            print(ix)
            uniqueSmilesCounts[ix] += 1        
    uniqueFragFrame = pd.DataFrame(uniqueSmiles,columns=['Smiles']) #here is where we subset to unique
    PandasTools.AddMoleculeColumnToFrame(uniqueFragFrame,'Smiles','Molecule',includeFingerprints=True)
    uniqueFragFrame['count']=uniqueSmilesCounts
    return uniqueFragFrame

def mergeUniques(frame1,frame2):
    """Given two frames of unique fragments, identify shared unique fragments, merge count and frames together.
    
    Args:
        frame1: a frame output from count_uniques
        frame2: a distinct frame also from count_uniques

    Returns:
        a frame resulting from the merge of frame1 and frame2. All rows that have Smiles that are in frame1 but not frame2(and vice versa) are included unmodified
        If a row's SMILES is in both frame1 and frame2, modify the row to update the count of that fragment as sum of frame1 and frame2, then include one row.

    Note:
        for best results, SMILES must be canonical so that they can be exactly compared.
        Smiles in frame should be resulting from Chem.MolToSmiles(Chem.MolFromSmiles(smile)) - this will create a molecule from the smile, and write the smile back, in canonical form    

    Example usage:
        frame1:
        Smiles  count
        C       2
        C1CCC1  1

        frame2:
        Smiles  count
        C       3
        C1CC1   2

        mergeUniques(frame1,frame2) returns
        Smiles  count
        C       5
        C1CCC1  1
        C1CC1   2
    """
    madeFrame=0
    rowsToDropOne = []
    rowsToDropTwo = []
    for i,smi in enumerate(frame1['Smiles']):
        if smi in list(frame2['Smiles']):
            j=list(frame2['Smiles']).index(smi)
            cumCount=frame1['count'][i] + frame2['count'][j]
            if madeFrame == 0:
                mergeFrame = pd.DataFrame.from_dict({'Smiles':[smi],'count':[cumCount]})
                madeFrame=1
            else:
                mergeFrame.loc[len(mergeFrame)] = [smi, cumCount]
            rowsToDropOne.append(i)
            rowsToDropTwo.append(j)
    print(rowsToDropTwo)
    print(rowsToDropOne)
    dropframe1 = frame1.drop(rowsToDropOne)
    dropframe2 = frame2.drop(rowsToDropTwo)
    print(frame1)
    print(frame2)
    for i,smi in enumerate(dropframe1['Smiles']):
        if madeFrame == 0:
            mergeFrame = pd.DataFrame.from_dict({'Smiles':[smi],'count':list(dropframe1['count'])[i]})
            madeFrame=1
        else:
            print(mergeFrame)
            print(len(mergeFrame))
            print(smi)
            print(frame1['count'][i])
            mergeFrame.loc[len(mergeFrame)] = [smi, list(dropframe1['count'])[i]]
    for i,smi in enumerate(dropframe2['Smiles']):
        if madeFrame == 0:
            mergeFrame = pd.DataFrame.from_dict({'Smiles':[smi],'count':list(dropframe2['count'])[i]})
            madeFrame=1
        else:
            mergeFrame.loc[len(mergeFrame)] = [smi, list(dropframe2['count'])[i]]           
    return mergeFrame

def count_groups_in_set(listOfSmiles,dropAttachments=False):
    """Identify unique fragments in molecules defined in the listOfSmiles, and count the number of occurences for duplicates.
    Args:
        listOfSmiles: A list, with each element being a SMILES string, e.g. ['CC','C1CCCC1']
        dropAttachments: Boolean for whether or not to drop attachment points from fragments
            if True, will remove all placeholder atoms indicating connectivity
            if False, placeholder atoms will remain
    Returns:
        an output pd.DataFrame, with columns 'Smiles' for fragment Smiles, 'count' for number of times each fragment occurs in the list, and 'Molecule' holding a rdkit.Chem.Molecule object
        
    Example usage:
        count_groups_in_set(['c1ccc(c(c1)c2ccc(o2)C(=O)N3C[C@H](C4(C3)CC[NH2+]CC4)C(=O)NCCOCCO)F','Cc1nc2ccc(cc2s1)NC(=O)c3cc(ccc3N4CCCC4)S(=O)(=O)N5CCOCC5'],dropAttachments=False)."""
    i=0
    for smile in listOfSmiles:
        frame = identify_connected_fragments(smile)
        uniqueFrame = count_uniques(frame,dropAttachments)
        if i==0:
            i+=1
            outFrame=uniqueFrame
        else:
            outFrame = mergeUniques(outFrame,uniqueFrame)
    PandasTools.AddMoleculeColumnToFrame(outFrame,'Smiles','Molecule',includeFingerprints=True)
    return outFrame        