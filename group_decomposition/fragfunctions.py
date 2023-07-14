"""
fragfunctions module

code used to generate fragments of molecules from SMILES code and analyze them

Main functions to call are:
identify_connected_fragments - takes one molecule SMILES, returns fragments with connections
count_uniques - takes output from above, removes attachments and counts unique fragments
count_groups_in_set - takes list of SMILES and counts unique fragments on set
"""
import sys
import re
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools, rdqueries #used for 3d coordinates
from rdkit.Chem.Scaffolds import rdScaffoldNetwork # scaffolding
import pandas as pd #lots of work with data frames
import numpy as np #for arrays in fragment identification
sys.path.append(sys.path[0].replace('/src',''))
from group_decomposition import ifg
from group_decomposition import utils

# def eliminate_nonring_atoms(nodemolecules):
#     """given list of molecules of utils.get_scaffold_vertices output, 
#     removes molecules that contain
#       atoms that are not in ring or not double bonded to ring."""
#     first_parse = []
#     for frag_mol in nodemolecules:
#         flag=1
#         for idx,atom in enumerate(frag_mol.GetAtoms()):
#             non_ring_double=0
#             #if atom is not in ring, check if it is double bonded to a ring
#             if not atom.IsInRing():
#                 for neigh in atom.GetNeighbors():
#                     bond_type = frag_mol.GetBondBetweenAtoms(idx,neigh.GetIdx()).GetBondType()
#                     #print(bondType)
#                     nir = frag_mol.GetAtomWithIdx(neigh.GetIdx()).IsInRing()
#                     if  nir and bond_type ==Chem.rdchem.BondType.DOUBLE:
#                         print('I passed the if')
#                         non_ring_double=1
#             #if not attachment (atomic number 0 used as attachments by rdScaffoldNetwork)
#             if atom.GetAtomicNum() != 0:
#                 if not atom.IsInRing(): #if atom is not in ring
#                     if non_ring_double==0: #if atom is not double bonded to ring
#                         flag=0 #all the above true, don't remove molecule from output
#                         #will remove from output if a single atom in the node fails the tests
#                         break
#         if flag == 1: #if pass all tests for all atoms, add to list to be returned
#             first_parse.append(frag_mol)
#     return first_parse


def _initialize_molecule_frame(molecule):
    """Given a molecule, assign create frame with atomic numbers, Boolean of if in ring
    and unknown column

    """
    atomic_numbers = utils.get_molecules_atomicnum(molecule)
    atoms_in_rings = utils.get_molecules_atomsinrings(molecule)
    initialization_data = {'atomNum': atomic_numbers,
                           'inRing': atoms_in_rings, 
                           'molPart': ['Unknown'] * molecule.GetNumAtoms()}
    return pd.DataFrame(initialization_data)


def _identify_ring_atom_index(molecule,ring_frags):
    """Given molecules and list of rings(from utils.find_smallest_rings), return list of lists of
    indices of the ring atoms. Each element in list is a ring, each element in that list is the
    index of an atom in ring."""
    list_of_rings = []
    for ring_id in ring_frags:
        ring = Chem.RWMol(ring_id)
        rings_no_placeholders = utils.trim_placeholders(ring)
        matches = molecule.GetSubstructMatches(rings_no_placeholders)
        for match in matches:
            list_of_rings.append(match)
    return list_of_rings

def _remove_subset_rings(indices):
    """Given _identify_ring_atom_index, remove lists from the index that are 
    subset of the other lists."""
    #For example, if a molecule had a phenyl ring and napthalene ring scaffolds,
    # the phenyl would also show up as a structure match for the napthalene.
    #Here, we check: phenyl: [1,2,3,4,5,6], phenyl2:[7,8,9,10,11,12],
    # napthalene: [7,8,9,10,11,12,13,14,15,16],
    # remove phenyl2 from list since all its atoms are in napthalene
    unique_not_subset_list = []
    for ring_id in indices:
        flag=0 #will change to 1 if it is a subset
        for ring_id_b in indices:
            #only check if ringIdex smaller than ringIdex2(otherwise can't be subset)
            if len(ring_id) < len(ring_id_b):
                if  all(x in ring_id_b for x in ring_id):
                    flag=1 #ringIdex is a subset, do not add it
        if flag == 0:
            unique_not_subset_list.append(ring_id)
    return unique_not_subset_list

def _assign_rings_to_mol_frame(indices,mol_frame):
    """Given list of indices, update mol_frame so that the atoms are assigned to rings."""
    ring_count = 1 #parts are labeled Ring 1, Ring 2 .... Ring R
    for ring_id in indices:
        #find index, update to correct ring
        mol_frame.loc[ring_id,['molPart']] = 'Ring {ring}'.format(ring=ring_count)
        ring_count += 1 #next ring is Ring 2
    return mol_frame

def _set_double_bonded_in_ring(mol_frame):
    """Given mol_frame updated by _assign_rings_to_mol_frame, and the parent molecule, 
    ensure that atoms double bonded to the ring are counted as inRing in the Boolean column."""
    # inRing = utils.get_molecules_atomsinrings(molecule)
    # notinring = []
    # inring = []
    # for atom in range(len(inRing)):
    #     if inRing[atom] == False:
    #         notinring.append(atom)
    #     else:
    #         inring.append(atom)
    mol_frame_subset = mol_frame.loc[mol_frame['inRing'] == False,:]
    #find the subset that have a label e.g. Ring 1, but not labeled inRing
    idx_to_update = list(mol_frame_subset.loc[mol_frame_subset['molPart'] != 'Unknown',:].index)
    mol_frame.loc[idx_to_update,['inRing']] = True #update the needed atoms to
    return mol_frame

def _find_in_ring_and_not(molecule):
    """given molecule, return np array of atoms in rings and not in rings"""
    in_ring_bool = utils.get_molecules_atomsinrings(molecule)
    not_in_ring = []
    in_ring = []
    for atom,atom_bool in enumerate(in_ring_bool):
        if not atom_bool:
            not_in_ring.append(atom)
        else:
            in_ring.append(atom)
    in_ring_py = np.array(in_ring)  #Generate list of atoms in rings
    check_atoms = np.array(not_in_ring) #We will check over the atoms not in rings
    return {'in_ring':in_ring_py, 'not_in_ring':check_atoms}


def _assign_side_and_linkers(mol_frame,molecule):
    """Given a mol_frame updated with Rings and the parent molecule, 
    assign the remaining atoms to side chains or linkers."""
    ring_assign = _find_in_ring_and_not(molecule)
    in_ring_py = ring_assign['in_ring']
    in_ring = in_ring_py.tolist()
    check_atoms = ring_assign['not_in_ring']
    not_in_ring = check_atoms.tolist()
    fgs=[]
    fg_type = []
    linker_count = 1 #For labeling Linker 1, Linker 2...start at 1
    periph_count=1 #see line above, but for side chains
    while check_atoms.size > 0: #while there are atoms left to check
        #initialize atoms in this group, starting with the first atom
        grp = _find_group(check_atoms,molecule,not_in_ring,in_ring_py)
        check_atoms=check_atoms[~np.isin(check_atoms,grp)] #remove atoms in grp from checkAtoms
        #at this point, we are done iterating over the set of connected atoms comprising the l
        # linker/peripheral
        #the next iteration, we will start at another atomt that we have not checked,
        # generate its connectivity and group, remove those etc.
        #counter number of atoms in the group that are bonded to rings,
        # if ==1, it is side chain, if ==2, it is linker
        at_ring = _count_rings_at_to_grp(grp,molecule,in_ring)
        if at_ring == 1:
            fg_type.append("Peripheral {count}".format(count=periph_count))
            periph_count+=1
        elif at_ring > 1:
            fg_type.append('Linker {count}'.format(count = linker_count))
            linker_count+=1
        fgs.append(list(grp))
    mol_frame = _assign_groups_to_frame(mol_frame,fgs,fg_type)
    return mol_frame

def _assign_groups_to_frame(mol_frame,fgs,fg_type):
    """given mol_frame and list of fg, assign the groups to mol_frame"""
    i=0
    while i < len(fgs): #update mol_frame with group parts
        for idx in fgs[i]:
            if mol_frame.loc[idx,'molPart'] == 'Unknown':
                mol_frame.loc[idx,'molPart'] = fg_type[i]
        i+=1
    return mol_frame

def _find_neighbors(atom):
    """given atom object a, find neighbours and return array of neighbour indices."""
    a_neigh = atom.GetNeighbors()
    #get the indices of the neighbours of ay in np array
    a_neigh_num = []
    for n_at in a_neigh:
        a_neigh_num.append(n_at.GetIdx())
    return np.array(a_neigh_num)

def _find_group(check_atoms, molecule, not_in_ring,in_ring_py):
    """Find a group of connected atoms."""
    grp = np.array([check_atoms[0]])
    #atom a is the first atom remaining in checkAtoms by index
    a_neigh_numpy = _find_neighbors(molecule.GetAtomWithIdx(int(check_atoms[0])))
    #Initialize newNeighbours to True, will be set to false in loop if an iteration
    # fails to look at an atom we ahve already seen
    new_neighbors=True
    while new_neighbors:
        #will be set to true if we find a new neighbour, and loop will continue to next
        #  iteration to check the neighbours of that atom
        new_neighbors = False
        for n_at in np.nditer(a_neigh_numpy): #iterate over the neighbours of atom a
            #only checking non-ring atoms as ring atoms won't be a side chain or a linker
            if n_at in not_in_ring:
                if n_at not in grp:
                    #if neighbour n is not in a ring, and not yet added to the group,
                    # add it to the group
                    grp = np.append(grp,n_at)
                n_neigh_numpy = _find_neighbors(molecule.GetAtomWithIdx(int(n_at)))
                #neighbours not in ring
                n_neigh_numpy = np.setdiff1d(n_neigh_numpy,in_ring_py)
                #not in ring and not in neighbours of a
                not_in_neigh = np.setdiff1d(n_neigh_numpy,a_neigh_numpy)
                #not in ring, not in neighbours of a and not in grp already
                not_in_neigh_gr = np.setdiff1d(not_in_neigh,grp)
                #if any remaining, we have new neighbours and continue iteration
                if not_in_neigh_gr.size > 0:
                    new_neighbors = True
                #find those not in group or ring to add
                not_in_gr = np.setdiff1d(n_neigh_numpy,grp)
                #add atoms to list we are iterating over so we check them too
                a_neigh_numpy = np.append(a_neigh_numpy,not_in_neigh_gr)
                #add the neighbours not in ring or not already in the group to the group
                grp = np.append(grp,not_in_gr)
    return grp

def _count_rings_at_to_grp(group,molecule,in_ring):
    """Return number of rings connected to a group in molecule."""
    at_ring=0
    for idx in group:
        neigh = molecule.GetAtomWithIdx(int(idx)).GetNeighbors()
        for n_at in neigh:
            if n_at.GetIdx() in in_ring:
                at_ring+=1
    return at_ring

def _generate_part_smiles(mol_frame,molecule):
    """Given complete mol_frame and molecule, generate SMILES for each 
    part(Ring/linker/peripheral), return as list."""
    fragments = mol_frame['molPart'].unique()
    frag_smi = []
    temp_smi = Chem.MolToSmiles(molecule)
    for frag in fragments:
        #print('Fragment is {frag}'.format(frag=frag))
        atoms_in_frag = list(mol_frame[mol_frame['molPart'] == frag].index)
        #print(mol_frame[mol_frame['molPart'] == frag])
        atoms_in_frag_and_at = []
        for atom in atoms_in_frag:
            atoms_in_frag_and_at.append(atom)
        #print(atomsInFrag)
        temp_mol = Chem.MolFromSmiles(temp_smi)
        #print(atomsInFrag)
        for atom in atoms_in_frag:
            #print('Frag atom {at}'.format(at=molecule.GetAtomWithIdx(at).GetIdx()))
            at_neigh = molecule.GetAtomWithIdx(atom).GetNeighbors()
            for an_at in at_neigh:
                an_idx = an_at.GetIdx()
                #print('Neighbour atom {an}'.format(an=anidx))
                if an_idx not in atoms_in_frag:
                    #print('not in fragment, updating to 0')
                    atoms_in_frag_and_at.append(an_idx)
                    #print(tempMol.GetAtomWithIdx(anidx).GetAtomicNum())
                    temp_mol.GetAtomWithIdx(an_idx).SetAtomicNum(0)
                    #print(tempMol.GetAtomWithIdx(anidx).GetAtomicNum())
        out_smi = Chem.MolFragmentToSmiles(temp_mol,atoms_in_frag_and_at)
        out_smi = utils.clean_smile(out_smi)
        frag_smi.append(out_smi)
    return frag_smi

def _find_alkyl_groups(mol_frame,frag_smi,molecule):
    """Given mol_frame(complete), list of molecule parts as SMILES, and a molecule object, 
    add all continuous alkyl groups to list of SMILES."""
    fragments = mol_frame['molPart'].unique()
    #Custom bond breaking - break single (-) bonds between carbons with 4 total attachments
    # (i.e. sp3) that are not in a ring ([C;X4;!R])
    #and atoms that are (ring atoms or not sp3 carbons) AND not atomic number 0
    # ([R,!$([C;X4]);!#0])
    #fragment 1 is the alkyl, fragment 2 is the remainder [*]-[*:1].[*]-[*:2]
    patt='[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0]):2]>>[*]-[*:1].[*]-[*:2]'
    alkyl_break_params = rdScaffoldNetwork.ScaffoldNetworkParams([patt])
    alkyl_break_params.includeGenericScaffolds=False
    alkyl_break_params.keepOnlyFirstFragment=False
    alkyl_break_params.includeScaffoldsWithoutAttachments=False
    alkyl_break_params.pruneBeforeFragmenting = False
    maybe_alkyl_smi=[]
    #for all molecule parts but rings, perform the fragmentation and generate smiles
    for part in fragments: #going over molecule parts again, this time looking for alkyl
        if 'Ring' not in part: #don't do this for ring systems
            atoms_in_part = list(mol_frame[mol_frame['molPart'] == part].index)
            maybe_alkyl_smi.append(Chem.MolFragmentToSmiles(molecule,atoms_in_part,canonical=True))
    maybe_alkyl_mol = [Chem.MolFromSmiles(x) for x in maybe_alkyl_smi]
    maybe_alkyl_net = rdScaffoldNetwork.CreateScaffoldNetwork(maybe_alkyl_mol,alkyl_break_params)
    maybe_alkyl_nodes = [Chem.MolFromSmiles(x) for x in maybe_alkyl_net.nodes]
    #check each node, and if all the atoms are placeholder or carbon, add it to fragments
    for node in maybe_alkyl_nodes:
        is_alkyl=1
        for atom in node.GetAtoms():
            if is_alkyl == 0:
                break
            if atom.GetAtomicNum() not in [0,6]:
                is_alkyl = 0
                continue
        if is_alkyl == 1 and '*' in Chem.MolToSmiles(node):
            frag_smi.append(Chem.MolToSmiles(node))
    return frag_smi

def _generate_fragment_frame(fragment_smiles):
    """Given list of SMILES Generate output frame with SMILES codes and molecules for unique 
    fragments."""
    frag_frame = pd.DataFrame(set(fragment_smiles),columns=['Smiles'])
    PandasTools.AddMoleculeColumnToFrame(frag_frame,'Smiles','Molecule',includeFingerprints=True)
    frag_frame.drop(frag_frame.index[frag_frame['Smiles'] == "*"].tolist())
    return frag_frame

def _add_ertl_functional_groups(frag_frame):
    """For each SMILES in frag_frame, identify heavy/double bonded atom functional groups and
    update the frame with them."""
    for molecule in frag_frame['Molecule']:
        if molecule.GetRingInfo().NumRings() == 0: #don't do this for rings
            fgs = ifg.identify_functional_groups(molecule)
            for fg_i in fgs: #iterate over identified functional groups
                if len(fg_i) != 0 and fg_i.atoms != '*':
                    #if one atom and it is a ring atom, skip
                    if len(fg_i.atomIds)==1 and molecule.GetAtomWithIdx(fg_i.atomIds[0]).IsInRing():
                        print("Skipping")
                        print(fg_i)
                    else:
                        frag_frame = _find_ertl_functional_group(fg_i,frag_frame)
    return frag_frame

def _find_ertl_functional_group(fg_i,frag_frame):
    """generate smiles and molecule object for ertl functional group."""
    #generate molecule of functional group
    just_grp = Chem.MolFromSmarts(fg_i.atoms)
    #generate molecule of functional group environment
    grp_and_att = Chem.MolFromSmiles(fg_i.type)
    if grp_and_att is not None: #if initialized successfully
        match_patt = grp_and_att.GetSubstructMatch(just_grp) #find the match
        if len(match_patt) >0:
            for atom in grp_and_att.GetAtoms():
                a_idx = atom.GetIdx()
                #set the atoms in grpAndAtt to 0 if they are not in the group
                if a_idx not in list(match_patt):
                    grp_and_att.GetAtomWithIdx(a_idx).SetAtomicNum(0)
            fg_smile = Chem.MolToSmiles(grp_and_att)
            flag=0
            for atom in grp_and_att.GetAtoms():
                if atom.GetAtomicNum() == 0 and atom.IsInRing():
                    flag = 1
            #append to frag_frame if placeholders not in ring
            if fg_smile not in frag_frame['Smiles'].unique():
                if fg_smile != len(fg_smile) * "*" and flag == 0:
                    fg_mol = Chem.MolFromSmiles(fg_smile)
                    frag_frame.loc[len(frag_frame)] = [fg_smile, fg_mol]
    return frag_frame

def _add_xyz_coords(frag_frame):
    """Given frag_frame with molecules, add xyz coordinates form MM94 optimization to it."""
    xyz_block_list = []
    query = rdqueries.AtomNumEqualsQueryAtom(0)
    for mol in frag_frame['Molecule']:
        h_mol_rw = Chem.RWMol(mol) # Change type of molecule object
        h_mol_rw = Chem.AddHs(h_mol_rw) # Add hydrogens
        zero_at = h_mol_rw.GetAtomsMatchingQuery(query)   #Replace placeholder * with At
        for atom in zero_at:
            h_mol_rw.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(85)
        AllChem.EmbedMolecule(h_mol_rw)
        AllChem.MMFFOptimizeMolecule(h_mol_rw) #Optimize with MMFF94
        xyz_block_list.append(AllChem.rdmolfiles.MolToXYZBlock(h_mol_rw)) #Store xyz coordinates
    frag_frame['xyz'] = xyz_block_list
    return frag_frame

def _add_number_attachements(frag_frame):
    """Add number of attachments column to frag_frame, counting number of *."""
    attach_list = []
    for molecule in frag_frame['Molecule']:
        attach=0
        atoms = molecule.GetAtoms()
        for atom in atoms:
            if atom.GetAtomicNum() == 0:
                attach += 1
        attach_list.append(attach)
    frag_frame['numAttachments'] = attach_list
    return frag_frame

def generate_full_mol_frame(mol1) -> pd.DataFrame:
    """Generate data frame for molecule assigning all atoms to rings/linkers/peripherals."""
    mol1nodemols = utils.get_scaffold_vertices(mol1)
    ring_frags = utils.find_smallest_rings(mol1nodemols)
    mol_frame = _initialize_molecule_frame(mol1)
    ring_atom_indices = _identify_ring_atom_index(mol1,ring_frags)
    ring_indices_nosubset = _remove_subset_rings(ring_atom_indices)
    mol_frame = _assign_rings_to_mol_frame(ring_indices_nosubset,mol_frame)
    mol_frame = _set_double_bonded_in_ring(mol_frame)
    mol_frame = _assign_side_and_linkers(mol_frame,mol1)
    return mol_frame


def identify_fragments(smile: str) -> pd.DataFrame:
    """Perform the full fragmentation.
    This will break down molecule into unique rings, linkers, functional groups, peripherals, 
    but not maintain connectivity.
    Atoms may be assigned to multiple groups. eg. C(=O)C, C=O, 
    C may all be SMILES included from an acetyl peripheral
    """
    mol = utils.get_canonical_molecule(smile)
    mol_frame = generate_full_mol_frame(mol)
    #print(mol_frame)
    frag_smi = _generate_part_smiles(mol_frame,molecule=mol)
    frag_smi = _find_alkyl_groups(mol_frame,frag_smi,mol)
    #print(fragSmi)
    frag_frame = _generate_fragment_frame(frag_smi)
    frag_frame = _add_ertl_functional_groups(frag_frame)
    frag_frame = _add_xyz_coords(frag_frame)
    frag_frame = _add_number_attachements(frag_frame)
    #print(frag_frame)
    return frag_frame


def _trim_molpart(mol_frame,mol_part_lst,molecule):
    """Given mol_frame, and unique parts in mol_frame, and the molecule, 
    break molecule into the unique parts."""
    #will return with connections set to count:*, not labeled
    count=1
    bonds = []
    labels = []
    for mol_part in mol_part_lst:
        atoms_in_mol_part = mol_frame.loc[mol_frame['molPart'] == mol_part,:].index
        for atom in atoms_in_mol_part:
            a_idx = molecule.GetAtomWithIdx(atom).GetIdx()
            for neigh in molecule.GetAtomWithIdx(atom).GetNeighbors():
                n_idx = neigh.GetIdx()
                if n_idx not in atoms_in_mol_part:
                    b = molecule.GetBondBetweenAtoms(a_idx,n_idx)
                    if b.GetIdx() not in bonds:
                        bonds.append(b.GetIdx())
                        labels.append([count,count])
                        count  = count+1
    frag_mol = Chem.FragmentOnBonds(molecule,bondIndices=bonds,dummyLabels=labels)
    #remove molecule from frag_frame, add fragments
    split_smiles  = Chem.MolToSmiles(frag_mol).split('.')
    new_smi=[]
    for split in split_smiles:
        new_smi.append(Chem.MolToSmiles(Chem.MolFromSmiles(split)))
    return {'smiles':new_smi, 'count':count}


def _break_molparts(mol_part_smi,count,drop_parent = True,
                    patt = '[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0;!#9;!#17;!#35]):2]'):
    """For a given list of Smiles of the molecule parts, break non-ring groups into 
    Ertl functional groups and alkyl groups."""
    el_to_rm = []
    new_smi=[]
    for i,partsmi in enumerate(mol_part_smi):
        molecule = Chem.MolFromSmiles(partsmi)
        if molecule.GetRingInfo().NumRings() == 0: #don't do this for rings
            #break sp3 carbon to ring/heteroatom bonds
            bonds_to_break = molecule.GetSubstructMatches(Chem.MolFromSmarts(patt))
            bonds = []
            labels = []
            #iterate over matches, storing bond index, and the dummy atom labels to be used in a
            # list
            for bond in bonds_to_break:
                b_obj = molecule.GetBondBetweenAtoms(bond[0],bond[1])
                bonds.append(b_obj.GetIdx())
                labels.append([count,count])
                #all dummy atom labels will be different for different bond breaking
                count  = count+1
            if bonds:
                el_to_rm.append(i)
                frag_mol = Chem.FragmentOnBonds(molecule,bondIndices=bonds,dummyLabels=labels)
                #FragmentOnBonds returns SMILES with fragments separated by .,
                #  get each fragment in its own string in a list
                split_smiles  = Chem.MolToSmiles(frag_mol).split('.')
                for split in split_smiles:
                    #store canonical fragment smiles in new list
                    new_smi.append(Chem.MolToSmiles(Chem.MolFromSmiles(split)))
    if drop_parent:
        el_to_rm = sorted(el_to_rm,reverse=True)
        for rm_el in el_to_rm:
            #if the molPart was broken apart, remove it from the output
            # so each atom is uniquely assigned
            del mol_part_smi[rm_el]
    out_smi  = mol_part_smi + new_smi #this is the Smiles of all fragments in the molecule
    return out_smi

def generate_acylic_mol_frame(molecule):
    atom_nums = utils.get_molecules_atomicnum(molecule)
    false_list = [False] * len(atom_nums)
    mol_part = ['Acyclic'] * len(atom_nums)
    initialization_data = {'atomNum': atom_nums,
                           'inRing': false_list, 
                           'molPart': mol_part}
    return pd.DataFrame(initialization_data)

def identify_connected_fragments(smile: str,keep_only_children:bool=True,
            bb_patt:str='[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0;!#9;!#17;!#35]):2]') -> pd.DataFrame:
    """
    Given Smiles string, identify fragments in the molecule as follows:
    Break all ring-non-ring atom single bonds
    Store the resulting fragments
    For non-ring fragments, separate those into alkyl chains and hetero/double bonded atoms 
    (similar to Ertl functional groups)
    Each bond breaking, connectivity is maintained through dummy atom labels.
    e.g. C-N -> C-[1*] N-[1*] - reattaching via the matching labels would reassemble the molecule
    
    Args:
        smile: a string containing smiles for a given molecule, does not need to be canonical
        keep_only_children: boolean, if True, when a group is broken down into its components
            remove the parent group from output. If False, parent group is retained
        bb_patt: string of SMARTS pattern for bonds to be broken in side chains and linkers
            defaults to cleaving sp3 carbon-((ring OR not sp3 carbon) AND not-placeholder/halogen)
    Returns:
        pandas data frame with columms 'Smiles', 'Molecule', 'numAttachments' and 'xyz'
        Containing, fragment smiles, fragment Chem.Molecule object, number of * placeholders,
          and rough xyz coordinates for the fragment is * were At
    Notes: currently will break apart a functional group if contains a ring-non-ring single bond.
    e.g. ring N-nonring S=O -> ring N-[1*] nonring S=O-[1*]    
    """
    #ensure smiles is canonical so writing and reading the smiles will result in same number
    # ordering of atoms
    mol = utils.get_canonical_molecule(smile)
    #assign molecule into parts (Rings, side chains, peripherals)
    if mol.GetRingInfo().NumRings() > 0:
        mol_frame = generate_full_mol_frame(mol)
        fragment_smiles = _trim_molpart(mol_frame,mol_frame['molPart'].unique(),mol)
    else:
        mol_frame = generate_acylic_mol_frame(mol)
        fragment_smiles = {'smiles':[Chem.MolToSmiles(mol)],'count':1}
    #break molecule into fragments defined by the unique parts in mol_frame (Ring 1, Peripheral 1,
    #  Linker 1, Linker 2, etc.)
    
    #break side chains and linkers into Ertl functional groups and alkyl chains
    full_smi = _break_molparts(fragment_smiles['smiles'],fragment_smiles['count']
                               ,drop_parent=keep_only_children,patt=bb_patt)
    #initialize the output data frame
    frag_frame = _generate_fragment_frame(full_smi)
    #add hydrogens and xyz coordinates resulting from MMFF94 opt, changing placeholders to At
    # frag_frame = _add_xyz_coords(frag_frame)
    #count number of placeholders in each fragment - it is the number of places it is attached
    frag_frame = _add_number_attachements(frag_frame)
    return frag_frame

def count_uniques(frag_frame:pd.DataFrame,drop_attachments=False) -> pd.DataFrame:
    """
    Given frag_frame resulting from identify_connected_fragments, remove dummy atom labels
    (and placeholders entirely if drop_attachments=True)
    Then, compare the Smiles to count the unique fragments, and return a version of frag_frame 
    that only includes unique fragments
    and the number of times each unique fragment occurs.

    Args:
        frag_frame: frame resulting from identify_connected_fragments typically, 
        or any similar frame with a list of SMILES codes in column ['Smiles']
        drop_attachments: boolean, if False, retains placeholder * at points of attachment, 
        if True, removes * for fragments with more than one atom

    Returns:
        pandas data frame with columns 'Smiles', 'count' and 'Molecule', 
        containing the Smiles string, the number of times the Smiles was in frag_frame,
          and rdkit.Chem.Molecule object    

    Notes: if drop_attachments=False, similar fragments with different number/positions of 
    attachments will not count as being the same.
    e.g. ortho-attached aromatics would not match with meta or para attached aromatics        
    """
    smile_list = frag_frame['Smiles']
    no_connect_smile=[]
    for smile in smile_list:
        if drop_attachments:
            no_connect_smile.append(_drop_smi_attach(smile))
        else:
            no_connect_smile.append(Chem.MolToSmiles(Chem.MolFromSmiles(re.sub('\[[0-9]+\*\]',
                                                                             '*', smile))))
    #identify unique smiles and count number of times they occur
    unique_smiles=[]
    unique_smiles_counts=[]
    for smile in no_connect_smile:
        if smile not in unique_smiles:
            unique_smiles.append(smile)
            unique_smiles_counts.append(1)
        else:
            smi_ix = unique_smiles.index(smile)
            unique_smiles_counts[smi_ix] += 1
    #create output frame
    return _construct_unique_frame(uni_smi=unique_smiles,uni_smi_count=unique_smiles_counts)
    # uniquefrag_frame = pd.DataFrame(unique_smiles,columns=['Smiles'])
    # PandasTools.AddMoleculeColumnToFrame(uniquefrag_frame,'Smiles','Molecule',
    #                                      includeFingerprints=True)
    # uniquefrag_frame['count']=unique_smiles_counts
    # uniquefrag_frame = _add_xyz_coords(uniquefrag_frame)
    # uniquefrag_frame = _add_number_attachements(uniquefrag_frame)
    # return uniquefrag_frame

def _construct_unique_frame(uni_smi:list[str],uni_smi_count:list[int]) -> pd.DataFrame:
    uniquefrag_frame = pd.DataFrame(uni_smi,columns=['Smiles'])
    PandasTools.AddMoleculeColumnToFrame(uniquefrag_frame,'Smiles','Molecule',
                                         includeFingerprints=True)
    uniquefrag_frame['count']=uni_smi_count
    uniquefrag_frame = _add_xyz_coords(uniquefrag_frame)
    uniquefrag_frame = _add_number_attachements(uniquefrag_frame)
    return uniquefrag_frame

def _drop_smi_attach(smile:str):
    mol = Chem.MolFromSmiles(smile)
    non_zero_atoms=0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            non_zero_atoms += 1
    if non_zero_atoms > 1:
        temp= re.sub('\[[0-9]+\*\]', '', smile)
        t_mol = Chem.MolFromSmiles(re.sub('\(\)', '', temp))
        if t_mol is None:
            t_mol=Chem.MolFromSmiles(re.sub('\[[0-9]+\*\]','*', smile))
            out_smi = Chem.MolToSmiles(t_mol)
            #Warning('Could not construct {smile} without attachments'.format(smile=smile))
        else:
            out_smi = Chem.MolToSmiles(t_mol)
    else:
        out_smi = Chem.MolToSmiles(Chem.MolFromSmiles(re.sub('\[[0-9]+\*\]','*', smile)))
    return out_smi

def merge_uniques(frame1:pd.DataFrame,frame2:pd.DataFrame) -> pd.DataFrame:
    """Given two frames of unique fragments, identify shared unique fragments,
    merge count and frames together.
    
    Args:
        frame1: a frame output from count_uniques
        frame2: a distinct frame also from count_uniques

    Returns:
        a frame resulting from the merge of frame1 and frame2. 
        All rows that have Smiles that are in frame1 but not frame2(and vice versa) are included 
        unmodified
        If a row's SMILES is in both frame1 and frame2, modify the row to update the count of 
        that fragment as sum of frame1 and frame2, then include one row.

    Note:
        for best results, SMILES must be canonical so that they can be exactly compared.
        Smiles in frame should be resulting from Chem.MolToSmiles(Chem.MolFromSmiles(smile)) -
        this will create a molecule from the smile, and write the smile back, in canonical form    

    Example usage:
        frame1:
        Smiles  count
        C       2
        C1CCC1  1

        frame2:
        Smiles  count
        C       3
        C1CC1   2

        >>> merge_uniques(frame1,frame2)
        Smiles  count
        C       5
        C1CCC1  1
        C1CC1   2
    """
    rows_to_drop = _find_rows_to_drop(frame1,frame2)
    merge_frame = rows_to_drop['merge_frame']
    drop_frame_1 = frame1.drop(rows_to_drop['drop_rows_1'])
    drop_frame_2 = frame2.drop(rows_to_drop['drop_rows_2'])
    merge_frame = _add_frame(drop_frame_1,merge_frame)
    merge_frame = _add_frame(drop_frame_2,merge_frame)
    return merge_frame

def _add_frame(frame1:pd.DataFrame,merge_frame=pd.DataFrame()) -> pd.DataFrame:
    if merge_frame.empty:
        made_frame=0
    else:
        made_frame=1    
    for i,smi in enumerate(frame1['Smiles']):
        if made_frame == 0:
            merge_frame = pd.DataFrame.from_dict({'Smiles':[smi],
                                                 'count':list(frame1['count'])[i]})
            made_frame=1
        else:
            merge_frame.loc[len(merge_frame)] = [smi, list(frame1['count'])[i]]
    if merge_frame.empty:
        merge_frame = pd.DataFrame()        
    return merge_frame

def _find_rows_to_drop(frame_a:pd.DataFrame,frame_b:pd.DataFrame) -> list[list[int]]:
    rows_to_drop_one = []
    rows_to_drop_two = []
    made_frame = 0
    merge_frame = pd.DataFrame()
    for i,smi in enumerate(frame_a['Smiles']):
        if smi in list(frame_b['Smiles']):
            j=list(frame_b['Smiles']).index(smi)
            cum_count=frame_a['count'][i] + frame_b['count'][j]
            if made_frame == 0:
                merge_frame = pd.DataFrame.from_dict({'Smiles':[smi],'count':[cum_count]})
                made_frame=1
            else:
                merge_frame.loc[len(merge_frame)] = [smi, cum_count]
            rows_to_drop_one.append(i)
            rows_to_drop_two.append(j)
    return {'drop_rows_1':rows_to_drop_one,'drop_rows_2':rows_to_drop_two,
            'merge_frame':merge_frame}


def count_groups_in_set(list_of_smiles:list[str],drop_attachments:bool=False) -> pd.DataFrame:
    """Identify unique fragments in molecules defined in the list_of_smiles, 
    and count the number of occurences for duplicates.
    Args:
        list_of_smiles: A list, with each element being a SMILES string, e.g. ['CC','C1CCCC1']
        drop_attachments: Boolean for whether or not to drop attachment points from fragments
            if True, will remove all placeholder atoms indicating connectivity
            if False, placeholder atoms will remain
    Returns:
        an output pd.DataFrame, with columns 'Smiles' for fragment Smiles, 
        'count' for number of times each fragment occurs in the list, and 
        'Molecule' holding a rdkit.Chem.Molecule object
        
    Example usage:
        count_groups_in_set(['c1ccc(c(c1)c2ccc(o2)C(=O)N3C[C@H](C4(C3)CC[NH2+]CC4)C(=O)NCCOCCO)F',
        'Cc1nc2ccc(cc2s1)NC(=O)c3cc(ccc3N4CCCC4)S(=O)(=O)N5CCOCC5'],drop_attachments=False)."""
    i=0
    for smile in list_of_smiles:
        frame = identify_connected_fragments(smile)
        unique_frame = count_uniques(frame,drop_attachments)
        if i==0:
            i+=1
            out_frame=unique_frame
        else:
            out_frame = merge_uniques(out_frame,unique_frame)
    PandasTools.AddMoleculeColumnToFrame(out_frame,'Smiles','Molecule',includeFingerprints=True)
    #out_frame = _add_xyz_coords(out_frame)
    out_frame = _add_number_attachements(out_frame)
    return out_frame
