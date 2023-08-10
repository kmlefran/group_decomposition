"""
fragfunctions module

code used to generate fragments of molecules from SMILES code and analyze them

Main functions to call are:
identify_connected_fragments - takes one molecule SMILES, returns fragments with connections
count_uniques - takes output from above, removes attachments and counts unique fragments
count_groups_in_set - takes list of SMILES and counts unique fragments on set
"""
import sys
import os
import re
import rdkit
from rdkit import Chem
import math
from rdkit.Chem import AllChem, PandasTools, rdqueries #used for 3d coordinates
from rdkit.Chem.Scaffolds import rdScaffoldNetwork # scaffolding
import pandas as pd #lots of work with data frames
import numpy as np #for arrays in fragment identification
sys.path.append(sys.path[0].replace('/src',''))
from group_decomposition import utils

_H_BOND_LENGTHS = {
    #from Gaussview default cleaned bond length
    'C':1.07,
    'O':0.96,
    'N':1.00,
    'F':0.88,
    'Cl':1.29,
    'B':1.18,
    'Al':1.55,
    'Si':1.47,
    'P':1.35,
    'S':1.31
}

def _initialize_molecule_frame(molecule, xyz_coords=[]):
    """Given a molecule, assign create frame with atomic numbers, Boolean of if in ring
    and unknown column

    """
    atomic_numbers = utils.get_molecules_atomicnum(molecule)
    atoms_in_rings = utils.get_molecules_atomsinrings(molecule)
    if xyz_coords:
        initialization_data = {'atomNum': atomic_numbers,
                           'inRing': atoms_in_rings, 
                           'molPart': ['Unknown'] * molecule.GetNumAtoms(),
                           'xyz':xyz_coords}
    else:    
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

def generate_full_mol_frame(mol1,xyz_coords=[]) -> pd.DataFrame:
    """Generate data frame for molecule assigning all atoms to rings/linkers/peripherals."""
    mol1nodemols = utils.get_scaffold_vertices(mol1)
    ring_frags = utils.find_smallest_rings(mol1nodemols)
    mol_frame = _initialize_molecule_frame(mol1,xyz_coords)
    ring_atom_indices = _identify_ring_atom_index(mol1,ring_frags)
    ring_indices_nosubset = _remove_subset_rings(ring_atom_indices)
    mol_frame = _assign_rings_to_mol_frame(ring_indices_nosubset,mol_frame)
    mol_frame = _set_double_bonded_in_ring(mol_frame)
    mol_frame = _set_hydrogens_in_ring(mol_frame,mol1)
    mol_frame = _assign_side_and_linkers(mol_frame,mol1)
    return mol_frame

def _set_hydrogens_in_ring(mol_frame,mol):
    """Include hydrogens in rings they are bonded to in mol_frame"""
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            nb_at = atom.GetNeighbors()[0]
            if nb_at.IsInRing():
                at_ring = mol_frame['molPart'][nb_at.GetIdx()]
                #If H bonded to Ring 1, will set molPart as Ring 1
                mol_frame.loc[idx,['molPart']] = at_ring 
    return mol_frame            


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
    # frag_mol = utils.mol_with_atom_index(frag_mol)
    #remove molecule from frag_frame, add fragments
    split_smiles  = Chem.MolToSmiles(frag_mol).split('.')
    new_smi=[]
    for split in split_smiles:
        new_smi.append(Chem.MolToSmiles(Chem.MolFromSmiles(split)))
    return {'smiles':new_smi, 'count':count}


def _break_molparts(mol_part_smi,count,drop_parent = True,
                    patt = '[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0;!#9;!#17;!#35,!#1]):2]'):
    """For a given list of Smiles of the molecule parts, break non-ring groups into 
    Ertl functional groups and (halo)alkyl groups."""
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

def generate_acyclic_mol_frame(molecule, xyz_coords=[]):
    """Create simple mol_frame, with molPart as Acyclic and all inRing=False"""
    atom_nums = utils.get_molecules_atomicnum(molecule)
    false_list = [False] * len(atom_nums)
    mol_part = ['Acyclic'] * len(atom_nums)
    if xyz_coords:
        initialization_data = {'atomNum': atom_nums,
                           'inRing': false_list, 
                           'molPart': mol_part,
                           'xyz':xyz_coords}
    else:
        initialization_data = {'atomNum': atom_nums,
                           'inRing': false_list, 
                           'molPart': mol_part}
    return pd.DataFrame(initialization_data)

def csv_frag_generation(csv_file):
    """Untested"""
    inp_frame = pd.read_csv(csv_file)
    out_frame = pd.DataFrame()
    for i in range(len(inp_frame.index)):
        mol_file = inp_frame['MolFile'][i]
        cml_file = inp_frame['CmlFile'][i]
        temp_frame = identify_connected_fragments(mol_file,input_type='molfile',
                                                  cml_file=cml_file,include_parent=True)
        temp_unique_frame = count_uniques(temp_frame)
        out_frame = merge_uniques(out_frame,temp_unique_frame)
    parent_mols = out_frame['Parent'].unique()
    for mol in parent_mols:
        output_ifc_gjf(mol,out_frame[[out_frame['Parent']==mol]])
    return out_frame

def identify_connected_fragments(input: str,keep_only_children:bool=True,
            bb_patt:str='[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0;!#9;!#17;!#35;!#1]):2]',
            input_type = 'smile',cml_file='',include_parent=False) -> pd.DataFrame:
    """
    Given Smiles string, identify fragments in the molecule as follows:
    Break all ring-non-ring atom single bonds
    Store the resulting fragments
    For non-ring fragments, separate those into alkyl chains and hetero/double bonded atoms 
    (similar to Ertl functional groups)
    Each bond breaking, connectivity is maintained through dummy atom labels.
    e.g. C-N -> C-[1*] N-[1*] - reattaching via the matching labels would reassemble the molecule
    
    Args:
        input: a string containing either a smiles or a .mol filename for a given molecule
            update input_type below to match provided input
        keep_only_children: boolean, if True, when a group is broken down into its components
            remove the parent group from output. If False, parent group is retained
        bb_patt: string of SMARTS pattern for bonds to be broken in side chains and linkers
            defaults to cleaving sp3 carbon-((ring OR not sp3 carbon) AND not-placeholder/halogen/H)
        input_type = 'smile' if SMILES code or 'molfile' if .mol file, or 'xyzfile' if .xyz file
            Note: xyz file REQUIRES .cml file as well
        cml_file: defaults to none, can be the cml file corresponding to the input .mol file
        include_parent = Boolean. If True, include column in output frame repeating parent molecule
            intended use for True when merging multiple molecule fragment frames but need to retain a parent molecule object
    Returns:
        pandas data frame with columms 'Smiles', 'Molecule', 'numAttachments' and 'xyz'
        Containing, fragment smiles, fragment Chem.Molecule object, number of * placeholders,
          and rough xyz coordinates for the fragment is * were At
    Notes: currently will break apart a functional group if contains a ring-non-ring single bond.
    e.g. ring N-nonring S=O -> ring N-[1*] nonring S=O-[1*]    
    """
    #ensure smiles is canonical so writing and reading the smiles will result in same number
    # ordering of atoms
    if input_type == 'smile':
        mol = utils.get_canonical_molecule(input)
        xyz_coords=[]
    elif input_type == 'molfile':
        mol_dict = utils.mol_from_molfile(input)
        mol, atomic_symb = mol_dict['Molecule'],  mol_dict['atomic_symbols']
        #use coordinates in cml file provided if able, else use xyz from mol file
        if cml_file:
            xyz_coords = utils.xyz_from_cml(cml_file)
        else:
            xyz_coords =mol_dict['xyz_pos']
    elif input_type == 'xyzfile':
        if not cml_file:
            raise ValueError('No cml file provided, expected one for xyz input type')
        mol_dict = utils.mol_from_xyzfile(xyz_file=input,cml_file=cml_file)
        mol, atomic_symb, xyz_coords = mol_dict['Molecule'],  mol_dict['atomic_symbols'], mol_dict['xyz_pos']
    else:
        raise ValueError(f"""{input_type} should either be molfile, xyzfile, or a smile string""")    
    #assign molecule into parts (Rings, side chains, peripherals)
    if mol.GetRingInfo().NumRings() > 0:
        mol_frame = generate_full_mol_frame(mol,xyz_coords)
        fragment_smiles = _trim_molpart(mol_frame,mol_frame['molPart'].unique(),mol)
    else:
        mol_frame = generate_acyclic_mol_frame(mol,xyz_coords)
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
    if input_type == 'molfile' or input_type == 'xyzfile': #clear map labels and add xyz coordinates that are available
        frag_frame = _add_frag_comp(frag_frame,mol)
        frag_frame['Smiles'] = frag_frame['Smiles'].map(lambda x:_clear_map_number(x))
        frag_frame['xyz'] = frag_frame['Atoms'].map(lambda x:_add_rtr_xyz(x,xyz_coords))
        frag_frame['Labels'] = frag_frame['Atoms'].map(lambda x:_add_rtr_label(x,atomic_symb))
        frag_frame['Molecule'] = frag_frame['Molecule'].map(lambda x:_clear_map_number(x,'mol'))    
    if include_parent:
        frag_frame['Parent'] = [mol] * len(frag_frame.index)
    return frag_frame

def _add_rtr_label(at_num_list,atomic_symb):
    out_list = []
    for atom in at_num_list:
        # out_str + atomic_symbols[atom-1] + '  ' + str(xyz_coords[atom-1][0]) + '  ' + str(xyz_coords[atom-1][1]) + '  ' +  str(xyz_coords[atom-1][2]) + '\n'
        out_list.append(atomic_symb[atom-1])
    return out_list

def _add_rtr_xyz(at_num_list,xyz_coords):
    """Construct string of atomSymbol x y z format, for use to map to frag_frame"""
    out_list = []
    for atom in at_num_list:
        # out_str + atomic_symbols[atom-1] + '  ' + str(xyz_coords[atom-1][0]) + '  ' + str(xyz_coords[atom-1][1]) + '  ' +  str(xyz_coords[atom-1][2]) + '\n'
        out_list.append(xyz_coords[atom-1])
    return out_list

def _clear_map_number(mol_input, ret_type='smi'):
    """Given str or Chem.Molecule input, remove atomMapnumbers from atoms and return 
    smiles (ret_type='smi') or molecule object (ret_type = 'mol')"""
    if type(mol_input) == str:
        mol = Chem.MolFromSmiles(mol_input)
    else:
        mol = mol_input    
    if not mol:
        raise ValueError(f"""Could not construct mol from {mol_input} in output frame""")
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    if ret_type == 'smi':
        return Chem.MolToSmiles(mol)
    elif ret_type == 'mol':
        return mol
    else:
        raise ValueError(f"""Invalid ret_type, expected smi or mol, got {ret_type}""")

def _get_atlabels_in_frag(molecule):
    """For a given molecule, extract list of atom map number for all non-H atoms"""
    #H's not included because the molecule object typically won't have explicit H
    out_list = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 0:
            out_list.append(int(atom.GetProp('molAtomMapNumber')))
    return out_list

def _add_frag_comp(frag_frame,mol):
    """Given frag_frame and mol, add Atoms col to frag_frame with indices of atoms starting at 1"""
    #create list of lists of indices of atoms in each fragment
    frag_atoms = []
    for frag_mol in frag_frame['Molecule']:
        frag_atoms.append(_get_atlabels_in_frag(frag_mol))
    #iterate over atoms, adding in hydrogens to the fragment since the above won't include H
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            neigh_idx = atom.GetNeighbors()[0].GetIdx() + 1
            for i, frag in enumerate(frag_atoms):
                if neigh_idx in frag:
                    frag_atoms[i].append(atom.GetIdx()+1)
                    break #only need to get here once - H has only one bond
    frag_frame['Atoms'] = frag_atoms
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
    col_names = list(frag_frame.columns)
    #if frag_frame already has xyz coordinates keep those and don't use others
    #typically this will be if a mol and/or cml file was provided in frag_frame construction
    if 'xyz' in col_names:
        xyz_inc=True
        xyz_list = frag_frame['xyz']
    else:
        xyz_inc = False    
    smile_list = frag_frame['Smiles']
    if 'Atoms' in col_names:
        atom_list = frag_frame['Atoms']
        atoms_inc = True
    else:
        atoms_inc = False
    if 'Labels' in col_names:
        labels_list = frag_frame['Labels']
        labels_inc = True
    else:
        labels_inc = False
    if 'Parent' in col_names:
        parent_list = frag_frame['Parent']
        parent_inc = True
    else:
        parent_inc = False
    no_connect_smile=[]
    #Clean smiles - either removing placeholder entirely(drop_attachments True)
    # Or just removing the dummyAtomLabel (drop_attachments False)
    for smile in smile_list:
        if drop_attachments:
            no_connect_smile.append(_drop_smi_attach(smile))
        else:
            t_mol = Chem.MolFromSmiles(re.sub('\[[0-9]+\*\]', '*', smile))
            no_connect_smile.append(Chem.MolToSmiles(t_mol))
    #identify unique smiles and count number of times they occur
    #initialize lists to be used making frame
    unique_smiles=[]
    unique_xyz = []
    unique_atoms = []
    unique_labels = []
    unique_parents = []
    unique_smiles_counts=[]
    #Identify unique smiles, counting every occurence and adding xyz if included
    for i,smile in enumerate(no_connect_smile):
        if smile not in unique_smiles:
            unique_smiles.append(smile)
            if xyz_inc:
                unique_xyz.append(xyz_list[i])
            if atoms_inc:
                unique_atoms.append(atom_list[i])
            if labels_inc:
                unique_labels.append(labels_list[i])
            if parent_inc:
                unique_parents.append(parent_list[i])
            unique_smiles_counts.append(1)
        else:
            smi_ix = unique_smiles.index(smile)
            unique_smiles_counts[smi_ix] += 1
    #create output frame
    
    un_frame =  _construct_unique_frame(uni_smi=unique_smiles,uni_smi_count=unique_smiles_counts,xyz=unique_xyz,atoms=unique_atoms,parents=unique_parents,labels=unique_labels)
    if atoms_inc:
        un_frame['Atoms'] = unique_atoms
    if labels_inc:
        un_frame['Labels'] = unique_labels
    if parent_inc:
        un_frame['Parent'] = unique_parents
    return un_frame

def _find_neigh_notin_frag(mol,at_list):
    #for atoms with one attachment point
    out_nbr=0
    for idx in at_list:
        atom = mol.GetAtomWithIdx(idx-1)
        nbrs = atom.GetNeighbors()
        for nbr in nbrs:
            nbr_idx = nbr.GetIdx()+1
            if nbr_idx not in at_list:
                out_nbr = nbr_idx
                out_at = atom.GetIdx()+1
                break
        if out_nbr:
            break
    return [out_at, out_nbr]

def _find_neigh_xyz(frag_frame,neigh_idx):
    """Given frag_frame and index of neighbor atom, find its xyz coordinates"""
    #column index
    atoms_idx = list(frag_frame.columns).index('Atoms')
    #determine which row contains the neighbor atom
    neigh_bool = list(frag_frame.apply(lambda row: neigh_idx in row[atoms_idx],axis=1))
    # print(neigh_bool)
    neigh_row = neigh_bool.index(True)
    list_xyz = frag_frame['xyz'][neigh_row]
    #find where in list neighbor atom is, return xyz at that index
    neigh_list_idx = frag_frame['Atoms'][neigh_row].index(neigh_idx)
    neigh_xyz = list_xyz[neigh_list_idx]
    return neigh_xyz

def _move_along_bond(at_xyz,neigh_xyz,at_symb):
    """Returns xyz coordinates of neigh_xyz moved along bond to H bond length"""
    at_np = np.array(at_xyz)
    nb_np = np.array(neigh_xyz)
    #vector for bond to move along
    bond = at_np-nb_np
    #bond length to find
    target_length = _H_BOND_LENGTHS[at_symb]
    steps = np.linspace(0.,1.,100)
    #step along bond starting at neigh_xyz in direction of at_xyz.
    #stop when bond length is target
    for s in steps:
        new_xyz = neigh_xyz + s * bond
        if abs(_get_dist(new_xyz,at_np) - target_length) < 0.01:
            end_xyz = new_xyz
    #return target point
    return end_xyz

def _get_dist(point_a,point_b):
    """Return distance btw two points"""
    return math.sqrt((point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2 + (point_a[2]-point_b[2])**2)

def _find_H_xyz(mol,at_list,xyz_list,frag_frame):
    """Finds xyz of hydrogen atom that would be connected to a fragment
    
    Args:
        mol: Chem.Mol object
        at_list: list[int] - list of atoms in the fragment
            Note: these start at 1, but those in molecule start at 0.
            To convert add/subtract 1
        xyz_list: list of xyz_coordinates of atoms in fragment
        frag_frame: full parent frag_frame WITHOUT filtering by number of attachments
    
    Returns:
        list[float]: xyz coordinates where H should be placed
    """
    #find attached atom index and neighbor index in molecule
    at_idx, neigh_idx = _find_neigh_notin_frag(mol,at_list)
    #get attached atom symbol
    symb = mol.GetAtomWithIdx(at_idx-1).GetSymbol()
    #index of the attached atom in at_list, which is the same as xyz_list
    list_idx = at_list.index(at_idx)
    #attached atom xyz
    at_xyz = xyz_list[list_idx]
    # neighbor atom xyz
    neigh_xyz = _find_neigh_xyz(frag_frame,neigh_idx)
    # move neighbor atom xyz along bond to H bond length (Gaussview defaults)
    h_xyz = _move_along_bond(at_xyz,neigh_xyz,symb)
    return h_xyz

def _clean_molecule_name(smile):
    """Removes symbols in smile code unfit for file name"""
    smile = smile.replace('-','Neg')
    smile = smile.replace('[','-')
    smile = smile.replace(']','-')
    smile = smile.replace('(','-')
    smile = smile.replace(')','-')
    smile = smile.replace('#','t')
    smile = smile.replace('=','d')
    smile = smile.replace('+','Pos')
    smile = smile.replace('*','Att')
    smile = smile.replace('@','')
    return smile

def output_ifc_gjf(mol,frag_frame,esm='wb97xd',basis_set='aug-cc-pvtz',wfx=True,n_procs=4,mem='3200MB',multiplicity=1):
    """ Takes a fragmented molecule and outputs gjf files of the fragments with one
    attachment point. Hydrogen is added in place of the connection to the rest of
    the molecule for the fragment

    Args:
        mol: Chem.Mol object for which fragmentation was performed
        frag_frame: output from either count_uniques or identify_connected_fragments
        esm: str, electronic structure method to include in gjf
        basis_set: str, basis set to include in gjf
        wfx: Boolean, if True add output=wfx to gjf file
        n_procs: int, >=0. if >0, add number of processors to be used to gjf
        mem: str, format "nMB" or "nGB", memory to be used in gjf
        multiplicity: int, defaults to 1. Multiplicity of molecule
    
    Returns:
        Creates gjf files in working directory for each fragment in frag_frame
    
    Notes:
        H position is set by taking the atom the fragment is bonded two, replacing it with H
          and moving that closer to the C until it reaches the default distance
        Default distances taken from Gaussview "clean" C-H, C-O, etc bond lengths
    """
    on_at_frame  = pd.DataFrame(frag_frame[frag_frame['numAttachments']==1])
    col_names = list(on_at_frame.columns)
    #Find indices of relevant columns
    xyz_idx = col_names.index('xyz')
    atoms_idx = col_names.index('Atoms')
    labels_idx = col_names.index('Labels')
    mol_idx = col_names.index('Molecule')
    #Find H xyz position and index of atom bonded to H
    on_at_frame['H_xyz'] = on_at_frame.apply(lambda row : _find_H_xyz(mol, row[atoms_idx],row[xyz_idx],frag_frame),axis=1)
    on_at_frame['at_idx'] = on_at_frame.apply(lambda row : _find_at_idx(mol,row[atoms_idx]),axis=1)
    hxyz = len(col_names)
    atidx = len(col_names)+1
    # print(on_at_frame)
    #write gjfs
    on_at_frame.apply(lambda row : _write_frag_gjf(frag_mol=row[mol_idx],xyz_list=row[xyz_idx],symb_list=row[labels_idx],h_xyz=row[hxyz],at_idx=row[atidx],esm=esm,basis_set=basis_set,wfx=wfx,n_procs=n_procs,mem=mem,multiplicity=multiplicity),axis=1)
    return

def _clean_basis(basis_set):
    basis_set = basis_set.replace('(','')
    basis_set = basis_set.replace(')','')
    basis_set = basis_set.replace(',','')
    basis_set = basis_set.replace('+','p')
    return basis_set

def _write_frag_gjf(frag_mol, xyz_list, symb_list,h_xyz,at_idx,esm='wb97xd',basis_set='aug-cc-pvtz',wfx=True,n_procs=4,mem='3200MB',multiplicity=1):
    """Write the gjf file for a fragment. 
    Args:
        frag_mol: Chem.Mol object
        xyz_list: list of xyz_coords of list
        symb_list: list of atomic symbols of fragment
        h_xyz: xyz coords of hydrogen attached to fragment
        at_idx: index of attached atom in fragment
        esm: str, electronic structure method to include in gjf
        basis_set: str, basis set to include in gjf
        wfx: Boolean, if True add output=wfx to gjf file
        n_procs: int, >=0. if >0, add number of processors to be used to gjf
        mem: str, format "nMB" or "nGB", memory to be used in gjf
        multiplicity: int, defaults to 1. Multiplicity of molecule
    
    Default template:
        %chk={filename}.chk
        %nprocs=4
        %mem=3200MB
        # esm/basis_set opt freq output=wfx

        smile

        charge mult
        {xyz}

        {filename}.wfx

    """
    num_atoms = len(symb_list)
    smile = re.sub('\[[0-9]+\*\]', '*', Chem.MolToSmiles(frag_mol,canonical=False))
    charge = Chem.GetFormalCharge(frag_mol)
    molecule_name = _clean_molecule_name(smile)
    # print(at_idx)
    # print(xyz_list[at_idx])
    # print(h_xyz)
    #build xyz of molecule
    out_xyz = [xyz_list[at_idx],h_xyz]
    for i in range(num_atoms):
        if i != at_idx:
            out_xyz.append(xyz_list[i])
    geom_frame = pd.DataFrame(out_xyz,columns=['x','y','z'])
    symb_list.insert(1,'H')
    geom_frame['Atom'] = symb_list
    geom_frame = geom_frame[['Atom','x','y','z']]
    #create file name
    clean_basis = _clean_basis(basis_set)
    new_file_name = 'SubH'+'_'+molecule_name + '_' + esm+'_'+clean_basis
    if os.path.exists(new_file_name+'.gjf'):
        # print('deleting')
        os.remove(new_file_name+'.gjf')
    #write file
    with open(new_file_name+'.gjf', 'a') as f:
        f.write("%chk={chk}\n".format(chk=new_file_name+'.chk'))
        if n_procs:
            f.write('%nprocs={nprocs}\n'.format(nprocs=n_procs))
        if mem:
            f.write('%mem={mem}\n'.format(mem=mem))
        if wfx:
            f.write('#p {esm}/{bas} opt freq output=wfx\n'.format(esm=esm,bas=basis_set))
        else:
            f.write('#p {esm}/{bas} opt freq\n'.format(esm=esm,bas=basis_set))
        f.write('\n')
        f.write(f'{smile}')
        f.write('\n\n')
        f.write('{q} {mul}\n'.format(q=charge,mul=multiplicity))
        dfAsString = geom_frame.to_string(header=False, index=False)
        f.write(dfAsString)
        f.write('\n\n')
        if wfx:
            f.write(new_file_name+'.wfx\n\n\n')
        else:
            f.write('\n\n\n')    
    return

def _find_at_idx(mol,at_list):
    """Returns index of atom connected to the remainder of the molecule in the list 
    of fragment atom numbers

    Args: 
        mol: Chem.Mol object
        at_list: list[int] of atoms in molecule belonging to fragment being studied
            Note: these indices start at 1, while those in the molecule start at 0
    
    Returns:
        int of index of connected atom in at_list. See examples for worked explanation
    
    Example: a fragment is defined by atoms [1,3,5,6,8]
    Atom 3 is where the remainder of the molecule attaches to the fragment
    The function returns the index of 3 in the atom list.
    In this case, it would return 1
    """
    #find
    at_idx = _find_neigh_notin_frag(mol,at_list)[0]
    list_idx = at_list.index(at_idx)
    return list_idx

def _construct_unique_frame(uni_smi:list[str],uni_smi_count:list[int],xyz=[],atoms=[],parents=[],labels=[]) -> pd.DataFrame:
    """given smiles, counts and (optional) xyz coordinates, create frame"""
    uniquefrag_frame = pd.DataFrame(uni_smi,columns=['Smiles'])
    if xyz:
        uniquefrag_frame['xyz'] = xyz
    if atoms:
        uniquefrag_frame['Atoms'] = atoms
    if parents:
        uniquefrag_frame['Parent'] = parents
    if labels:
        uniquefrag_frame['Labels'] = labels
    PandasTools.AddMoleculeColumnToFrame(uniquefrag_frame,'Smiles','Molecule',
                                         includeFingerprints=True)
    uniquefrag_frame['count']=uni_smi_count
    #if we don't have xyz already add them, from MMFF94 opt
    if not xyz:
        uniquefrag_frame = _add_xyz_coords(uniquefrag_frame)
    #count number placeholders
    uniquefrag_frame = _add_number_attachements(uniquefrag_frame)
    return uniquefrag_frame

def _drop_smi_attach(smile:str):
    """completely remove placeholder if number of non-placeholder in smiles is > 1"""
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
    if frame1.empty:
        merge_frame = frame2
    elif frame2.empty:
        merge_frame = frame1
    else:
        rows_to_drop = _find_rows_to_drop(frame1,frame2)
        merge_frame = rows_to_drop['merge_frame']
        #TODO simply concat data frames
        print(rows_to_drop['drop_rows_1'])
        print(frame1)
        drop_frame_1 = frame1.drop(rows_to_drop['drop_rows_1'])
        drop_frame_2 = frame2.drop(rows_to_drop['drop_rows_2'])
        merge_frame = pd.concat([drop_frame_1,drop_frame_2,merge_frame])
        merge_frame.reset_index(inplace=True,drop=True)
        # _add_frame(drop_frame_1,merge_frame)
        # merge_frame = _add_frame(drop_frame_2,merge_frame)
    return merge_frame

def _add_frame(frame1:pd.DataFrame,merge_frame=pd.DataFrame()) -> pd.DataFrame:
    """combine two frames - probably a better way to do this"""
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

def _find_rows_to_drop(frame_a:pd.DataFrame,frame_b:pd.DataFrame):
    """for two frames, find rows with same smile, store what rows to drop in each"""
    rows_to_drop_one = []
    rows_to_drop_two = []
    col_names = list(frame_a.columns)
    merge_frame = pd.DataFrame(columns=col_names)
    frame_a_idx = list(frame_a.index)
    frame_b_idx  = list(frame_b.index)
    for i,smi in enumerate(frame_a['Smiles']):
        if smi in list(frame_b['Smiles']):
            # print(f'i is {i}')
            j=list(frame_b['Smiles']).index(smi)
            # print(f'j is {j}')
            # print(frame_a.shape)
            cum_count=frame_a.at[frame_a_idx[i],'count'] + frame_b.at[frame_b_idx[j],'count']
            # print('cum_count is {cum_count}')
            merge_frame = pd.concat([merge_frame,pd.DataFrame([list(frame_a.iloc[frame_a_idx[i]])],columns=col_names)])
            merge_frame.reset_index(inplace=True,drop=True)
            # print(f'merge frame before updating cumcount {merge_frame}')
            # print(merge_frame.shape[0]-1)
            merge_frame.at[merge_frame.shape[0]-1,'count'] = cum_count
            # print(f'merge_frame after updating cumcount {merge_frame}')
            # if made_frame == 0:
            #     merge_frame = pd.DataFrame.from_dict({'Smiles':[smi],'count':[cum_count]})
            #     made_frame=1
            # else:
            #     merge_frame.loc[len(merge_frame)] = [smi, cum_count]
            rows_to_drop_one.append(i)
            rows_to_drop_two.append(j)
    print(merge_frame)
    return {'drop_rows_1':rows_to_drop_one,'drop_rows_2':rows_to_drop_two,
            'merge_frame':merge_frame}


def count_groups_in_set(list_of_inputs:list[str],drop_attachments:bool=False,input_type='smile',bb_patt= '[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0;!#9;!#17;!#35;!#1]):2]',cml_list=[]) -> pd.DataFrame:
    """Identify unique fragments in molecules defined in the list_of_smiles, 
    and count the number of occurences for duplicates.
    Args:
        list_of_smiles: A list, with each element being a SMILES string, e.g. ['CC','C1CCCC1']
        drop_attachments: Boolean for whether or not to drop attachment points from fragments
            if True, will remove all placeholder atoms indicating connectivity
            if False, placeholder atoms will remain
        input_type: smile or molfile, based on elements of lists_of_inputs
        cml_list = defaults empty, but can be a list of cml files corresponding to the molfile inputs
        #bb_patt = SMARTS pattern for bonds to break in linkers and side chains. Defaults to breaking 
            bonds between nonring carbons with four bonds single bonded to ring atoms or carbons that don't have four bonds, and are not H, halide, or placeholder
    Returns:
        an output pd.DataFrame, with columns 'Smiles' for fragment Smiles, 
        'count' for number of times each fragment occurs in the list, and 
        'Molecule' holding a rdkit.Chem.Molecule object
        
    Example usage:
        count_groups_in_set(['c1ccc(c(c1)c2ccc(o2)C(=O)N3C[C@H](C4(C3)CC[NH2+]CC4)C(=O)NCCOCCO)F',
        'Cc1nc2ccc(cc2s1)NC(=O)c3cc(ccc3N4CCCC4)S(=O)(=O)N5CCOCC5'],drop_attachments=False)."""
    for i,inp in enumerate(list_of_inputs):
        print(inp)
        if cml_list:
            frame = identify_connected_fragments(inp,bb_patt=bb_patt,input_type=input_type,cml_file=cml_list[i],include_parent=True)
        else:
            frame = identify_connected_fragments(inp,bb_patt=bb_patt,input_type=input_type,include_parent=True)
        unique_frame = count_uniques(frame,drop_attachments)
        if i==0:
            out_frame=unique_frame
        else:
            out_frame = merge_uniques(out_frame,unique_frame)
    out_frame.drop('Molecule',axis=1)
    PandasTools.AddMoleculeColumnToFrame(out_frame,'Smiles','Molecule',includeFingerprints=True)
    #out_frame = _add_xyz_coords(out_frame)
    # out_frame = _add_number_attachements(out_frame)
    return out_frame

#old version without connectivity, included here just in case

# def identify_fragments(smile: str) -> pd.DataFrame:
#     """Perform the full fragmentation.
#     This will break down molecule into unique rings, linkers, functional groups, peripherals, 
#     but not maintain connectivity.
#     Atoms may be assigned to multiple groups. eg. C(=O)C, C=O, 
#     C may all be SMILES included from an acetyl peripheral
#     """
#     mol = utils.get_canonical_molecule(smile)
#     mol_frame = generate_full_mol_frame(mol)
#     #print(mol_frame)
#     frag_smi = _generate_part_smiles(mol_frame,molecule=mol)
#     frag_smi = _find_alkyl_groups(mol_frame,frag_smi,mol)
#     #print(fragSmi)
#     frag_frame = _generate_fragment_frame(frag_smi)
#     frag_frame = _add_ertl_functional_groups(frag_frame)
#     frag_frame = _add_xyz_coords(frag_frame)
#     frag_frame = _add_number_attachements(frag_frame)
#     #print(frag_frame)
#     return frag_frame

# def _add_ertl_functional_groups(frag_frame):
#     """For each SMILES in frag_frame, identify heavy/double bonded atom functional groups and
#     update the frame with them."""
#     for molecule in frag_frame['Molecule']:
#         if molecule.GetRingInfo().NumRings() == 0: #don't do this for rings
#             fgs = ifg.identify_functional_groups(molecule)
#             for fg_i in fgs: #iterate over identified functional groups
#                 if len(fg_i) != 0 and fg_i.atoms != '*':
#                     #if one atom and it is a ring atom, skip
#                     if len(fg_i.atomIds)==1 and molecule.GetAtomWithIdx(fg_i.atomIds[0]).IsInRing():
#                         print("Skipping")
#                         print(fg_i)
#                     else:
#                         frag_frame = _find_ertl_functional_group(fg_i,frag_frame)
#     return frag_frame

# def _find_ertl_functional_group(fg_i,frag_frame):
#     """generate smiles and molecule object for ertl functional group."""
#     #generate molecule of functional group
#     just_grp = Chem.MolFromSmarts(fg_i.atoms)
#     #generate molecule of functional group environment
#     grp_and_att = Chem.MolFromSmiles(fg_i.type)
#     if grp_and_att is not None: #if initialized successfully
#         match_patt = grp_and_att.GetSubstructMatch(just_grp) #find the match
#         if len(match_patt) >0:
#             for atom in grp_and_att.GetAtoms():
#                 a_idx = atom.GetIdx()
#                 #set the atoms in grpAndAtt to 0 if they are not in the group
#                 if a_idx not in list(match_patt):
#                     grp_and_att.GetAtomWithIdx(a_idx).SetAtomicNum(0)
#             fg_smile = Chem.MolToSmiles(grp_and_att)
#             flag=0
#             for atom in grp_and_att.GetAtoms():
#                 if atom.GetAtomicNum() == 0 and atom.IsInRing():
#                     flag = 1
#             #append to frag_frame if placeholders not in ring
#             if fg_smile not in frag_frame['Smiles'].unique():
#                 if fg_smile != len(fg_smile) * "*" and flag == 0:
#                     fg_mol = Chem.MolFromSmiles(fg_smile)
#                     frag_frame.loc[len(frag_frame)] = [fg_smile, fg_mol]
#     return frag_frame
