"""
utils.py

Various utilities used in fragmenting molecules - retrieving information from molecules
Identifying small parts of molecules, and tools for minor manipulations of SMILES or molecules
"""
import sys
sys.path.append(sys.path[0].replace('/src',''))
import rdkit
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem.Scaffolds import rdScaffoldNetwork # scaffolding
from rdkit.Chem import rdqueries # search for rdScaffoldAttachment points * to remove

def find_smallest_rings(node_molecules):
    """Given get_scaffold_vertices list of molecules, remove non-smallest nodes
    # (those with non-ring atoms or non-ring bonds)."""
    # has_rings = any_ring_atoms(node_molecules[0])
    if Chem.MolToSmiles(node_molecules[1]) != '':
        no_nonring_atoms = eliminate_nonring_atoms(node_molecules)
        no_nonring_atoms_or_bonds = eliminate_nonring_bonds(no_nonring_atoms)
    else:
        no_nonring_atoms_or_bonds = [node_molecules[0]]    
    return no_nonring_atoms_or_bonds

# def any_ring_atoms(molecule):
#     any_ring_atoms = False
#     for atom in molecule.GetAtoms():
#         if atom.IsInRing():
#             any_ring_atoms = True
#             break
#     return any_ring_atoms

def get_scaffold_vertices(molecule):
    """given rdkit Chem.molecule object return list of molecules of fragments generated by 
    scaffolding."""
    scaffold_params = set_scaffold_params()
    scaffold_network = rdScaffoldNetwork.CreateScaffoldNetwork([molecule],scaffold_params)
    node_molecules = [Chem.MolFromSmiles(x) for x in scaffold_network.nodes]
    return node_molecules

def set_scaffold_params():
    """Defines rdScaffoldNetwork parameters."""
    #use default bond breaking (break non-ring - ring single bonds, see paper for reaction SMARTS)
    scafnet_params = rdScaffoldNetwork.ScaffoldNetworkParams()
    #maintain attachments in scaffolds
    scafnet_params.includeScaffoldsWithoutAttachments = False
    #don't include scaffolds without atom labels
    scafnet_params.includeGenericScaffolds = False
    #keep all generated fragments - some were discarded messing with code if True
    scafnet_params.keepOnlyFirstFragment = False
    return scafnet_params

def get_molecules_atomicnum(molecule):
    """Given molecule object, get list of atomic numbers."""
    atom_num_list = []
    for atom in molecule.GetAtoms():
        atom_num_list.append(atom.GetAtomicNum())
    return atom_num_list

def get_molecules_atomsinrings(molecule):
    """Given molecule object, get Boolean list of if atoms are in a ring."""
    is_in_ring_list = []
    for atom in molecule.GetAtoms():
        is_in_ring_list.append(atom.IsInRing())
    return is_in_ring_list

def trim_placeholders(rwmol):
    """Given Chem.RWmol, remove atoms with atomic number 0."""
    qa = rdqueries.AtomNumEqualsQueryAtom(0) #define query for atomic number 0
    if len(rwmol.GetAtomsMatchingQuery(qa)) > 0: #if there are matches
        query_match = rwmol.GetAtomsMatchingQuery(qa)
        rm_at_idx = []
        for atom in query_match: #identify atoms to be removed
            rm_at_idx.append(atom.GetIdx())
             #remove starting from highest number so upcoming indices not affected
        rm_at_idx_sort = sorted(rm_at_idx,reverse=True)
        #e.g. need to remove 3 and 5, if don't do this and you remove 3,
        # then the 5 you want to remove is now 4, and you'll remove wrong atom
        for idx in rm_at_idx_sort: #remove atoms
            rwmol.RemoveAtom(idx)
    return rwmol

def mol_with_atom_index(mol):
    #from https://www.rdkit.org/docs/Cookbook.html
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            atom.SetAtomMapNum(atom.GetIdx()+1)
    return mol


def mol_from_molfile(mol_file):
    """takes mol_file and returns mol wth atom numbers the same
    #modified for mol file structure from retrievium
    from stackexchange https://mattermodeling.stackexchange.com/questions/7234/how-to-input-3d-coordinates-from-xyz-file-and-connectivity-from-smiles-in-rdkit"""
    m = Chem.MolFromMolFile(mol_file,removeHs=False)
    if not m:
        raise ValueError(f"""Problem creating molecule from {mol_file}""")
    # this assumes whatever program you use doesn't re-order atoms
    #  .. which is usually a safe assumption
    #  .. so we don't bother tracking atoms
    m = mol_with_atom_index(m)
    atomic_symbols = []
    xyz_coordinates = []
    ats_read = 0
    num_atoms= m.GetNumAtoms()
    print(num_atoms)
    with open(mol_file, "r") as file:
        for line_number,line in enumerate(file):
            print(line_number)
            print(line)
            if ats_read <  num_atoms and line_number > 3:
                ats_read += 1
                x, y, z, atomic_symbol = line.split()[:4]
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x),float(y),float(z)])
            elif ats_read == num_atoms:
                break
    # from https://github.com/rdkit/rdkit/issues/2413
    # conf = m.GetConformer()
# in principal, you should check that the atoms match
    # for i in range(m.GetNumAtoms()):
    #     print(i)
    #     x,y,z = xyz_coordinates[i]
    #     conf.SetAtomPosition(i,Point3D(x,y,z))
    return {'Molecule': m, 'xyz_pos':xyz_coordinates,'atomic_symbols':atomic_symbols}

def xyz_from_cml(cml_file):
    num_atom_array=0
    geom_list = []
    with open(cml_file, "r") as file:
        for line in file:
            if 'atomArray' in line:
                num_atom_array += 1
                if num_atom_array == 5:
                    continue
            if num_atom_array == 5:
                print(line)
                space_split = line.split()
                x_split = space_split[3].split("=")
                y_split = space_split[4].split("=")
                z_split = space_split[5].split("=")
                geom_list.append([float(eval(x_split[1])),float(eval(y_split[1])), float(eval(z_split[1]))])
            elif num_atom_array == 6:
                break        
    return geom_list

def get_canonical_molecule(smile: str):
    """Ensures that molecule numbering is consistent with creating molecule from canonical 
    SMILES for consistency."""
    mol = Chem.MolFromSmiles(smile)
    if mol:
        mol_smi = Chem.MolToSmiles(mol) #molsmi is canonical SMILES
    else:
        raise ValueError(f"""{smile} is not a valid SMILES code or 
                         rdkit cannot construct a molecule from it""")    
    #create canonical molecule numbering from canonical SMILES
    return Chem.MolFromSmiles(mol_smi)


    

def copy_molecule(molecule):
    """create a copy of molecule object in new object(not pointer)"""
    #see link https://sourceforge.net/p/rdkit/mailman/message/33652439/
    return Chem.Mol(molecule)

def clean_smile(trim_smi):
    """remove leftover junk from smiles when atom deleted."""
    trim_smi = trim_smi.replace('[*H]','*')
    trim_smi = trim_smi.replace('[*H3]','*')
    trim_smi = trim_smi.replace('[*H2]','*')
    trim_smi = trim_smi.replace('[*H+]','*')
    trim_smi = trim_smi.replace('[*H3+]','*')
    trim_smi = trim_smi.replace('[*H2+]','*')
    trim_smi = trim_smi.replace('[*H-]','*')
    trim_smi = trim_smi.replace('[*H3-]','*')
    trim_smi = trim_smi.replace('[*H2-]','*')
    return trim_smi

def eliminate_nonring_bonds(nodemolecules):
    """Given list of molecules of eliminate_nonring_atoms output, 
    remove molecules that contain bonds that are not ring bonds or double bonded to ring."""
    #mainly removes ring-other ring single bonds, as in biphenyl
    ring_frags=[]
    for frag in nodemolecules:
        flag=1
        for bond in frag.GetBonds():
            if not bond.IsInRing():
                b_at = bond.GetBeginAtom().GetAtomicNum()
                e_at = bond.GetEndAtom().GetAtomicNum()
                if bond.GetBondType() != Chem.rdchem.BondType.DOUBLE and  b_at != 0 and e_at != 0:
                    flag=0
                    break
        if flag == 1:
            ring_frags.append(frag)
    return ring_frags

def eliminate_nonring_atoms(nodemolecules):
    """given list of molecules of utils.get_scaffold_vertices output, removes molecules that 
    contain atoms that are not in ring or not double bonded to ring."""
    first_parse = []
    for frag_mol in nodemolecules:
        flag=1
        for idx,atom in enumerate(frag_mol.GetAtoms()):
            non_ring_double=0
            #if atom is not in ring, check if it is double bonded to a ring
            if not atom.IsInRing():
                for neigh in atom.GetNeighbors():
                    bond_type = frag_mol.GetBondBetweenAtoms(idx,neigh.GetIdx()).GetBondType()
                    #print(bond_type)
                    n_in_r = frag_mol.GetAtomWithIdx(neigh.GetIdx()).IsInRing()
                    if  n_in_r and bond_type ==Chem.rdchem.BondType.DOUBLE:
                        print('I passed the if')
                        non_ring_double=1
            #if not attachment (atomic number 0 used as attachments by rdScaffoldNetwork)
            if atom.GetAtomicNum() != 0:
                if not atom.IsInRing(): #if atom is not in ring
                    if non_ring_double==0: #if atom is not double bonded to ring
                        flag=0 #all the above true, don't remove molecule from output
                        #will remove from output if a single atom in the node fails the tests
                        break
        if flag == 1: #if pass all tests for all atoms, add to list to be returned
            first_parse.append(frag_mol)
    return first_parse
