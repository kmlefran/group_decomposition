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
from rdkit.Chem import rdDetermineBonds, AllChem
import pandas as pd
import os
from rdkit.RDLogger import logger



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

def _get_charge_from_cml(cml_file):
    with open(cml_file,"r") as file:
        for line in file:
            if "formalCharge" in line:
                split_line = line.split(" ")
                for i, word in enumerate(split_line):
                    if "formalCharge" in word:
                        charge = int(word.replace("formalCharge=","").replace(">\n","").replace('"',''))
                        break
    return charge
                

def mol_from_xyzfile(xyz_file:str,cml_file):
    charge = _get_charge_from_cml(cml_file)
    # raw_mol = Chem.MolFromXYZFile('DUDE_67368827_adrb2_decoys_C19H25N3O4_CIR.xyz')
    # mol = Chem.Mol(raw_mol)
    # rdDetermineBonds.DetermineConnectivity(mol)
    # rdDetermineBonds.DetermineBondOrders(mol)
    raw_mol = Chem.MolFromXYZFile(xyz_file)
    mol = Chem.Mol(raw_mol)
    try:
        rdDetermineBonds.DetermineBonds(mol,charge=charge)
    except:
        if os.path.isfile('error_log.txt'):
            er_file = open('error_log.txt','a')
            er_file.write(f'Could not determine bond orders for {cml_file}\n')
            er_file.close()
        else:
            er_file = open('error_log.txt','w')
            er_file.write(f'Could not determine bond orders for {cml_file}\n')
            er_file.close()
        return None
    else:
        atomic_symbols = []
        xyz_coordinates = []
        ats_read = 0
        num_atoms= mol.GetNumAtoms()
        with open(xyz_file, "r") as file:
            for line_number,line in enumerate(file):
                if ats_read <  num_atoms and line_number > 1:
                    ats_read += 1
                    atomic_symbol, x, y, z  = line.split()[:4]
                    atomic_symbols.append(atomic_symbol)
                    xyz_coordinates.append([float(x),float(y),float(z)])
                elif ats_read == num_atoms:
                    break
        return {'Molecule': mol_with_atom_index(mol), 'xyz_pos':xyz_coordinates,'atomic_symbols':atomic_symbols}
    # from https://github.com/rdkit/rdkit/issues/2413
    # conf = m.GetConformer()
# in principal, you should check that the atoms match
    # for i in range(m.GetNumAtoms()):
    #     print(i)
    #     x,y,z = xyz_coordinates[i]
    #     conf.SetAtomPosition(i,Point3D(x,y,z))
    

def get_cml_atom_types(cml_file):
    n_atl = 0
    type_list = []
    idx_list = []
    with open(cml_file, "r") as file:
        for line_number,line in enumerate(file):
            if 'atomTypeList' in line:
                n_atl += 1
                if n_atl == 2:
                    break
            elif n_atl == 1:
                split_line = line.split()
                idx_list.append(int(split_line[1].split('=')[1].replace('"','').replace('a',''))-1)
                at_label = split_line[2].split('=')[1].replace('"','')
                at_type = int(split_line[3].split('=')[1].replace('"',''))
                at_valence = int(split_line[4].split('=')[1].split('/')[0].replace('"',''))
                type_list.append((at_label,at_type,at_valence))
    temp_frame = pd.DataFrame(list(zip(idx_list,type_list)),columns=['idx','type'])
    temp_frame.sort_values(by='idx',inplace=True)
    return temp_frame['type']

def add_cml_atoms_bonds(el_list,bond_list):
    """create a molecule from cml file, building it one atom at a time then 
    adding in bond orders.
    
    Note: Bond orders from Retrievium cml are problematic at times.
    
    Recommended construction is to build atoms and connectivity from cml file
    Then assign bond orders based on template smiles also found in cml file"""
    flag=1
    for atom in el_list:
        if flag:
            mol = Chem.MolFromSmiles(atom)
            rwmol = Chem.RWMol(mol)
            rwmol.BeginBatchEdit()
            flag=0
        else:
            rwmol.AddAtom(Chem.Atom(atom))
    #mw.AddBond(6,7,Chem.BondType.SINGLE)
    for bond in bond_list:
        if bond[2] == 'S':
            rwmol.AddBond(bond[0]-1,bond[1]-1,Chem.BondType.SINGLE)
        elif bond[2] == 'D':
            rwmol.AddBond(bond[0]-1,bond[1]-1,Chem.BondType.DOUBLE)
        elif bond[2] =='T':
            rwmol.AddBond(bond[0]-1,bond[1]-1,Chem.BondType.TRIPLE)
        elif bond[2] == 'A':
            rwmol.AddBond(bond[0]-1,bond[1]-1,Chem.BondType.AROMATIC)
    rwmol.CommitBatchEdit()
    return rwmol

def add_cml_single_atoms_bonds(el_list,bond_list):
    """Build a mol from list of atoms and bonds, one atom at a time
    Bonds assigned are only single bonds.
    
    Adjust bonds after with modAssignBondOrdersFromTemplate"""
    flag=1
    rwmol = Chem.RWMol(Chem.Mol())
    for atom in el_list:
        rwmol.AddAtom(Chem.Atom(atom))
    #mw.AddBond(6,7,Chem.BondType.SINGLE)
    for bond in bond_list:
        rwmol.AddBond(bond[0]-1,bond[1]-1,Chem.BondType.SINGLE)
    rwmol.CommitBatchEdit()
    return rwmol



def smiles_from_cml(cml_file):
    """Finds the Retreivium input SMILES in a cml file"""
    flag=0
    with open(cml_file, "r") as file:
        for line in file:
            if 'retrievium:inputSMILES' in line:
                flag=1
            elif flag==1:
                smile = line.split('>')[1].split('<')[0]
                break
    return smile


def mol_from_cml(cml_file, input_type='cmlfile'):
    """Creates a molecule from a cml file and returns atoms, xyz and types
    
    Builds molecule one atom at a time connected by single bonds
    Then determines bond orders by mapping to a template smiles in the cml
    Finally, updates property cache, initializes ring info, and sanitizes mol
    If no match between SMILEs and connectivity, returns None and writes 
    error to file

    Args:
        cml_file - name of cml in current directory or path to the file
        input_type - cmlfile or cmldict - cmlfile if just raw cml file, cmldict if filename
    Returns:
        list of [Molecule, list of atom symbols in molecule, 
        list of xyz coords of atoms in molecule, 
        list of atom types of atoms in molecule]
        Note: list order matches mol numbering in cml
    """
    #'geom':geom_list, 'atom_types':list(temp_frame['type']),'bonds':bond_list,'labels':el_list,'charge':charge,'multiplicity':multiplicity,'smiles':smile}
    if input_type=='cmlfile':
        xyz_coords, at_types, bond_list,el_list,charge = data_from_cml(cml_file,True)
        smile = smiles_from_cml(cml_file)
    elif input_type == 'cmldict':
        xyz_coords = cml_file['geom']
        at_types = cml_file['atom_types']
        bond_list = cml_file['bonds']
        el_list = cml_file['labels']
        charge = cml_file['charge']
        smile = cml_file['smiles']
    rwmol = add_cml_single_atoms_bonds(el_list,bond_list)
    for atom in rwmol.GetAtoms():
        atom.SetNoImplicit(True)
    
    rwmol2 = Chem.RemoveHs(rwmol,implicitOnly=True,updateExplicitCount=False)
    template = AllChem.MolFromSmiles(smile)
    bond_mol = modAssignBondOrdersFromTemplate(template,rwmol2,cml_file)
     # need rings for aromaticity check
        # if os.path.isfile('error_log.txt'):
        #     er_file = open('error_log.txt','a')
        #     er_file.write(f'Could not sanitize {cml_file}\n')
        #     er_file.close()
        # else:
        #     er_file = open('error_log.txt','w')
        #     er_file.write(f'Could not sanitize {cml_file}\n')
        #     er_file.close()
    if bond_mol:
        bond_mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(bond_mol)
        try:
            Chem.SanitizeMol(bond_mol)
        except:
            write_error(f'Could not sanitize {cml_file}\n')
            return [None,None,None,None]
        return [mol_with_atom_index(bond_mol),el_list,xyz_coords,at_types]
    else:
        # if os.path.isfile('error_log.txt'):
        #     er_file = open('error_log.txt','a')
        #     er_file.write(f'No match between template smiles and connected geom for {cml_file}\n')
        #     er_file.close()
        # else:
        #     er_file = open('error_log.txt','w')
        #     er_file.write(f'No match between template smiles and connected geom for {cml_file}\n')
        #     er_file.close()
        return [None,None,None,None]

def write_error(errormessage):
    if os.path.isfile('error_log.txt'):
        er_file = open('error_log.txt','a')
        er_file.write(errormessage)
        er_file.close()
    else:
        er_file = open('error_log.txt','w')
        er_file.write(errormessage)
        er_file.close()
    return

def modAssignBondOrdersFromTemplate(refmol, mol,cml_file):
  """ This is originally from RDKit AllChem module. 
  Modified here by Kevin Lefrancois-Gagnon(KLG) to disallow implicit hydrogens on all
  molcule objects used. This corresponds to the 4 for loops after creation of mol
  objects.
  Also Returns None for no match instead of value error, for use on large number of 
  molecules, a single error won't stop the whole thing
  All other code is unmodified from the original.
  This was required by KLG for the following reason:
    using the original AssignBondOrdersFromTemplate on a molecule with 
    only explicit hydrogens with only connectivity, no  bond types other than
    single bonds. When a charged atom like nitrogen was present, extra H were being added
    For example when a part was expected as C-[NH2+]-C, the following was seen
    C-[NH2+]([H])([H])-C, essentially doubling the expected # hydrogens
    These could not be deleted by Chem.RemoveHs(mol,implicitOnly=True) as for
    some reason these were explicit. The reason for why these were added is unclear

    Beyond the obviously problematic extra hydrogens, it also produced an error case
    when the later part of the code would clean up bond orders, like for some ring ketones
    in a ring labeled aromatic by the SMILEs.
    Adjusting implicit Hs to be disallowed for all molecules and their copies at the start
    resulted in the normal expected outputs.
Resume original documentation:
  assigns bond orders to a molecule based on the
    bond orders in a template molecule

    Arguments
      - refmol: the template molecule
      - mol: the molecule to assign bond orders to

    An example, start by generating a template from a SMILES
    and read in the PDB structure of the molecule

    >>> import os
    >>> from rdkit.Chem import AllChem
    >>> template = AllChem.MolFromSmiles("CN1C(=NC(C1=O)(c2ccccc2)c3ccccc3)N")
    >>> mol = AllChem.MolFromPDBFile(os.path.join(RDConfig.RDCodeDir, 'Chem', 'test_data', '4DJU_lig.pdb'))
    >>> len([1 for b in template.GetBonds() if b.GetBondTypeAsDouble() == 1.0])
    8
    >>> len([1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 1.0])
    22

    Now assign the bond orders based on the template molecule

    >>> newMol = AllChem.AssignBondOrdersFromTemplate(template, mol)
    >>> len([1 for b in newMol.GetBonds() if b.GetBondTypeAsDouble() == 1.0])
    8

    Note that the template molecule should have no explicit hydrogens
    else the algorithm will fail.

    It also works if there are different formal charges (this was github issue 235):

    >>> template=AllChem.MolFromSmiles('CN(C)C(=O)Cc1ccc2c(c1)NC(=O)c3ccc(cc3N2)c4ccc(c(c4)OC)[N+](=O)[O-]')
    >>> mol = AllChem.MolFromMolFile(os.path.join(RDConfig.RDCodeDir, 'Chem', 'test_data', '4FTR_lig.mol'))
    >>> AllChem.MolToSmiles(mol)
    'COC1CC(C2CCC3C(O)NC4CC(CC(O)N(C)C)CCC4NC3C2)CCC1N(O)O'
    >>> newMol = AllChem.AssignBondOrdersFromTemplate(template, mol)
    >>> AllChem.MolToSmiles(newMol)
    'COc1cc(-c2ccc3c(c2)Nc2ccc(CC(=O)N(C)C)cc2NC3=O)ccc1[N+](=O)[O-]'

    """
  refmol = Chem.AddHs(refmol)
  refmol2 = Chem.Mol(refmol)
  refmol2 = Chem.AddHs(refmol2)
  mol2 = Chem.Mol(mol)
  for atom in mol.GetAtoms():
    atom.SetNoImplicit(True)
  for atom in mol.GetAtoms():
    atom.SetNoImplicit(True)
  for atom in refmol2.GetAtoms():
    atom.SetNoImplicit(True)
  for atom in refmol.GetAtoms():
    atom.SetNoImplicit(True)
  # do the molecules match already?
  matching = mol2.GetSubstructMatch(refmol2)
  if not matching:  # no, they don't match
    # check if bonds of mol are SINGLE Chem.rdchem.BondType.DOUBLE
    for b in mol2.GetBonds():
      if b.GetBondType() != Chem.rdchem.BondType.SINGLE:
        b.SetBondType(Chem.rdchem.BondType.SINGLE)
        b.SetIsAromatic(False)
    # set the bonds of mol to SINGLE
    for b in refmol2.GetBonds():
      b.SetBondType(Chem.rdchem.BondType.SINGLE)
      b.SetIsAromatic(False)
    # set atom charges to zero;
    for a in refmol2.GetAtoms():
      a.SetFormalCharge(0)
    for a in mol2.GetAtoms():
      a.SetFormalCharge(0)

    matching = mol2.GetSubstructMatches(refmol2, uniquify=False)
    # do the molecules match now?
    if matching:
    #   if len(matching) > 1:
    #     logger.warning(msg="More than one matching pattern found - picking one")
      matching = matching[0]
      # apply matching: set bond properties
      for b in refmol.GetBonds():
        atom1 = matching[b.GetBeginAtomIdx()]
        atom2 = matching[b.GetEndAtomIdx()]
        b2 = mol2.GetBondBetweenAtoms(atom1, atom2)
        b2.SetBondType(b.GetBondType())
        b2.SetIsAromatic(b.GetIsAromatic())
      # apply matching: set atom properties
      for a in refmol.GetAtoms():
        a2 = mol2.GetAtomWithIdx(matching[a.GetIdx()])
        a2.SetHybridization(a.GetHybridization())
        a2.SetIsAromatic(a.GetIsAromatic())
        a2.SetNumExplicitHs(a.GetNumExplicitHs())
        a2.SetFormalCharge(a.GetFormalCharge())
      try:
        Chem.SanitizeMol(mol2)
      except:
        er_message = f'Could not sanitize {cml_file}\n'
        # smile = smiles_from_cml(cml_file)
        er_message += at_num_er(refmol2,mol2)
        # smimol = Chem.MolFromSmiles()
        write_error(er_message)
        return None
      if hasattr(mol2, '__sssAtoms'):
        mol2.__sssAtoms = None  # we don't want all bonds highlighted
    else:
      er_message = f'No match between template smiles and connected geom for {cml_file}\n'
      er_message += at_num_er(refmol2,mol2)
      write_error(er_message)
      return None
  return mol2

def at_num_er(refmol2,mol2):
    er_message = ''
    smi_num = []
    smi_lab = []
    symb_list=[]
    num_list=[]
    for atom in mol2.GetAtoms():
        if atom.GetSymbol() not in symb_list:
            symb_list.append(atom.GetSymbol())
            num_list.append(1)
        else:
            num_list[symb_list.index(atom.GetSymbol())] += 1
    for atom in refmol2.GetAtoms():
        if atom.GetSymbol() not in smi_lab:
            smi_lab.append(atom.GetSymbol())
            smi_num.append(1)
        else:
            smi_num[smi_lab.index(atom.GetSymbol())] += 1
    for i in range(0,len(num_list)):
        j = smi_lab.index(symb_list[i])
        if smi_num[j] != num_list[i]:
            er_message += f'Expected {smi_num[j]} {smi_lab[j]} from SMILES, observed {num_list[i]} in xyz\n'
    return er_message

def data_from_cml(cml_file,bonds=False):
    """Gets symbols, xyz coords, bonds and charge of a mol from cml file"""
    num_atom_array=0
    geom_list = []
    n_atl = 0
    n_bary = 0
    type_list = []
    idx_list = []
    bond_list = []
    el_list = []
    with open(cml_file, "r") as file:
        for line in file:
            if "formalCharge" in line:
                split_line = line.split(" ")
                for i, word in enumerate(split_line):
                    if "formalCharge" in word:
                        charge = int(word.replace("formalCharge=","").replace(">\n","").replace('"',''))
                        continue
            if 'atomArray' in line:
                num_atom_array += 1
                if num_atom_array == 5:
                    continue
            if num_atom_array == 5:
                quote_split = line.split('"')
                #maybe only do el_list with bonds?
                el_list.append(quote_split[3])
                x_split = quote_split[5]
                y_split = quote_split[7]
                z_split = quote_split[9]
                geom_list.append([float(eval(x_split)),float(eval(y_split)), float(eval(z_split))])
            elif num_atom_array == 6:
                if bonds:
                    if 'bondArray' in line:
                        n_bary+=1
                        if n_bary ==1 or n_bary==2:
                            continue
                    elif n_bary == 1:
                        split_line = line.split()
                        at_1 = int(split_line[1].split('"')[1].replace('a',''))
                        at_2 = int(split_line[2].replace('"','').replace('a',''))
                        b_ord = split_line[4].split('"')[1]
                        bond_list.append((at_1,at_2,b_ord))
                if 'atomTypeList' in line:
                    n_atl += 1
                    if n_atl ==1:
                        continue
                    elif n_atl == 2:
                        break
                elif n_atl == 1:
                    split_line = line.split()
                    idx_list.append(int(split_line[1].split('=')[1].replace('"','').replace('a',''))-1)
                    at_label = split_line[2].split('=')[1].replace('"','')
                    at_type = int(split_line[3].split('=')[1].replace('"',''))
                    at_valence = int(split_line[4].split('=')[1].split('/')[0].replace('"',''))
                    type_list.append((at_label,at_type,at_valence))
    temp_frame = pd.DataFrame(list(zip(idx_list,type_list)),columns=['idx','type'])
    temp_frame.sort_values(by='idx',inplace=True)
    if bonds:
        return [geom_list, list(temp_frame['type']),bond_list,el_list,charge]
    else:
        return [geom_list, list(temp_frame['type'])]

def all_data_from_cml(data):
    """Gets symbols, xyz coords, bonds and charge of a mol from cml file"""
    num_atom_array=0
    geom_list = []
    n_atl = 0
    n_bary = 0
    type_list = []
    idx_list = []
    bond_list = []
    el_list = []
    smi_flag=0
    for line in data:
        if "formalCharge" in line:
            split_line = line.split(" ")
            for i, word in enumerate(split_line):
                if "spinMultiplicity" in word:
                    multiplicity = int(word.replace("spinMultiplicity=").replace('"',''))
                if "formalCharge" in word:
                    charge = int(word.replace("formalCharge=","").replace(">\n","").replace('"',''))
                    continue
        if 'retrievium:inputSMILES' in line:
            smi_flag=1
        elif smi_flag==1:
            smile = line.split('>')[1].split('<')[0]
            smi_flag=2
        if 'atomArray' in line:
            num_atom_array += 1
            if num_atom_array == 5:
                continue
        if num_atom_array == 5:
            quote_split = line.split('"')
            #maybe only do el_list with bonds?
            el_list.append(quote_split[3])
            x_split = quote_split[5]
            y_split = quote_split[7]
            z_split = quote_split[9]
            geom_list.append([float(eval(x_split)),float(eval(y_split)), float(eval(z_split))])
        elif num_atom_array == 6:               
            if 'bondArray' in line:
                n_bary+=1
                if n_bary ==1 or n_bary==2:
                    continue
            elif n_bary == 1:
                split_line = line.split()
                at_1 = int(split_line[1].split('"')[1].replace('a',''))
                at_2 = int(split_line[2].replace('"','').replace('a',''))
                b_ord = split_line[4].split('"')[1]
                bond_list.append((at_1,at_2,b_ord))
            if 'atomTypeList' in line:
                n_atl += 1
                if n_atl ==1:
                    continue
                elif n_atl == 2:
                    break
            elif n_atl == 1:
                split_line = line.split()
                idx_list.append(int(split_line[1].split('=')[1].replace('"','').replace('a',''))-1)
                at_label = split_line[2].split('=')[1].replace('"','')
                at_type = int(split_line[3].split('=')[1].replace('"',''))
                at_valence = int(split_line[4].split('=')[1].split('/')[0].replace('"',''))
                type_list.append((at_label,at_type,at_valence))
    temp_frame = pd.DataFrame(list(zip(idx_list,type_list)),columns=['idx','type'])
    temp_frame.sort_values(by='idx',inplace=True)
    
    return {'geom':geom_list, 'atom_types':list(temp_frame['type']),'bonds':bond_list,'labels':el_list,'charge':charge,'multiplicity':multiplicity,'smiles':smile}


def mol_from_molfile(mol_file,inc_xyz=False):
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
    with open(mol_file, "r") as file:
        for line_number,line in enumerate(file):
            if ats_read <  num_atoms and line_number > 3:
                ats_read += 1
                x, y, z, atomic_symbol = line.split()[:4]
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x),float(y),float(z)])
            elif ats_read == num_atoms:
                break
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

def link_molecules(mol_1:Chem.Mol,mol_2:Chem.Mol,dl_1:int,dl_2:int):
    """Given two mols, each with dummy atoms that have dummyAtomLabels, link the molecules between the dummy atoms specified by labels dl_1 and dl_2
    
    Modified from https://www.oloren.ai/blog/add_rgroup.html
    Written by David Huang, Oloren AI, modified by Kevin Lefrancois-Gagnon

    Args:
        mol_1: Chem.Mol object
        mol_2: Chem.Mol object
        dl_1: the isotope of the dummy atom in mol_1 which will be replaced by mol_2
         dl_2: the isotope of the dummy atom in mol_2 which will be replaced by mol_1
          
    Returns:
        Chem.Mol object with mol_1 and mol_2 linked where dl_1 and dl_2 were """


    # Loop over atoms until there are no wildcard atoms
    # Find wildcard atom if available, otherwise exit
    #We use the isotope here are FragmentOnBonds labels the dummy atoms by changing their isotope
    a = None
    for a_ in mol_1.GetAtoms():
        if a_.GetAtomicNum() == 0 and a_.GetIsotope() == dl_1:
            a = a_
            break
    if not a:
        raise ValueError(f"""Input molecule mol_1 does not have atom with dummy label {dl_1}""")
    b = None
    for b_ in mol_2.GetAtoms():
        if b_.GetAtomicNum() == 0 and b_.GetIsotope() == dl_2:
            b = b_
            break
    if not b:
        raise ValueError(f"""Input molecule mol_1 does not have atom with dummy label {dl_2}""")
    # Set wildcard atoms to having AtomMapNum 1000 for tracking
    a.SetAtomMapNum(1000)
    b.SetAtomMapNum(1000)
    # Put group and base molecule together and make it editable
    m = Chem.CombineMols(mol_1, mol_2)
    m = Chem.RWMol(m)
    # Find using tracking number the atoms to merge in new molecule
    a1 = None
    a2 = None
    for at in m.GetAtoms():
        if at.GetAtomMapNum() == 1000:
            if a1 is None:
                a1 = at
            else:
                a2 = at
    # Find atoms to bind together based on atoms to merge
    b1 = a1.GetBonds()[0]
    start = (b1.GetBeginAtomIdx() if b1.GetEndAtomIdx() == a1.GetIdx()
        else b1.GetEndAtomIdx())

    b2 = a2.GetBonds()[0]
    end = (b2.GetBeginAtomIdx() if b2.GetEndAtomIdx() == a2.GetIdx()
        else b2.GetEndAtomIdx())

    # Add the connection and remove original wildcard atoms
    m.AddBond(start, end, order=Chem.rdchem.BondType.SINGLE)
    m.RemoveAtom(a1.GetIdx())
    m.RemoveAtom(a2.GetIdx())

    return m

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

# def eliminate_nonring_bonds(nodemolecules):
#     """Given list of molecules of eliminate_nonring_atoms output, 
#     remove molecules that contain bonds that are not ring bonds or double bonded to ring."""
#     #mainly removes ring-other ring single bonds, as in biphenyl
#     ring_frags=[]
#     for frag in nodemolecules:
#         flag=1
#         for bond in frag.GetBonds():
#             if not bond.IsInRing():
#                 b_at = bond.GetBeginAtom().GetAtomicNum()
#                 e_at = bond.GetEndAtom().GetAtomicNum()
#                 if bond.GetBondType() != Chem.rdchem.BondType.DOUBLE and  b_at != 0 and e_at != 0:
#                     flag=0
#                     break
#         if flag == 1:
#             ring_frags.append(frag)
#     return ring_frags

# def eliminate_nonring_atoms(nodemolecules):
#     """given list of molecules of utils.get_scaffold_vertices output, removes molecules that 
#     contain atoms that are not in ring or not double bonded to ring."""
#     first_parse = []
#     for frag_mol in nodemolecules:
#         flag=1
#         for idx,atom in enumerate(frag_mol.GetAtoms()):
#             non_ring_double=0
#             #if atom is not in ring, check if it is double bonded to a ring
#             if not atom.IsInRing():
#                 for neigh in atom.GetNeighbors():
#                     bond_type = frag_mol.GetBondBetweenAtoms(idx,neigh.GetIdx()).GetBondType()
#                     #print(bond_type)
#                     n_in_r = frag_mol.GetAtomWithIdx(neigh.GetIdx()).IsInRing()
#                     if  n_in_r and bond_type ==Chem.rdchem.BondType.DOUBLE:
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

# def remove_a_bond(rwmol,ex_b_dict):
#     a_idx = list(ex_b_dict.keys())
#     atom = rwmol.GetAtomWithIdx(a_idx)
#     extra_bonds = ex_b_dict[a_idx]
#     bond_types = [b.GetBondType() for b in atom.GetBonds()]
#     bond_ids = [b.GetIdx() for b in atom.GetBonds()]
#     if extra_bonds == 0.5:
#         bond_types.index(Chem.rdchem.BondType.AROMATIC)
#     return

# def has_too_many_bonds(atom):
#     bo = 0
#     for bond in atom.GetBonds():
#         bo += bond.GetBondTypeAsDouble()
#     symb = atom.GetSymbol()
#     charge = atom.GetFormalCharge()
#     num_neigh = len(atom.GetNeighbors())
#     extra_bonds = bo - _valence_dict_bn[symb][charge][num_neigh]
#     return {atom.GetIdx():extra_bonds}
    # Chem.SanitizeMol(rwmol)
    # return [mol_with_atom_index(rwmol),el_list,xyz_coords,at_types]

    # def get_charged_at_dict(smile):
#     mol = Chem.AddHs(Chem.MolFromSmiles(smile))
#     out_dict = {i:{'symbol':atom.GetSymbol(),'charge':atom.GetFormalCharge(),
#                'neighbors':list(zip([x.GetSymbol() for x in atom.GetNeighbors()],[tuple(sorted([y.GetSymbol() for y in x.GetNeighbors()])) for x in atom.GetNeighbors()])),
#             #    'num_neigh_bonds':[len(x.GetBonds()) for x in atom.GetNeighbors()],
#                'bonds':[x for x in atom.GetBonds()],
#                'bondTypes':list(zip([x.GetBeginAtom().GetSymbol() for x in atom.GetBonds()],[x.GetEndAtom().GetSymbol() for x in atom.GetBonds()],[x.GetBondType() for x in atom.GetBonds()])),
#                'aromaticity':atom.GetIsAromatic(),
#                'in_ring':atom.IsInRing()} for i,atom in enumerate(mol.GetAtoms()) if atom.GetFormalCharge()!=0}
    
#     out_list = list(out_dict.values())
#     # for i_dict in out_list:
#     # for el in i_dict['neighbors']:
#     #     el[1].sort()
#             # el.append(tuple(el[1]))
#             # del el[1]
#     return out_list

# def id_charge_atoms(mol,smile_dicts):
#     possible_ats={}
#     print(smile_dicts)
#     # count=0
#     for atom in mol.GetAtoms():
#         for i,dict in enumerate(smile_dicts):
#             # print(atom.GetSymbol())
#             # print(dict['symbol'])
#             if atom.GetSymbol() != dict['symbol']:
#                 continue
#             # print(atom.IsInRing())
#             # print(dict['in_ring'])
#             if atom.IsInRing() != dict['in_ring']:
#                 continue
#             neighs = list(zip([x.GetSymbol() for x in atom.GetNeighbors()],[tuple(sorted([y.GetSymbol() for y in x.GetNeighbors()])) for x in atom.GetNeighbors()]))
#             # print(dict['neighbors'])
#             # print(neighs)
#             # first_set = set(map(tuple, neighs))
#             # secnd_set = set(map(tuple, dict['neighbors']))
#             # print(first_set)
#             # print(secnd_set)
#             if set(neighs) != set(dict['neighbors']):
#                 continue
#             if not possible_ats:
#                 possible_ats.update({'mol_atom':[atom.GetIdx()]})
#                 possible_ats.update({'list_el':[i]})
#             else:
#                 possible_ats['mol_atom'].append(atom.GetIdx())
#                 possible_ats['list_el'].append(i)    
#             # possible_ats.update({count:(atom.GetIdx(),i)})
#             # count += 1
#     if len(set((possible_ats['mol_atom']))) != len(possible_ats['mol_atom']):
#         return None
#     elif len(set((possible_ats['list_el']))) != len(possible_ats['list_el']):
#         return None
#     else:
#         return possible_ats

# def get_total_bo(atom):
#     order = 0
#     b_idx = {}
#     for bond in atom.GetBonds():
#         order += bond.GetBondTypeAsDouble()
#         b_idx.update({bond.GetIdx():bond.GetBondType()})
#     return {'order':order,'bonds':b_idx}

# def adjust_bond_orders(rwmol,mol_charge_locs,smile_charge_locs):
#     charge_els = list(zip(mol_charge_locs['mol_atom'],mol_charge_locs['list_el']))
#     orders_to_check = list(range(0,rwmol.GetNumAtoms()))
#     for c_at in charge_els:
#         charge = smile_charge_locs[c_at[1]]['charge']
#         cur_at = rwmol.GetAtomWithIdx(c_at[0])
#         cur_at.SetFormalCharge(charge)
#         cur_at_bo = get_total_bo(cur_at)
#         if cur_at_bo['order'] not in _valence_dict_bc[cur_at.GetSymbol()][charge]:
#             #currently only for atoms with one possible valence
#             lower_by = cur_at_bo['order'] - _valence_dict_bc[cur_at.GetSymbol()][charge][0]
#             if lower_by == -1:

#             elif lower_by == 1:

# def charge_atoms(rwmol,mol_charge_locs,smile_charge_locs):
#     charge_els = list(zip(mol_charge_locs['mol_atom'],mol_charge_locs['list_el']))
#     for c_at in charge_els:
#         charge = smile_charge_locs[c_at[1]]['charge']
#         cur_at = rwmol.GetAtomWithIdx(c_at[0])
#         cur_at.SetFormalCharge(charge)
#     return

# def adjust_neutral_n_valence(rwmol):
#     for atom in rwmol.GetAtoms():
#         if atom.GetSymbol() == 'N':
#             bo=0.
#             for bond in atom.GetBonds():
#                 bo += bond.GetBondTypeAsDouble()
#             amt_to_adj = bo - 3. + atom.GetFormalCharge()
#             check_aromaticity = []
#             if bo > 3. + atom.GetFormalCharge():
#                 # bonds_to_lower_idx = [x.GetIdx() for x in atom.GetBonds() if x.GetBondTypeAsDouble() > 1.]
#                 # bonds_to_lower_order = [x for x in atom.GetBonds() if x.GetBondTypeAsDouble() > 1.]
#                 # num_double = list.count(2.)
#                 # num_aro = list.count(1.5)
#                 # num_triple = list.count()
#                 aro_idx,double_idx,triple_idx,num_aro,num_double,num_triple = get_bond_indices(atom)
#                 # aro_idx = [x.GetIdx() for x in atom.GetBonds() if x.GetBondTypeAsDouble()==1.5]
#                 # double_idx = aro_idx = [x.GetIdx() for x in atom.GetBonds() if x.GetBondTypeAsDouble()==2.0]
#                 # triple_idx = [x.GetIdx() for x in atom.GetBonds() if x.GetBondTypeAsDouble()==3.0]
#                 # num_aro = len(aro_idx)
#                 # num_double = len(double_idx)
#                 # num_triple = len(triple_idx)
#                 if amt_to_adj == 0.5 and num_aro == 1:
#                     rwmol.GetBondWithIdx(aro_idx[0]).SetBondType(Chem.rdchem.BondType.SINGLE)
#                     # bg_at = rwmol.GetBondWithIdx(aro_idx[0]).GetBeginAtom()
#                     # end_at = rwmol.GetBondWithIdx(aro_idx[0]).GetEndAtom()
#                     check_aromaticity.append(rwmol.GetBondWithIdx(aro_idx[0]))
#                 elif amt_to_adj == 1.0 and num_aro == 2:
#                     rwmol.GetBondWithIdx(aro_idx[0]).SetBondType(Chem.rdchem.BondType.SINGLE)
#                     rwmol.GetBondWithIdx(aro_idx[1]).SetBondType(Chem.rdchem.BondType.SINGLE)
#                     check_aromaticity.append(rwmol.GetBondWithIdx(aro_idx[1]))
#                     check_aromaticity.append(rwmol.GetBondWithIdx(aro_idx[0]))
#                 elif amt_to_adj == 0.5 and num_aro == 3:
#                     raise ValueError('You did not account for a nitrogen with 3 aromatic bonds did you. You said it probably would not happen. You were wrong.')
#                 elif amt_to_adj == 1.0 and num_double == 1:
#                     rwmol.GetBondWithIdx(double_idx[0]).SetBondType(Chem.rdchem.BondType.SINGLE)
#     return check_aromaticity

# def update_aromaticity(rwmol,aro_check):
#     for bid in aro_check:
#         bond = rwmol.GetBondWithIdx(bid)
#         bg_at = bond.GetBeginAtom()
#         end_at = bond.GetEndAtom()
#         if bg_at.IsInRing() and end_at.IsInRing():
#             bg_at.SetIsAromatic(False)
#             end_at.SetIsAromatic(False)
#         elif bg_at.IsInRing():
#             bg_at.SetIsAromatic(False)
#         elif end_at.IsInRing():
#             end_at.SetIsAromatic(False)
#     return


# def adjust_o_bo(o_at):
#     o_charge = o_at.GetFormalCharge()
#     o_valence = o_at.GetExplicitValence()
#     if o_charge == -1 and o_valence == 1:
#         return
#     elif o_charge == 0 and o_valence == 2:
#         return
#     elif o_charge == 1 and o_valence == 3:
#         return
#     else:
#         o_bonds = o_at.GetBonds()
#         num_bonds = len(o_bonds)
#         if num_bonds == 2 and o_charge == 0:
#             o_bonds[0].SetBondType(Chem.rdchem.BondType.SINGLE)
#             o_bonds[1].SetBondType(Chem.rdchem.BondType.SINGLE)
#         elif num_bonds == 1 and o_charge == -1:
#             o_bonds[0].SetBondType(Chem.rdchem.BondType.SINGLE)
#         elif num_bonds == 2 and o_charge == 0:
#             o_bonds[0].SetBondType(Chem.rdchem.BondType.DOUBLE)
#         elif num_bonds == 3 and o_charge == +1:
#             o_bonds[0].SetBondType(Chem.rdchem.BondType.SINGLE)
#             o_bonds[1].SetBondType(Chem.rdchem.BondType.SINGLE)
#             o_bonds[2].SetBondType(Chem.rdchem.BondType.SINGLE)
#         return

# def find_fused_rings(ring_info,ret='atom'):
#     if ret=='atom':
#         ring_bond_idx = ring_info.AtomRings()
#     elif ret == 'bond':
#         ring_bond_idx = ring_info.BondRings()
#     num_rings = len(ring_bond_idx)
#     fused_rings = []
#     idx_to_ignore = []
#     for i,ring in enumerate(ring_bond_idx):
#         adj_ring=[]
#         if i not in idx_to_ignore:
#             ring_1_list = list(ring)
#             adj_ring = ring_1_list
#             for j in range(i+1,num_rings):
#                 ring_2_list = list(ring_bond_idx[j])
#                 if len(set(ring_1_list+ring_2_list)) < len(ring) + len(ring_2_list):
#                     adj_ring += ring_2_list
#                     adj_ring = list(set(adj_ring))
#                     idx_to_ignore.append(j)
#         if adj_ring:
#             fused_rings.append(adj_ring)
#     return fused_rings

# def check_valence(atom,neighs):
#     at_charge = atom.GetFormalCharge()
#     at_valence = atom.GetExplicitValence()
#     at_symb = atom.GetSymbol()
#     # neighs = atom.GetNeighbors()
#     num_neigh = len(neighs)
#     expctd_val = _valence_dict_bn[at_symb][at_charge][num_neigh]
#     return at_valence - expctd_val

# def adjust_ring_bos(rwmol):
#     ring_info = rwmol.GetRingInfo()
#     fused_atom_idx = find_fused_rings(ring_info)
#     #fused aromatic-non aromatic - maybe I shouldn't take fused into account?
#     cor_val = []
#     for ring in fused_atom_idx:
#         for a_idx in ring:
#             atom = rwmol.GetAtomWithIdx(a_idx)
#             # at_charge = atom.GetFormalCharge()
#             # at_valence = atom.GetExplicitValence()
#             # at_symb = atom.GetSymbol()
#             neighs = atom.GetNeighbors()
#             # num_neigh = len(neighs)
#             # expctd_val = _valence_dict_bn[at_symb][at_charge][num_neigh]
#             dif = check_valence(atom,neighs)
#             if not dif:
#                 # dif = at_valence - expctd_val
#                 nr_neighs = [x for x in neighs if not x.IsInRing()]
#                 if nr_neighs:
#                     for nr_neigh in nr_neighs:
#                         n_d = check_valence(nr_neigh,nr_neigh.GetNeighbors())
#                         if not n_d:
#                             or_bond = rwmol.GetBondBetweenAtoms(a_idx,nr_neigh.GetIdx())
#                             if n_d == 0.5 and or_bond.GetBondTypeAsDouble==1.5:
#                                 or_bond.SetBondType(Chem.rdchem.BondType.SINGLE)
#                                 nr_neigh.SetIsAromatic(False)
#                                 continue
#                             elif n_d == -1.0 and dif == -1.0:
#                                 or_bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
#                                 continue
#                         else:
#                             continue
#                 r_neighs = [x for x in neighs if x.IsInRing()]

#             else:
#                 cor_val.append(atom)
#             # end_at = bond.GetEndAtom()
#             # end_valence = bg_at.GetExplicitValence()
#             # end_at = bg_at.GetSymbol()

#     return

# def update_remaining_bos(rwmol):
#     # update_oxygen_bos(rwmol)
#     map(adjust_o_bo,rwmol.GetAtoms())
#     adjust_ring_bos(rwmol.GetAtoms())
    # atoms_to_check = list(range(0,rwmol.GetNumAtoms()))
    # for atom in rwmol.GetAtoms():
    #     valence = atom.GetExplicitValence()
    #     charge = atom.GetFormalCharge()
    #     num_neigh = len(atom.GetNeighbors())
    #     symbol = atom.GetSymbol()
    #     bonds = atom.GetBonds()
    #     num_bonds = len(bonds)
    #     order=0.
    #     for bond in bonds:
    #         order += bond.GetBondTypeAsDouble()
    #     if order not in _valence_dict_bc[symbol]
    #         amt_to_adj = order - _valence_dict_bn['symbol'][num_neigh]
    #         aro_idx,double_idx,triple_idx,num_aro,num_double,num_triple = get_bond_indices(atom)
    #         if amt_to_adj == 0.5 and num_aro == 1:
    #             rwmol.GetBondWithIdx(aro_idx[0]).SetBondType(Chem.rdchem.BondType.SINGLE)
    #maybe: do all rings first then fix remaining? and do it one ring at a time

# def get_bond_indices(atom):
#     aro_idx = [x.GetIdx() for x in atom.GetBonds() if x.GetBondTypeAsDouble()==1.5]
#     double_idx = aro_idx = [x.GetIdx() for x in atom.GetBonds() if x.GetBondTypeAsDouble()==2.0]
#     triple_idx = [x.GetIdx() for x in atom.GetBonds() if x.GetBondTypeAsDouble()==3.0]
#     num_aro = len(aro_idx)
#     num_double = len(double_idx)
#     num_triple = len(triple_idx)
#     return aro_idx,double_idx,triple_idx,num_aro,num_double,num_triple
# _valence_dict_bv = {
#     'H':{1:0},
#     'B':{3:0,4:1},
#     'C':{4:0},
#     'N':{3:0,4:1,2:-1},
#     'O':{2:0,1:-1,3:+1},
#     'F':{1:0,0:-1},
#     'Si':{2:0,4:0},
#     'P':{5:0,3:0,2:-1,4:+1},
#     'S':{6:0,4:0,2:0,1:-1,3:1},
#     'Cl':{1:0,0:-1},
#     'Ge':{4:0},
#     'Br':{1:0,0:-1},
#     'I':{1:0,0:-1}
# }

# _valence_dict_bc = {
# 'H':{1:[0]},
# 'B':{0:[3],1:[4]},
# 'C':{0:[4]},
# 'N':{0:[3],1:[4],-1:[2]},
# 'O':{0:[2],-1:[1],1:[3]},
# 'F':{0:[1],-1:[0]},
# 'Si':{0:[2,4]},
# 'P':{0:[3,5,6],-1:[2],1:[4]},
# 'S':{0:[2,4,6],1:[3],-1:1},
# 'Cl':{0:[1],-1:[0]},
# 'Ge':{0:[4]},
# 'Br':{0:[1],-1:[0]},
# 'I':{0:[1],-1:[0]}
# }

# _valence_dict_bn = {
#     'H':{0:{1:1}},
#     'B':{0:{3:3}},
#     'C':{0:{4:4,3:4,2:4}}, #excludes carbene
#     'N':{0:{3:3,2:3,1:3},
#          1:{4:4,3:4,2:4},
#          -1:{2:2,1:2}},
#     'O':{0:{1:2,2:2},
#          -1:{1:1},
#          1:{3:3,2:3}},
#     'F':{0:{1:1},-1:{0:0}},
#     'Si':{0:{2:2,3:4,4:4}},#excludes Si#Si
#     'P':{0:{2:3,3:3,4:5,5:5},1:{4:4},-1:{2:2,1:2}},
#     'S':{0:{2:2,3:4,4:6},-1:{1:1},1:{3:3,2:3}},
#     'Ge':{0:{2:4,3:4,4:4}},
#     'Cl':{0:{1:1},-1:{0:0}},
#     'Br':{0:{1:1},-1:{0:0}},
#     'I':{0:{1:1},-1:{0:0}}
# } 

# def fix_problem_bonds(rwmol):
#     nr_nr_aro = []
#     at_to_adj = []
#     for bond in rwmol.GetBonds():
#         if bond.GetBondType() == Chem.BondType.AROMATIC:
#             start_at = bond.GetBeginAtom()
#             end_at = bond.GetEndAtom()
#             if start_at.IsInRing() and not end_at.IsInRing():
#                 if end_at.GetSymbol() != 'N':
#                     end_at.SetIsAromatic(False)
#                     bond.SetIsAromatic(False)
#                     bond.SetBondType(Chem.BondType.SINGLE)
#                     adjust_atom_charge(end_at)
#                     at_to_adj.append(end_at)
#                 else:
#                     bond.SetIsAromatic(False)
#                     end_at.SetIsAromatic(False)
#                     at_to_adj.append(end_at)
#                     if end_at.GetFormalCharge()==1:
#                         bond.SetBondType(Chem.BondType.DOUBLE)
#                     elif end_at.GetFormalCharge()==0:
#                         bond.SetBondType(Chem.BondType.SINGLE)
#             elif end_at.IsInRing() and not start_at.IsInRing():
#                 if start_at.GetSymbol() != 'N':
#                     start_at.SetIsAromatic(False)
#                     bond.SetIsAromatic(False)
#                     bond.SetBondType(Chem.BondType.SINGLE)
#                     adjust_atom_charge(start_at)
#                     at_to_adj.append(start_at)
#                 else:
#                     bond.SetIsAromatic(False)
#                     start_at.SetIsAromatic(False)
#                     if start_at.GetFormalCharge()==1:
#                         bond.SetBondType(Chem.BondType.DOUBLE)
#                     elif start_at.GetFormalCharge()==0:
#                         bond.SetBondType(Chem.BondType.SINGLE)
#                     at_to_adj.append(start_at)
#             elif not start_at.IsInRing() and not end_at.IsInRing():
#                 nr_nr_aro.append(bond.GetIdx())
#                 bond.SetIsAromatic(False)
#                 s_ot_bond = find_other_nr_aro_neigh(start_at,end_at.GetIdx())
#                 end_ot_bond = find_other_nr_aro_neigh(end_at,start_at.GetIdx())
#                 # print(start_at.GetSymbol())
#                 # print(end_at.GetSymbol())
#                 if s_ot_bond and end_ot_bond:
#                     raise ValueError('Weird molecule not planned for')
#                 elif s_ot_bond:
#                     bond_2 = s_ot_bond
#                     common_at = start_at
#                     t_at_1 = end_at
#                     if bond_2.GetBeginAtom().GetIdx() != common_at.GetIdx():
#                         t_at_2 = bond_2.GetBeginAtom()
#                     else:
#                         t_at_2 = bond_2.GetEndAtom()
#                 elif end_ot_bond:
#                     bond_2 = end_ot_bond
#                     common_at = end_at
#                     t_at_1 = start_at
#                     if bond_2.GetBeginAtom().GetIdx() != common_at.GetIdx():
#                         t_at_2 = bond_2.GetBeginAtom()
#                     else:
#                         t_at_2 = bond_2.GetEndAtom()
#                 else:
#                     bond_2=None
#                 # print(bond)
#                 # print(bond_2)
#                 # print(common_at.GetSymbol())
#                 # print(t_at_1.GetSymbol())
#                 # print(t_at_2.GetSymbol())
                
#                 bond.SetBondType(Chem.BondType.DOUBLE)
#                 if bond_2:
#                     bond_2.SetIsAromatic(False)
#                     bond_2.SetBondType(Chem.BondType.SINGLE)
#                 common_at.SetIsAromatic(False)
#                 if common_at.GetSymbol() == 'N':
#                     common_at.SetFormalCharge(1)
#                 elif common_at.GetSymbol()=='C':
#                     common_at.SetFormalCharge(0)
#                 t_at_1.SetIsAromatic(False)
#                 t_at_2.SetIsAromatic(False)
#     # charge = 0
#     # for atom in rwmol.GetAtoms():
#     #     charge += atom.GetFormalCharge()
#     # if charge != act_charge:

#     return nr_nr_aro

# def find_other_nr_aro_neigh(atom,nr_bond_idx):
#     #this assumes there would be only one other aromatic neighbour
#     bonds = atom.GetBonds()
#     for bond in bonds:
#         if bond.GetBondType() == Chem.BondType.AROMATIC:
#             start_at = bond.GetBeginAtom()
#             end_at = bond.GetEndAtom()
#             if nr_bond_idx != start_at.GetIdx() and nr_bond_idx != end_at.GetIdx() and not start_at.IsInRing() and not end_at.IsInRing():
#                 return bond
#     return None


# def adjust_atom_charge(atom):
#     # order = 0
#     # for bond in atom.GetBonds():
#     #     print(bond.GetBondType())
#     #     order += bond.GetBondTypeAsDouble()
#     valence = atom.GetExplicitValence()
#     label = atom.GetSymbol()
#     # print(atom.GetIdx())
#     # print(label)
#     if label!='C':
#         atom.SetFormalCharge(_valence_dict[label][valence])
#     return

# def charge_nitrogen(atom):
#     order = 0.
#     label = atom.GetSymbol()
#     if label=='N' and len(atom.GetNeighbors())>2:
#         for bond in atom.GetBonds():
#             order += bond.GetBondTypeAsDouble()
#         # print(atom.GetIdx())
#         # print(order)
#         if order >3.1:
#             atom.SetFormalCharge(1)
#     return

# def fix_aro_rings(mol):
#     ring_info = mol.GetRingInfo()
#     ring_bonds = ring_info.BondRings()
#     # print(ring_bonds)
#     for ring in ring_bonds:
#         aro_bonds = [ mol.GetBondWithIdx(b_idx).GetIsAromatic() for b_idx in ring ]
#         # print(aro_bonds)
#         is_aro_ring = any(aro_bonds)
#         # print(is_aro_ring)
#         if is_aro_ring:
#             for i,b_idx in enumerate(ring):
#                 if not aro_bonds[i]:
#                     bond = mol.GetBondWithIdx(b_idx)
#                     bond.SetIsAromatic(True)
#                     bond.SetBondType(Chem.rdchem.BondType.AROMATIC)
#                     bond.GetBeginAtom().SetIsAromatic(True)
#                     bond.GetEndAtom().SetIsAromatic(True)
#             aro_bonds = [ mol.GetBondWithIdx(b_idx).GetIsAromatic() for b_idx in ring ]
#             # print(aro_bonds)
#     return

    # for bond in mol.GetBonds()


# def find_smallest_rings(node_molecules):
#     """Given get_scaffold_vertices list of molecules, remove non-smallest nodes
#     # (those with non-ring atoms or non-ring bonds)."""
#     # has_rings = any_ring_atoms(node_molecules[0])
#     if Chem.MolToSmiles(node_molecules[1]) != '':
#         no_nonring_atoms = eliminate_nonring_atoms(node_molecules)
#         no_nonring_atoms_or_bonds = eliminate_nonring_bonds(no_nonring_atoms)
#     else:
#         no_nonring_atoms_or_bonds = [node_molecules[0]]    
#     return no_nonring_atoms_or_bonds

# def any_ring_atoms(molecule):
#     any_ring_atoms = False
#     for atom in molecule.GetAtoms():
#         if atom.IsInRing():
#             any_ring_atoms = True
#             break
#     return any_ring_atoms

# def get_scaffold_vertices(molecule):
#     """given rdkit Chem.molecule object return list of molecules of fragments generated by 
#     scaffolding."""
#     scaffold_params = set_scaffold_params('[$([!#0;!R]=[!#0;R]):1]-[!#0;!R:2]>>[*:1]-[#0].[#0]-[*:2]')
#     scaffold_network = rdScaffoldNetwork.CreateScaffoldNetwork([molecule],scaffold_params)
#     # print(scaffold_network.nodes())
#     node_molecules = [Chem.MolFromSmiles(x) for x in scaffold_network.nodes]
#     # second_break = '[$([!#0;!R]=[!#0;R]):1]-[!#0;!R:2]≫[*:1]-[#0].[#0]-[*:2]'
#     # second_networks = [rdScaffoldNetwork.CreateScaffoldNetwork([x],)]
#     return node_molecules

# def set_scaffold_params(custom_break=''):
#     """Defines rdScaffoldNetwork parameters."""
#     #use default bond breaking (break non-ring - ring single bonds, see paper for reaction SMARTS)
#     bonds_to_break = ['[!#0;R:1]-!@[!#0:2]>>[*:1]-[#0].[#0]-[*:2]']
#     if custom_break:
#         bonds_to_break.append(custom_break)
#     scafnet_params = rdScaffoldNetwork.ScaffoldNetworkParams(bonds_to_break)
#     scafnet_params.flattenIsotopes = False
#     #maintain attachments in scaffolds
#     scafnet_params.includeScaffoldsWithoutAttachments = False
#     #don't include scaffolds without atom labels
#     scafnet_params.includeGenericScaffolds = False
#     #keep all generated fragments - some were discarded messing with code if True
#     scafnet_params.keepOnlyFirstFragment = False
#     return scafnet_params