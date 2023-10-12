"""
utils.py

Various utilities used in fragmenting molecules - retrieving information from molecules
Identifying small parts of molecules, and tools for minor manipulations of SMILES or molecules
"""
import ast
import os

import pandas as pd
from rdkit import Chem  # pylint:disable=import-error
from rdkit.Chem import (  # pylint:disable=import-error # search for rdScaffoldAttachment points * to remove; pylint:disable=import-error; rdDetermineBonds,
    AllChem,
    rdDetermineBonds,
    rdqueries,
)


def get_molecules_atomicnum(molecule):
    """Given molecule object, get list of atomic numbers."""
    # a
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
    qa = rdqueries.AtomNumEqualsQueryAtom(0)  # define query for atomic number 0
    if len(rwmol.GetAtomsMatchingQuery(qa)) > 0:  # if there are matches
        query_match = rwmol.GetAtomsMatchingQuery(qa)
        rm_at_idx = []
        for atom in query_match:  # identify atoms to be removed
            rm_at_idx.append(atom.GetIdx())
            # remove starting from highest number so upcoming indices not affected
        rm_at_idx_sort = sorted(rm_at_idx, reverse=True)
        # e.g. need to remove 3 and 5, if don't do this and you remove 3,
        # then the 5 you want to remove is now 4, and you'll remove wrong atom
        for idx in rm_at_idx_sort:  # remove atoms
            rwmol.RemoveAtom(idx)
    return rwmol


def mol_with_atom_index(mol):
    """Label a molecule with map numbers for order atoms are in molecule.

    Useful for tracking atoms after fragmenting

    Args:
        mol: an rdkit molecule object

    Returns:
        rdkit molecule object with AtomMapNum set so that atom 0
        has map number 1, atom 1 has map number 2...etc

    """
    # from https://www.rdkit.org/docs/Cookbook.html
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    return mol


def _get_charge_from_cml(cml_file):
    with open(cml_file, encoding="utf-8") as file:
        for line in file:
            if "formalCharge" in line:
                split_line = line.split(" ")
                for word in split_line:
                    if "formalCharge" in word:
                        charge = int(
                            word.replace("formalCharge=", "")
                            .replace(">\n", "")
                            .replace('"', "")
                        )
                        break
    return charge


def mol_from_xyzfile(xyz_file: str, cml_file):
    """Attempt to create a molecule from an xyz file by automatically determining bond orders"""
    # raw_mol = Chem.MolFromXYZFile('DUDE_67368827_adrb2_decoys_C19H25N3O4_CIR.xyz')
    # mol = Chem.Mol(raw_mol)
    # rdDetermineBonds.DetermineConnectivity(mol)
    # rdDetermineBonds.DetermineBondOrders(mol)
    mol = Chem.Mol(Chem.MolFromXYZFile(xyz_file))
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=_get_charge_from_cml(cml_file))
    except:  # pylint:disable=bare-except
        if os.path.isfile("error_log.txt"):
            with open("error_log.txt", "a", encoding="utf-8") as er_file:
                er_file.write(
                    f"Could not determine bond orders from xyz for {cml_file}\n"
                )
        else:
            with open("error_log.txt", "w", encoding="utf-8") as er_file:
                er_file.write(f"Could not determine bond orders for {cml_file}\n")
        return None

    atomic_symbols = []
    xyz_coordinates = []
    ats_read = 0
    num_atoms = mol.GetNumAtoms()
    with open(xyz_file, encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if ats_read < num_atoms and line_number > 1:
                ats_read += 1
                atomic_symbol, x, y, z = line.split()[:4]
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])
            elif ats_read == num_atoms:
                break
    return {
        "Molecule": mol_with_atom_index(mol),
        "xyz_pos": xyz_coordinates,
        "atomic_symbols": atomic_symbols,
    }


# from https://github.com/rdkit/rdkit/issues/2413
# conf = m.GetConformer()


# in principal, you should check that the atoms match
# for i in range(m.GetNumAtoms()):
#     print(i)
#     x,y,z = xyz_coordinates[i]
#     conf.SetAtomPosition(i,Point3D(x,y,z))


def get_cml_atom_types(cml_file):
    """Extract atom types from cml file

    Args:
        cml_file: cml file name

    Returns:
        Data Frame column whose elements are tuples of
        atom types as defined by Retrievium. (atom number, type, valence)

    """
    n_atl = 0
    type_list = []
    idx_list = []
    with open(cml_file, encoding="utf-8") as file:
        for line in file:
            if "atomTypeList" in line:
                n_atl += 1
                if n_atl == 2:
                    break
            elif n_atl == 1:
                split_line = line.split()
                idx_list.append(
                    int(split_line[1].split("=")[1].replace('"', "").replace("a", ""))
                    - 1
                )
                at_label = split_line[2].split("=")[1].replace('"', "")
                at_type = int(split_line[3].split("=")[1].replace('"', ""))
                at_valence = int(
                    split_line[4].split("=")[1].split("/")[0].replace('"', "")
                )
                type_list.append((at_label, at_type, at_valence))
    temp_frame = pd.DataFrame(list(zip(idx_list, type_list)), columns=["idx", "type"])
    temp_frame.sort_values(by="idx", inplace=True)
    return temp_frame["type"]


def add_cml_atoms_bonds(el_list, bond_list):
    """create a molecule from cml file, building it one atom at a time then
    adding in bond orders.

    Note: Bond orders from Retrievium cml are problematic at times.

    Recommended construction is to build atoms and connectivity from cml file
    Then assign bond orders based on template smiles also found in cml file"""
    flag = 1
    for atom in el_list:
        if flag:
            mol = Chem.MolFromSmiles(atom)
            rwmol = Chem.RWMol(mol)
            rwmol.BeginBatchEdit()
            flag = 0
        else:
            rwmol.AddAtom(Chem.Atom(atom))
    # mw.AddBond(6,7,Chem.BondType.SINGLE)
    for bond in bond_list:
        if bond[2] == "S":
            rwmol.AddBond(bond[0] - 1, bond[1] - 1, Chem.BondType.SINGLE)
        elif bond[2] == "D":
            rwmol.AddBond(bond[0] - 1, bond[1] - 1, Chem.BondType.DOUBLE)
        elif bond[2] == "T":
            rwmol.AddBond(bond[0] - 1, bond[1] - 1, Chem.BondType.TRIPLE)
        elif bond[2] == "A":
            rwmol.AddBond(bond[0] - 1, bond[1] - 1, Chem.BondType.AROMATIC)
    rwmol.CommitBatchEdit()
    return rwmol


def _add_cml_single_atoms_bonds(el_list, bond_list):
    """Build a mol from list of atoms and bonds, one atom at a time
    Bonds assigned are only single bonds.

    Adjust bonds after with _modAssignBondOrdersFromTemplate"""
    rwmol = Chem.RWMol(Chem.Mol())
    for atom in el_list:
        rwmol.AddAtom(Chem.Atom(atom))
    # mw.AddBond(6,7,Chem.BondType.SINGLE)
    for bond in bond_list:
        rwmol.AddBond(bond[0] - 1, bond[1] - 1, Chem.BondType.SINGLE)
    rwmol.CommitBatchEdit()
    return rwmol


def smiles_from_cml(cml_file):
    """Finds the Retreivium input SMILES in a cml file

    Args:
        cml_file: cml file name

    Returns:
        string of the input SMILES code tagged in the file as retrievium:inputSMILES

    Note:
        Must be used on .cml files from the Retrievium database https://retrievium.ca

    """
    flag = 0
    with open(cml_file, encoding="utf-8") as file:
        for line in file:
            if "retrievium:inputSMILES" in line:
                flag = 1
            elif flag == 1:
                smile = line.split(">")[1].split("<")[0]
                break
    return smile


def mol_from_cml(cml_file, input_type="cmlfile"):
    """Creates a molecule from a cml file and returns atoms, xyz and types

    Builds molecule one atom at a time connected by single bonds
    Then determines bond orders by mapping to a template smiles in the cml
    Finally, updates property cache, initializes ring info, and sanitizes mol
    If no match between SMILEs and connectivity, returns None and writes
    error to file

    Args:
        cml_file: cml file name or dictionary containing cml data
        input_type: cmlfile or cmldict. Use cmlfile if just raw cml file, use cmldict if dictionary

    Returns:
        list with elements: [Molecule, list of atom symbols in molecule,
        list of xyz coords of atoms in molecule,
        list of atom types of atoms in molecule]

    Note:
        list order matches mol numbering in cml

    """

    if input_type == "cmlfile":
        xyz_coords, at_types, bond_list, el_list, _ = data_from_cml(cml_file, True)
        smile = smiles_from_cml(cml_file)
    elif input_type == "cmldict":
        xyz_coords = cml_file["geom"]
        at_types = cml_file["atom_types"]
        bond_list = cml_file["bonds"]
        el_list = cml_file["labels"]
        # charge = cml_file["charge"]
        smile = cml_file["smiles"]
    rwmol = _add_cml_single_atoms_bonds(el_list, bond_list)
    for atom in rwmol.GetAtoms():
        atom.SetNoImplicit(True)

    rwmol2 = Chem.RemoveHs(rwmol, implicitOnly=True, updateExplicitCount=False)
    template = AllChem.MolFromSmiles(smile)
    bond_mol = _modAssignBondOrdersFromTemplate(template, rwmol2, cml_file)
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
        except:  # pylint:disable=bare-except
            _write_error(f"Could not sanitize {cml_file}\n")
            return [None, None, None, None]
        return [mol_with_atom_index(bond_mol), el_list, xyz_coords, at_types]
        # if os.path.isfile('error_log.txt'):
        #     er_file = open('error_log.txt','a')
        #     er_file.write(f'No match between template smiles and connected geom for {cml_file}\n')
        #     er_file.close()
        # else:
        #     er_file = open('error_log.txt','w')
        #     er_file.write(f'No match between template smiles and connected geom for {cml_file}\n')
        #     er_file.close()
    return [None, None, None, None]


def _write_error(errormessage):
    """writes errormessage to errorlog.txt"""
    if os.path.isfile("error_log.txt"):
        with open("error_log.txt", "a", encoding="utf-8") as er_file:
            er_file.write(errormessage)
    else:
        with open("error_log.txt", "w", encoding="utf-8") as er_file:
            er_file.write(errormessage)


def _modAssignBondOrdersFromTemplate(refmol, mol, cml_file):
    """This is originally from RDKit AllChem module.
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
    # pylint:disable=too-many-branches
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
            except:  # pylint:disable=bare-except
                er_message = f"Could not sanitize {cml_file}\n"
                # smile = smiles_from_cml(cml_file)
                er_message += _at_num_er(refmol2, mol2)
                # smimol = Chem.MolFromSmiles()
                _write_error(er_message)
                return None
            if hasattr(mol2, "__sssAtoms"):  # pylint:disable=protected-access
                mol2.__sssAtoms = None  # pylint:disable=protected-access # we don't want all bonds highlighted
        else:
            er_message = (
                f"No match between template smiles and connected geom for {cml_file}\n"
            )
            er_message += _at_num_er(refmol2, mol2)
            _write_error(er_message)
            return None
    return mol2


def _at_num_er(refmol2, mol2):
    """check numbering of atoms"""
    er_message = ""
    smi_num = []
    smi_lab = []
    symb_list = []
    num_list = []
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
    for i, num in enumerate(num_list):
        j = smi_lab.index(symb_list[i])
        if smi_num[j] != num:
            er_message += f"Expected {smi_num[j]} {smi_lab[j]} from SMILES, observed {num_list[i]} in xyz\n"
    return er_message


def data_from_cml(cml_file, bonds=False):
    """Gets symbols, xyz coords, bonds and charge of a mol from cml file

    Args:
        cml_file: .cml filename

    Returns:
        list with relevant data from cml file.
        Elements in order are: molecular geometry, atom types,
        list of bonds, list of elements, charge


    Note:
        This is designed to parse files specifically from the Retrievium database
        https://retrievium.ca

        the SMILEs extracted is labelled with tag retrievium:inputSMILES

        The geometry extracted is from the third atomArray block in the .cml file

    Example Usage:
        >>> utils.data_from_cml(cml_file)

    """
    # pylint:disable=too-many-locals
    # pylint:disable=too-many-branches
    # pylint:disable=too-many-statements
    num_atom_array = 0
    geom_list = []
    n_atl = 0
    n_bary = 0
    type_list = []
    idx_list = []
    bond_list = []
    el_list = []
    with open(cml_file, encoding="utf-8") as file:
        for line in file:
            if "formalCharge" in line:
                split_line = line.split(" ")
                for word in split_line:
                    if "formalCharge" in word:
                        charge = int(
                            word.replace("formalCharge=", "")
                            .replace(">\n", "")
                            .replace('"', "")
                        )
                        continue
            if "atomArray" in line:
                num_atom_array += 1
                if num_atom_array == 5:
                    continue
            if num_atom_array == 5:
                quote_split = line.split('"')
                # maybe only do el_list with bonds?
                el_list.append(quote_split[3])
                x_split = quote_split[5]
                y_split = quote_split[7]
                z_split = quote_split[9]
                geom_list.append(
                    [
                        float(ast.literal_eval(x_split)),
                        float(ast.literal_eval(y_split)),
                        ast.literal_eval(z_split),
                    ]
                )
            elif num_atom_array == 6:
                if bonds:
                    if "bondArray" in line:
                        n_bary += 1
                        if n_bary in [1, 2]:
                            continue
                    elif n_bary == 1:
                        split_line = line.split()
                        at_1 = int(split_line[1].split('"')[1].replace("a", ""))
                        at_2 = int(split_line[2].replace('"', "").replace("a", ""))
                        b_ord = split_line[4].split('"')[1]
                        bond_list.append((at_1, at_2, b_ord))
                if "atomTypeList" in line:
                    n_atl += 1
                    if n_atl == 1:
                        continue
                    if n_atl == 2:
                        break
                elif n_atl == 1:
                    split_line = line.split()
                    idx_list.append(
                        int(
                            split_line[1]
                            .split("=")[1]
                            .replace('"', "")
                            .replace("a", "")
                        )
                        - 1
                    )
                    at_label = split_line[2].split("=")[1].replace('"', "")
                    at_type = int(split_line[3].split("=")[1].replace('"', ""))
                    at_valence = int(
                        split_line[4].split("=")[1].split("/")[0].replace('"', "")
                    )
                    type_list.append((at_label, at_type, at_valence))
    temp_frame = pd.DataFrame(list(zip(idx_list, type_list)), columns=["idx", "type"])
    temp_frame.sort_values(by="idx", inplace=True)
    if bonds:
        return [geom_list, list(temp_frame["type"]), bond_list, el_list, charge]
    return [geom_list, list(temp_frame["type"])]


def all_data_from_cml(data):
    """Gets symbols, xyz coords, bonds and charge of a mol from cml file

    Args:
        data: lines of a .cml file

    Returns:
        dictionary with relevant data from cml file.
        Keys included are 'geom', 'atom_types', 'bonds',
        'labels', 'charge', 'multiplicity', 'smiles'

    Note:
        Not used in :attr:`group_decomposition.fragfunctions.identify_connected_fragments`
        This is used in an AiiDA workflow employing this package

        This is designed to parse files specifically from the Retrievium database
        https://retrievium.ca

        the SMILEs extracted is labelled with tag retrievium:inputSMILES

        The geometry extracted is from the third atomArray block in the .cml file

    Example Usage:
        >>> utils.all_data_from_cml(cml_file)

    """
    # pylint:disable=too-many-locals
    # pylint:disable=too-many-branches
    # pylint:disable=too-many-statements
    num_atom_array = 0
    geom_list = []
    n_atl = 0
    n_bary = 0
    type_list = []
    idx_list = []
    bond_list = []
    el_list = []
    smi_flag = 0
    for line in data:
        if "formalCharge" in line:
            split_line = line.split(" ")
            for word in split_line:
                if "spinMultiplicity" in word:
                    multiplicity = int(
                        word.replace("spinMultiplicity=", "").replace('"', "")
                    )
                if "formalCharge" in word:
                    charge = int(
                        word.replace("formalCharge=", "")
                        .replace('"', "")
                        .replace(">", "")
                        .replace("\n", "")
                    )
                    continue
        if "retrievium:inputSMILES" in line:
            smi_flag = 1
        elif smi_flag == 1:
            smile = line.split(">")[1].split("<")[0]
            smi_flag = 2
        if "atomArray" in line:
            num_atom_array += 1
            if num_atom_array == 5:
                continue
        if num_atom_array == 5:
            quote_split = line.split('"')
            # maybe only do el_list with bonds?
            el_list.append(quote_split[3])
            x_split = quote_split[5]
            y_split = quote_split[7]
            z_split = quote_split[9]
            geom_list.append(
                [
                    float(ast.literal_eval(x_split)),
                    float(ast.literal_eval(y_split)),
                    float(ast.literal_eval(z_split)),
                ]
            )
        elif num_atom_array == 6:
            if "bondArray" in line:
                n_bary += 1
                if n_bary in [1, 2]:
                    continue
            elif n_bary == 1:
                split_line = line.split()
                at_1 = int(split_line[1].split('"')[1].replace("a", ""))
                at_2 = int(split_line[2].replace('"', "").replace("a", ""))
                b_ord = split_line[4].split('"')[1]
                bond_list.append((at_1, at_2, b_ord))
            if "atomTypeList" in line:
                n_atl += 1
                if n_atl == 1:
                    continue
                if n_atl == 2:
                    break
            elif n_atl == 1:
                split_line = line.split()
                idx_list.append(
                    int(split_line[1].split("=")[1].replace('"', "").replace("a", ""))
                    - 1
                )
                at_label = split_line[2].split("=")[1].replace('"', "")
                at_type = int(split_line[3].split("=")[1].replace('"', ""))
                at_valence = int(
                    split_line[4].split("=")[1].split("/")[0].replace('"', "")
                )
                type_list.append((at_label, at_type, at_valence))
    temp_frame = pd.DataFrame(list(zip(idx_list, type_list)), columns=["idx", "type"])
    temp_frame.sort_values(by="idx", inplace=True)

    return {
        "geom": geom_list,
        "atom_types": list(temp_frame["type"]),
        "bonds": bond_list,
        "labels": el_list,
        "charge": charge,
        "multiplicity": multiplicity,
        "smiles": smile,  # pylint:disable=used-before-assignment
    }


def xyz_list_to_str(xyz_list):
    """Convert 2d array to string [[a,b,c],[d,e,f]] -> a, b, c\n d, e, f\n"""
    outstr = ""
    for row in xyz_list:
        for at in row:
            outstr += str(at)
            outstr += ", "
        outstr += "\n"
    return outstr


def list_to_str(lst):
    """Convert a list to string. [a,b,c] -> a b c"""
    out_str = ""
    for it in lst:
        out_str += str(it) + " "
    return out_str


def mol_from_molfile(mol_file):
    """takes mol_file and returns mol wth atom numbers the same

    Modified for mol file structure from retrievium
    from stackexchange
    https://mattermodeling.stackexchange.com/questions/7234/how-to-input-3d-coordinates-from-xyz-file-and-connectivity-from-smiles-in-rdkit

    Args:
        mol_file: name of .mol file

    Returns:
        mol with AtomMapNumbers labeled by :attr:`group_decomposition.utils.mol_with_atom_index`

    """  # pylint:disable=line-too-long
    m = Chem.MolFromMolFile(mol_file, removeHs=False)
    if not m:
        raise ValueError(f"""Problem creating molecule from {mol_file}""")
    # this assumes whatever program you use doesn't re-order atoms
    #  .. which is usually a safe assumption
    #  .. so we don't bother tracking atoms
    m = mol_with_atom_index(m)
    atomic_symbols = []
    xyz_coordinates = []
    ats_read = 0
    num_atoms = m.GetNumAtoms()
    with open(mol_file, encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if ats_read < num_atoms and line_number > 3:
                ats_read += 1
                x, y, z, atomic_symbol = line.split()[:4]
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])
            elif ats_read == num_atoms:
                break
    return {"Molecule": m, "xyz_pos": xyz_coordinates, "atomic_symbols": atomic_symbols}


def xyz_from_cml(cml_file):
    """Extract xyz coordinates from cml file

    Args:
        cml_file: cml file name

    Returns:
        list of length 3 lists containing a molecule's xyz coordinates

    Note:
        Must be used on .cml files from the Retrievium database https://retrievium.ca

    """
    num_atom_array = 0
    geom_list = []
    with open(cml_file, encoding="utf-8") as file:
        for line in file:
            if "atomArray" in line:
                num_atom_array += 1
                if num_atom_array == 5:
                    continue
            if num_atom_array == 5:
                space_split = line.split()
                x_split = space_split[3].split("=")
                y_split = space_split[4].split("=")
                z_split = space_split[5].split("=")
                geom_list.append(
                    [
                        float(ast.literal_eval(x_split[1])),
                        float(ast.literal_eval(y_split[1])),
                        float(ast.literal_eval(z_split[1])),
                    ]
                )
            elif num_atom_array == 6:
                break
    return geom_list


def get_canonical_molecule(smile: str):
    """Ensures that molecule numbering is consistent with creating molecule from canonical
    SMILES for consistency."""
    mol = Chem.MolFromSmiles(smile)
    if mol:
        mol_smi = Chem.MolToSmiles(mol)  # molsmi is canonical SMILES
    else:
        raise ValueError(
            f"""{smile} is not a valid SMILES code or
                         rdkit cannot construct a molecule from it"""
        )
    # create canonical molecule numbering from canonical SMILES
    return Chem.MolFromSmiles(mol_smi)


def link_molecules(mol_1: Chem.Mol, mol_2: Chem.Mol, dl_1: int, dl_2: int):
    """Given two mols, each with dummy atoms that have dummyAtomLabels, link the molecules between
    the dummy atoms specified by labels dl_1 and dl_2

    Modified from https://www.oloren.ai/blog/add_rgroup.html
    Written by David Huang, Oloren AI, modified by Kevin Lefrancois-Gagnon

    Args:
        mol_1: Chem.Mol object
        mol_2: Chem.Mol object
        dl_1: the isotope of the dummy atom in mol_1 which will be replaced by mol_2
        dl_2: the isotope of the dummy atom in mol_2 which will be replaced by mol_1

    Returns:
        Chem.Mol object with mol_1 and mol_2 linked where dl_1 and dl_2 were"""
    # pylint:disable=too-many-locals
    # Loop over atoms until there are no wildcard atoms
    # Find wildcard atom if available, otherwise exit
    # We use the isotope here are FragmentOnBonds labels the dummy atoms by changing their isotope
    a = None
    for a_ in mol_1.GetAtoms():
        if a_.GetAtomicNum() == 0 and a_.GetIsotope() == dl_1:
            a = a_
            break
    if not a:
        raise ValueError(
            f"""Input molecule mol_1 does not have atom with dummy label {dl_1}"""
        )
    b = None
    for b_ in mol_2.GetAtoms():
        if b_.GetAtomicNum() == 0 and b_.GetIsotope() == dl_2:
            b = b_
            break
    if not b:
        raise ValueError(
            f"""Input molecule mol_1 does not have atom with dummy label {dl_2}"""
        )
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
    start = (
        b1.GetBeginAtomIdx()
        if b1.GetEndAtomIdx() == a1.GetIdx()
        else b1.GetEndAtomIdx()
    )

    b2 = a2.GetBonds()[0]
    end = (
        b2.GetBeginAtomIdx()
        if b2.GetEndAtomIdx() == a2.GetIdx()
        else b2.GetEndAtomIdx()
    )

    # Add the connection and remove original wildcard atoms
    m.AddBond(start, end, order=Chem.rdchem.BondType.SINGLE)
    m.RemoveAtom(a1.GetIdx())
    m.RemoveAtom(a2.GetIdx())

    return m


# def copy_molecule(molecule):
#     """create a copy of molecule object in new object(not pointer)"""
#     # see link https://sourceforge.net/p/rdkit/mailman/message/33652439/
#     return Chem.Mol(molecule)


def _clean_smile(trim_smi):
    """remove leftover junk from smiles when atom deleted."""
    trim_smi = trim_smi.replace("[*H]", "*")
    trim_smi = trim_smi.replace("[*H3]", "*")
    trim_smi = trim_smi.replace("[*H2]", "*")
    trim_smi = trim_smi.replace("[*H+]", "*")
    trim_smi = trim_smi.replace("[*H3+]", "*")
    trim_smi = trim_smi.replace("[*H2+]", "*")
    trim_smi = trim_smi.replace("[*H-]", "*")
    trim_smi = trim_smi.replace("[*H3-]", "*")
    trim_smi = trim_smi.replace("[*H2-]", "*")
    return trim_smi
