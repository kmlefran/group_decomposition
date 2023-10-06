"""
fragfunctions module

code used to generate fragments of molecules from SMILES code and analyze them

Main functions to call are:
identify_connected_fragments - takes one molecule SMILES, returns fragments with connections
count_uniques - takes output from above, removes attachments and counts unique fragments
count_groups_in_set - takes list of SMILES and counts unique fragments on set
"""
# pylint:disable=too-many-lines
import math
import os
import re
from collections import Counter

import numpy as np  # for arrays in fragment identification
import pandas as pd  # lots of work with data frames
from rdkit import Chem  # pylint:disable=import-error
from rdkit.Chem import AllChem, PandasTools, rdqueries  # pylint:disable=import-error

from group_decomposition import utils

_num_bonds_broken = 1

_H_BOND_LENGTHS = {
    # from Gaussview default cleaned bond length
    "C": 1.07,
    "O": 0.96,
    "N": 1.00,
    "F": 0.88,
    "Cl": 1.29,
    "B": 1.18,
    "Al": 1.55,
    "Si": 1.47,
    "P": 1.35,
    "S": 1.31,
}


def _generate_fragment_frame(fragment_smiles):
    """Given list of SMILES Generate output frame with SMILES codes and molecules for unique
    fragments."""
    frag_frame = pd.DataFrame(set(fragment_smiles), columns=["Smiles"])
    PandasTools.AddMoleculeColumnToFrame(
        frag_frame, "Smiles", "Molecule", includeFingerprints=True
    )
    frag_frame.drop(frag_frame.index[frag_frame["Smiles"] == "*"].tolist())
    return frag_frame


def _add_xyz_coords(frag_frame):
    """Given frag_frame with molecules, add xyz coordinates form MM94 optimization to it."""
    xyz_block_list = []
    query = rdqueries.AtomNumEqualsQueryAtom(0)
    for mol in frag_frame["Molecule"]:
        h_mol_rw = Chem.RWMol(mol)  # Change type of molecule object
        h_mol_rw = Chem.AddHs(h_mol_rw)  # Add hydrogens
        zero_at = h_mol_rw.GetAtomsMatchingQuery(query)  # Replace placeholder * with At
        for atom in zero_at:
            h_mol_rw.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(85)
        AllChem.EmbedMolecule(h_mol_rw)
        AllChem.MMFFOptimizeMolecule(h_mol_rw)  # Optimize with MMFF94
        xyz_block_list.append(
            AllChem.rdmolfiles.MolToXYZBlock(h_mol_rw)  # pylint:disable=no-member
        )  # Store xyz coordinates
    frag_frame["xyz"] = xyz_block_list
    return frag_frame


def _add_number_attachements(frag_frame):
    """Add number of attachments column to frag_frame, counting number of *."""
    attach_list = []
    for molecule in frag_frame["Molecule"]:
        attach = 0
        atoms = molecule.GetAtoms()
        for atom in atoms:
            if atom.GetAtomicNum() == 0:
                attach += 1
        attach_list.append(attach)
    frag_frame["numAttachments"] = attach_list
    return frag_frame


def _fragment_molecule(
    mol_list, patt, exld_ring=False, drop_parent=True, recombine_mono=False
):
    """Break molecules into fragments based on fragmenting pattern"""
    global _num_bonds_broken  # pylint:disable=global-statement
    out_mols = []
    pat_mol = Chem.MolFromSmarts(patt)
    # print(mol_list)
    for mol in mol_list:
        if mol.HasSubstructMatch(pat_mol):
            if exld_ring and mol.GetRingInfo().NumRings() > 0:
                continue
            bonds_at_idx = mol.GetSubstructMatches(pat_mol)
            bonds_to_break = [
                mol.GetBondBetweenAtoms(x[0], x[1]).GetIdx() for x in bonds_at_idx
            ]
            labels = [
                [x, x]
                for x in range(_num_bonds_broken, _num_bonds_broken + len(bonds_at_idx))
            ]
            _num_bonds_broken += len(bonds_at_idx)
            frag_mols_comb = Chem.FragmentOnBonds(
                mol, bondIndices=bonds_to_break, dummyLabels=labels
            )
            frag_smis = Chem.MolToSmiles(frag_mols_comb).split(".")
            # print(frag_smis)
            frag_mols = [Chem.MolFromSmiles(x) for x in frag_smis]
            if recombine_mono:
                frag_mols = _recombine_monoatomic(frag_mols)

            if not drop_parent:
                out_mols = out_mols + [mol]
            out_mols = out_mols + frag_mols
        else:
            out_mols = out_mols + [mol]
    return out_mols


def _recombine_monoatomic(frag_mols):
    """Take fragments and recombine"""
    # pylint:disable=too-many-nested-blocks
    for m in frag_mols:
        link = None
        if m.GetNumHeavyAtoms() == 1:
            dummy_atoms = [x for x in m.GetAtoms() if x.GetAtomicNum() == 0]
            dummy_atom_iso = [x.GetIsotope() for x in dummy_atoms]
            dummy_match = []
            for i, dum in enumerate(dummy_atom_iso):
                for mo in frag_mols:
                    if mo != m:
                        for at in mo.GetAtoms():
                            if at.GetAtomicNum() == 0 and at.GetIsotope() == dum:
                                dummy_match.append(mo)
            for i, iso in enumerate(dummy_atom_iso):
                if i == 0:
                    link = utils.link_molecules(m, dummy_match[i], iso, iso)
                else:
                    link = utils.link_molecules(link, dummy_match[i], iso, iso)
        if link:
            frag_mols.append(link)
            frag_mols.remove(m)
            for mo in dummy_match:
                frag_mols.remove(mo)
    return frag_mols


# def get_num_placeholders(mol):
#     """Find number of * in a molecule"""
#     n_p = 0
#     for atom in mol.GetAtoms():
#         if atom.GetAtomicNum() == 0:
#             n_p += 1
#     return n_p


def generate_molecule_fragments(
    mol,
    patt: str = "[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0;!#9;!#17;!#35;!#1]):2]",
    drop_parent: bool = False,
    recombine_mono: bool = True,
):
    r"""Fragment a molecule into constituent groups

    The molecule is first broken along ring-nonring single bonds,
    then single bonds to atoms double bonded to ring (e.g. C-N={ring}),
    then breaking based on the pattern provided.



    Args:
        mol: the rdkit molecule object to be fragmented
        patt: SMARTs string that matches the bonds to be broken
             after ring-non-ring are separated
             Defaults to breaking alkyl-non-alkyl bonds. See notes
        drop_parent: if False, do include the parent structure in
            the third break. Defaults to breaking identifying alkyl chains.
            If True, do not include the parents
        recombine_mono: If True (default), will recombine separated one heavy
             atom groups with the chains they are broken from in the last step

    Returns:
        list[mol]: a list of rdkit molecules generated by fragmenting

    Note:
        Bonds broken are labelled by integers. Breaking FragA-FragB-FragC will result in FragA-1 \*:1-FragB-\*:2, \*:2-FragC

        In this way fragments can be rejoined by recombining matching integers.
        The integers are added to the molecule as Isotopes

        In the last step, halides are not separated from the alkyl groups by default,
        and one-heavy atom groups are rejoined to acyclic portions of the molecule

    """
    global _num_bonds_broken  # pylint:disable=global-statement
    _num_bonds_broken = 0
    first_break = _fragment_molecule([mol], patt="[!#0;R:1]-!@[!#0;!#1:2]")
    second_break = _fragment_molecule(
        first_break, patt="[$([!#0;!R]=[!#0;R]):1]-[!#0;!#1;!R:2]"
    )
    third_break = _fragment_molecule(
        second_break,
        patt,
        exld_ring=True,
        drop_parent=drop_parent,
        recombine_mono=recombine_mono,
    )
    return third_break


def identify_connected_fragments(
    inp: str,
    keep_only_children: bool = True,
    bb_patt: str = "[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0;!#9;!#17;!#35;!#1]):2]",
    input_type: str = "smile",
    cml_file: str = "",
    include_parent: bool = False,
    aiida: bool = False,
) -> pd.DataFrame:
    r"""
    Given Smiles string, identify fragments in the molecule:

    Break all ring-non-ring atom single bonds.
    For atoms double bonded to rings, break their single bonds.
    For non-ring fragments, separate those into alkyl chains and hetero/double bonded atoms.
    (similar to Ertl functional groups)

    Args:
        input: a string containing either a smiles, .xyz, .mol or .cml filename for a given molecule
            update input_type below to match provided input
        keep_only_children: boolean, if True, when a group is broken down into its components
            remove the parent group from output. If False, parent group is retained
        bb_patt: string of SMARTS pattern for bonds to be broken in side chains and linkers
            defaults to cleaving sp3 carbon-((ring OR not sp3 carbon) AND not-placeholder/halogen/H)
        input_type: 'smile' if SMILES code or 'molfile' if .mol file, or 'xyzfile' if .xyz file, or 'cmlfile' if .cml file
            Note: xyz file REQUIRES cml_file to be provided as well
        cml_file: defaults to none, can be the cml file corresponding to the input .mol file
        include_parent: If True, include column in output frame repeating parent molecule
            intended use for True when merging multiple molecule fragment frames but need to retain a parent molecule object
        aiida: if True, format output to be able to be used in aiida database.
            That is, no molecule objects

    Returns:
        DataFrame with columms 'Smiles', 'Molecule', 'numAttachments' and 'xyz'
        Containing, fragment smiles, fragment Chem.Molecule object, number of \* placeholders,
        and rough xyz coordinates for the fragment is \* were At

    Note:
        Each bond breaking, connectivity is maintained through dummy atom labels.
        e.g. C-N -> C-[1*] N-[1*] - reattaching via the matching labels would reassemble the molecule

        currently will break apart a functional group if contains a ring-non-ring single bond.
        e.g. ring N-nonring S=O -> ring N-[1*] nonring S=O-[1*]

    """
    # pylint:disable=too-many-arguments
    # pylint:disable=too-many-branches
    if aiida:
        inp = inp.value
        keep_only_children = keep_only_children.value
        bb_patt = bb_patt.value
        input_type = input_type.value
        if cml_file:
            cml_file = cml_file.value
        include_parent = include_parent.value

    # ensure smiles is canonical so writing and reading the smiles will result in same number
    # ordering of atoms
    if input_type == "smile":
        mol = utils.get_canonical_molecule(inp)
        xyz_coords = []
    elif input_type in ["cmlfile", "cmldict"]:
        mol, atomic_symb, xyz_coords, atom_types = utils.mol_from_cml(
            inp, input_type=input_type
        )
        if mol is None:
            return None
    elif input_type == "molfile":
        # use coordinates in cml file provided if able, else use xyz from mol file
        mol_dict = utils.mol_from_molfile(inp)
        mol, atomic_symb = mol_dict["Molecule"], mol_dict["atomic_symbols"]
        if cml_file:
            xyz_coords, atom_types, _, _, _ = utils.data_from_cml(cml_file)
            # atom_types = utils.get_cml_atom_types(cml_file)
        else:
            xyz_coords = mol_dict["xyz_pos"]
    # elif input_type == "xyzfile":
    #     if not cml_file:
    #         raise ValueError("No cml file provided, expected one for xyz input type")
    #     mol_dict = utils.mol_from_xyzfile(xyz_file=inp, cml_file=cml_file)
    #     if mol_dict is None:
    #         return None
    #     mol, atomic_symb, xyz_coords = (
    #         mol_dict["Molecule"],
    #         mol_dict["atomic_symbols"],
    #         mol_dict["xyz_pos"],
    #     )
    #     atom_types = utils.get_cml_atom_types(cml_file)
    else:
        raise ValueError(
            f"""{input_type} should either be molfile, xyzfile, cmlfile, or a smile string"""
        )

    mol_frags = generate_molecule_fragments(
        mol, patt=bb_patt, drop_parent=keep_only_children
    )
    # initialize the output data frame
    frag_frame = _generate_fragment_frame([Chem.MolToSmiles(x) for x in mol_frags])
    # add hydrogens and xyz coordinates resulting from MMFF94 opt, changing placeholders to At
    # frag_frame = _add_xyz_coords(frag_frame)
    # count number of placeholders in each fragment - it is the number of places it is attached
    frag_frame = _add_number_attachements(frag_frame)
    if input_type in [
        "molfile",
        "xyzfile",
        "cmlfile",
        "cmldict",
    ]:  # clear map labels and add xyz coordinates that are available
        frag_frame = _add_frag_comp(frag_frame, mol)
        frag_frame["Smiles"] = frag_frame["Smiles"].map(
            _clear_map_number
        )  # (lambda x: _clear_map_number(x))
        frag_frame["xyz"] = frag_frame["Atoms"].map(
            lambda x: _add_rtr_xyz(x, xyz_coords)
        )
        frag_frame["Labels"] = frag_frame["Atoms"].map(
            lambda x: _add_rtr_label(x, atomic_symb)
        )
        frag_frame["Molecule"] = frag_frame["Molecule"].map(
            lambda x: _clear_map_number(x, "mol")
        )
        if cml_file or input_type == "cmlfile" or input_type == "cmldict":
            frag_frame["atom_types"] = frag_frame["Atoms"].map(
                lambda x: _add_rtr_type(x, atom_types)
            )
    if include_parent and not aiida:
        mol = _clear_map_number(mol, "mol")
        frag_frame["Parent"] = [mol] * len(frag_frame.index)
    if aiida:
        frag_frame = frag_frame.drop("Molecule", axis=1)
    return frag_frame


def _add_rtr_label(at_num_list, atomic_symb):
    out_list = []
    for atom in at_num_list:
        out_list.append(atomic_symb[atom - 1])
    return out_list


def _add_rtr_type(at_num_list, atom_types):
    out_list = []
    for atom in at_num_list:
        out_list.append(atom_types[atom - 1])
    return out_list


def _add_rtr_xyz(at_num_list, xyz_coords):
    """Construct string of atomSymbol x y z format, for use to map to frag_frame"""
    out_list = []
    for atom in at_num_list:
        out_list.append(xyz_coords[atom - 1])
    return out_list


def _clear_map_number(mol_input, ret_type="smi"):
    """Given str or Chem.Molecule input, remove atomMapnumbers from atoms and return
    smiles (ret_type='smi') or molecule object (ret_type = 'mol')"""
    if isinstance(mol_input, str):
        mol = Chem.MolFromSmiles(mol_input)
    else:
        mol = mol_input
    if not mol:
        raise ValueError(
            f"""Could not construct mol from {mol_input} in output frame"""
        )
    for atom in mol.GetAtoms():
        atom.ClearProp("molAtomMapNumber")
    if ret_type == "smi":
        return Chem.MolToSmiles(mol)
    if ret_type == "mol":
        return mol
    raise ValueError(f"""Invalid ret_type, expected smi or mol, got {ret_type}""")


def _get_atlabels_in_frag(molecule):
    """For a given molecule, extract list of atom map number for all non-H atoms"""
    # H's not included because the molecule object typically won't have explicit H
    out_list = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 0:
            out_list.append(int(atom.GetProp("molAtomMapNumber")))
    return out_list


def _add_frag_comp(frag_frame, mol):
    """Given frag_frame and mol, add Atoms col to frag_frame with indices of atoms starting at 1"""
    # create list of lists of indices of atoms in each fragment
    frag_atoms = []
    for frag_mol in frag_frame["Molecule"]:
        frag_atoms.append(_get_atlabels_in_frag(frag_mol))
    # iterate over atoms, adding in hydrogens to the fragment since the above won't include H
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            neigh_idx = atom.GetNeighbors()[0].GetIdx() + 1
            for i, frag in enumerate(frag_atoms):
                if neigh_idx in frag:
                    frag_atoms[i].append(atom.GetIdx() + 1)
                    break  # only need to get here once - H has only one bond
    frag_frame["Atoms"] = frag_atoms
    return frag_frame


def count_uniques(
    frag_frame: pd.DataFrame, drop_attachments: bool = False, uni_smi_type: bool = True
) -> pd.DataFrame:
    r"""Identify unique fragments in a frame and count the number of times they occur

    Given frag_frame output from :attr:`group_decomposition.fragfunctions.identify_connected_fragments`,
    remove dummy atom labels(and placeholders entirely if drop_attachments=True), then count unique fragments
    using SMILES to identify unique fragments

    Should also work on frames from :attr:`group_decomposition.fragfunctions.merge_uniques` or
    :attr:`group_decomposition.fragfunctions.count_groups_in_set` as well

    Args:
        frag_frame: frame resulting from :attr:`group_decomposition.fragfunctions.identify_connected_fragments` typically,
            or any similar frame with a list of SMILES codes in column ['Smiles']
        drop_attachments: boolean, if False, retains placeholder \* at points of attachment,
            if True, removes \* for fragments with more than one atom
        uni_smi_type: include atom types in determination of unique
            fragments. If false, only determine unique by SMILES

    Returns:
        pandas data frame with columns 'Smiles', 'count' and 'Molecule',
        containing the Smiles string, the number of times the Smiles was in frag_frame,
        and rdkit.Chem.Molecule object

    Note:
        if drop_attachments=False, similar fragments with different number/positions of
        attachments will not count as being the same.
        e.g. ortho-attached aromatics would not match with meta or para attached aromatics

        If you've ran this previously with uni_smi_type=True, running on the output frame
        (or other frame derived from such frame)
        with uni_smi_type=False will collapse the output uniques determined by SMILE only

    """
    # Change column indices to dict
    # pylint:disable=too-many-locals
    # pylint:disable=too-many-branches
    # pylint:disable=too-many-statements
    col_names = list(frag_frame.columns)
    # if frag_frame already has xyz coordinates keep those and don't use others
    # typically this will be if a mol and/or cml file was provided in frag_frame construction
    if "xyz" in col_names:
        xyz_inc = True
        xyz_list = frag_frame["xyz"]
    else:
        xyz_inc = False
    smile_list = frag_frame["Smiles"]
    if "Atoms" in col_names:
        atom_list = frag_frame["Atoms"]
        atoms_inc = True
    else:
        atoms_inc = False
    if "Labels" in col_names:
        labels_list = frag_frame["Labels"]
        labels_inc = True
    else:
        labels_inc = False
    if "Parent" in col_names:
        parent_list = frag_frame["Parent"]
        parent_inc = True
    else:
        parent_inc = False
    if "atom_types" in col_names:
        type_list = frag_frame["atom_types"]
        type_inc = True
    if "count" in col_names:
        count_list = frag_frame["count"]
        count_inc = True
    else:
        count_inc = False
    # print(type_list)
    no_connect_smile = []
    # Clean smiles - either removing placeholder entirely(drop_attachments True)
    # Or just removing the dummyAtomLabel (drop_attachments False)
    for smile in smile_list:
        if drop_attachments:
            no_connect_smile.append(_drop_smi_attach(smile))
        else:
            t_mol = Chem.MolFromSmiles(re.sub(r"\[[0-9]+\*\]", "*", smile))
            no_connect_smile.append(Chem.MolToSmiles(t_mol))
    # identify unique smiles and count number of times they occur
    # initialize lists to be used making frame
    unique_smiles = []
    unique_xyz = []
    unique_atoms = []
    unique_labels = []
    unique_parents = []
    unique_smiles_counts = []
    unique_types = []
    counter_list = []
    # Identify unique smiles, counting every occurence and adding xyz if included
    if uni_smi_type:
        ty_tup = list(map(tuple, frag_frame["atom_types"]))
        smi_ty = list(zip(no_connect_smile, ty_tup))
        for i, val in enumerate(smi_ty):
            smile = val[0]
            at_tys = val[1]
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
                if type_inc:
                    unique_types.append(type_list[i])
                    counter_list.append(Counter(at_tys))
                if count_inc:
                    unique_smiles_counts.append(count_list[i])
                else:
                    unique_smiles_counts.append(1)
            else:
                at_ty_count = Counter(at_tys)
                if at_ty_count in counter_list:
                    smi_ix = counter_list.index(at_ty_count)
                    if count_inc:
                        unique_smiles_counts[smi_ix] += count_list[i]
                    else:
                        unique_smiles_counts[smi_ix] += 1
                else:
                    unique_smiles.append(smile)
                    if xyz_inc:
                        unique_xyz.append(xyz_list[i])
                    if atoms_inc:
                        unique_atoms.append(atom_list[i])
                    if labels_inc:
                        unique_labels.append(labels_list[i])
                    if parent_inc:
                        unique_parents.append(parent_list[i])
                    # if type_inc:
                    #     unique_types.append(labels_list[i])
                    counter_list.append(Counter(at_tys))
                    unique_types.append(type_list[i])
                    if count_inc:
                        unique_smiles_counts.append(count_list[i])
                    else:
                        unique_smiles_counts.append(1)
    else:
        for i, smile in enumerate(no_connect_smile):
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
                if type_inc:
                    unique_types.append(type_list[i])
                if count_inc:
                    unique_smiles_counts.append(count_list[i])
                else:
                    unique_smiles_counts.append(1)
            else:
                smi_ix = unique_smiles.index(smile)
                if count_inc:
                    unique_smiles_counts[smi_ix] += count_list[i]
                else:
                    unique_smiles_counts[smi_ix] += 1
    # create output frame

    un_frame = _construct_unique_frame(
        uni_smi=unique_smiles,
        uni_smi_count=unique_smiles_counts,
        xyz=unique_xyz,
        atoms=unique_atoms,
        parents=unique_parents,
        labels=unique_labels,
        at_types=unique_types,
    )
    # if atoms_inc:
    #     un_frame['Atoms'] = unique_atoms
    # if labels_inc:
    #     un_frame['Labels'] = unique_labels
    # if parent_inc:
    #     un_frame['Parent'] = unique_parents
    # if type_inc:
    #     un_frame['atom_types'] = unique_types
    return un_frame


def _find_neigh_notin_frag(mol, at_list):
    # for atoms with one attachment point
    out_nbr = 0
    for idx in at_list:
        atom = mol.GetAtomWithIdx(idx - 1)
        nbrs = atom.GetNeighbors()
        for nbr in nbrs:
            nbr_idx = nbr.GetIdx() + 1
            if nbr_idx not in at_list:
                out_nbr = nbr_idx
                out_at = atom.GetIdx() + 1
                break
        if out_nbr:
            break
    return [out_at, out_nbr]


def _find_neigh_xyz(frag_frame, neigh_idx):
    """Given frag_frame and index of neighbor atom, find its xyz coordinates"""
    # column index
    atoms_idx = list(frag_frame.columns).index("Atoms")
    # determine which row contains the neighbor atom
    neigh_bool = list(frag_frame.apply(lambda row: neigh_idx in row[atoms_idx], axis=1))
    # print(neigh_bool)
    neigh_row = neigh_bool.index(True)
    list_xyz = frag_frame["xyz"][neigh_row]
    # find where in list neighbor atom is, return xyz at that index
    neigh_list_idx = frag_frame["Atoms"][neigh_row].index(neigh_idx)
    neigh_xyz = list_xyz[neigh_list_idx]
    return neigh_xyz


def _move_along_bond(at_xyz, neigh_xyz, at_symb):
    """Returns xyz coordinates of neigh_xyz moved along bond to H bond length"""
    at_np = np.array(at_xyz)
    nb_np = np.array(neigh_xyz)
    # vector for bond to move along
    bond = at_np - nb_np
    # bond length to find
    target_length = _H_BOND_LENGTHS[at_symb]
    steps = np.linspace(0.0, 1.0, 100)
    # step along bond starting at neigh_xyz in direction of at_xyz.
    # stop when bond length is target
    for s in steps:
        new_xyz = neigh_xyz + s * bond
        if abs(_get_dist(new_xyz, at_np) - target_length) < 0.01:
            end_xyz = new_xyz
    # return target point
    return end_xyz


def _get_dist(point_a, point_b):
    """Return distance btw two points"""
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2
        + (point_a[1] - point_b[1]) ** 2
        + (point_a[2] - point_b[2]) ** 2
    )


def _find_H_xyz(mol, at_list, xyz_list, frag_frame):
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
    # find attached atom index and neighbor index in molecule
    at_idx, neigh_idx = _find_neigh_notin_frag(mol, at_list)
    # get attached atom symbol
    symb = mol.GetAtomWithIdx(at_idx - 1).GetSymbol()
    # index of the attached atom in at_list, which is the same as xyz_list
    list_idx = at_list.index(at_idx)
    # attached atom xyz
    at_xyz = xyz_list[list_idx]
    # neighbor atom xyz
    neigh_xyz = _find_neigh_xyz(frag_frame, neigh_idx)
    # move neighbor atom xyz along bond to H bond length (Gaussview defaults)
    h_xyz = _move_along_bond(at_xyz, neigh_xyz, symb)
    return h_xyz


def _clean_molecule_name(smile):
    """Removes symbols in smile code unfit for file name"""
    smile = smile.replace("-", "Neg")
    smile = smile.replace("[", "-")
    smile = smile.replace("]", "-")
    smile = smile.replace("(", "-")
    smile = smile.replace(")", "-")
    smile = smile.replace("#", "t")
    smile = smile.replace("=", "d")
    smile = smile.replace("+", "Pos")
    smile = smile.replace("*", "Att")
    smile = smile.replace("@", "")
    return smile


# def generate_fragment_structures(
#     inp: str,
#     keep_only_children: bool = True,
#     bb_patt: str = "[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0;!#9;!#17;!#35;!#1]):2]",
#     input_type="smile",
#     cml_file="",
#     include_parent=True,
# ):
#     """generate fragments and return a dictionary of fragments"""
#     # pylint:disable=too-many-arguments
#     frag_frame = identify_connected_fragments(
#         inp=inp,
#         keep_only_children=keep_only_children,
#         bb_patt=bb_patt,
#         input_type=input_type,
#         cml_file=cml_file,
#         include_parent=include_parent,
#     )
#     mol = frag_frame.at[0, "Parent"]
#     frag_dict = output_ifc_dict(mol, frag_frame)
#     return frag_dict


# show


# def output_cgis_dicts(cgis_frame):
#     """generate fragments and return a dictionary of fragments"""
#     # should have parent col
#     on_at_frame = pd.DataFrame(cgis_frame[cgis_frame["numAttachments"] == 1])
#     col_names = list(on_at_frame.columns)
#     xyz_idx = col_names.index("xyz")
#     atoms_idx = col_names.index("Atoms")
#     parent_idx = col_names.index("Parent")
#     on_at_frame["H_xyz"] = on_at_frame.apply(
#         lambda row: _find_H_xyz(
#             row[parent_idx], row[atoms_idx], row[xyz_idx], cgis_frame
#         ),
#         axis=1,
#     )
#     on_at_frame["at_idx"] = on_at_frame.apply(
#         lambda row: _find_at_idx(row[parent_idx], row[atoms_idx]), axis=1
#     )
#     out_dict = {
#         re.sub(
#             r"\[[0-9]+\*\]", "*", on_at_frame.at[i, "Smiles"]
#         ): _write_frag_structure(
#             frag_mol=on_at_frame.at[i, "Molecule"],
#             xyz_list=on_at_frame.at[i, "xyz"],
#             symb_list=on_at_frame.at[i, "Labels"],
#             h_xyz=on_at_frame.at[i, "H_xyz"],
#             at_idx=on_at_frame.at[i, "at_idx"],
#             atom_types=frag_frame.at[i, "atom_types"],
#         )
#         for i in range(0, nrow)
#     }
#     return out_dict


def output_ifc_dict(mol, frag_frame: pd.DataFrame, done_smi: list[str]):
    """generate a dictionary containing identify_connected_fragment information

    Only new fragments are included. Previously parsed fragments are listed in done_smi.

    Args:
        mol: rdkit molecule object that was fragmented
        frag_frame: identify_connected_fragments frame generated for mol
        done_smi: list of fragments which have been identified already

    Returns:
        list containing dict for fragments and done_smi lengthened by the number of fragments done

    Note:
        mainly for use in generating information for unique fragments in
        an AiiDA workflow

    """
    on_at_frame = pd.DataFrame(frag_frame[frag_frame["numAttachments"] == 1])
    col_names = list(on_at_frame.columns)
    # Find indices of relevant columns
    xyz_idx = col_names.index("xyz")
    atoms_idx = col_names.index("Atoms")

    # Find H xyz position and index of atom bonded to H
    on_at_frame["H_xyz"] = on_at_frame.apply(
        lambda row: _find_H_xyz(mol, row[atoms_idx], row[xyz_idx], frag_frame), axis=1
    )
    on_at_frame["at_idx"] = on_at_frame.apply(
        lambda row: _find_at_idx(mol, row[atoms_idx]), axis=1
    )

    nrow = on_at_frame.shape[0]
    on_at_frame = on_at_frame.reset_index(drop=True)
    print(on_at_frame)
    # print(_clean_molecule_name(on_at_frame.at[0,'Smiles']))
    print(on_at_frame.at[0, "Molecule"])
    print(on_at_frame.at[0, "Labels"])
    print(on_at_frame.at[0, "xyz"])
    print(on_at_frame.at[0, "H_xyz"])
    print(on_at_frame.at[0, "at_idx"])
    print(on_at_frame.at[0, "atom_types"])

    out_dict = {}
    for i in range(0, nrow):
        smi = on_at_frame.at[i, "Smiles"]
        if smi not in done_smi:
            done_smi.append(smi)
            key = re.sub(r"\[[0-9]+\*\]", "*", smi)
            out_dict[key] = _write_frag_structure(
                frag_mol=on_at_frame.at[i, "Molecule"],
                xyz_list=on_at_frame.at[i, "xyz"],
                symb_list=on_at_frame.at[i, "Labels"],
                h_xyz=on_at_frame.at[i, "H_xyz"],
                at_idx=on_at_frame.at[i, "at_idx"],
                atom_types=frag_frame.at[i, "atom_types"],
            )
    return [out_dict, done_smi]


def _write_frag_structure(frag_mol, xyz_list, symb_list, h_xyz, at_idx, atom_types):
    """get output structure"""
    # pylint:disable=too-many-arguments
    print("in function")
    num_atoms = len(symb_list)
    # smile = re.sub('\[[0-9]+\*\]', '*', Chem.MolToSmiles(frag_mol,canonical=False))
    charge = Chem.GetFormalCharge(frag_mol)
    out_xyz = [xyz_list[at_idx], h_xyz]
    for i in range(num_atoms):
        if i != at_idx:
            out_xyz.append(xyz_list[i])
    # geom_frame = pd.DataFrame(out_xyz,columns=['x','y','z'])
    symb_list.insert(1, "H")
    return {
        "geom": out_xyz,
        "charge": charge,
        "atom_types": atom_types,
        "atom_symbols": symb_list,
    }


def output_ifc_gjf(
    mol,
    frag_frame,
    esm="wb97xd",
    basis_set="aug-cc-pvtz",
    wfx=True,
    n_procs=4,
    mem="3200MB",
    multiplicity=1,
):
    """Takes a fragmented molecule and outputs gjf files of the fragments with one
    attachment point.

    Hydrogen is added in place of the connection to the rest of
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

    Note:
        H position is set by taking the atom the fragment is bonded two, replacing it with H
        and moving that closer to the C until it reaches the default distance

        Default distances taken from Gaussview "clean" C-H, C-O, etc bond lengths

    """
    on_at_frame = pd.DataFrame(frag_frame[frag_frame["numAttachments"] == 1])
    col_names = list(on_at_frame.columns)
    # Find indices of relevant columns
    xyz_idx = col_names.index("xyz")
    atoms_idx = col_names.index("Atoms")
    labels_idx = col_names.index("Labels")
    mol_idx = col_names.index("Molecule")
    # Find H xyz position and index of atom bonded to H
    on_at_frame["H_xyz"] = on_at_frame.apply(
        lambda row: _find_H_xyz(mol, row[atoms_idx], row[xyz_idx], frag_frame), axis=1
    )
    on_at_frame["at_idx"] = on_at_frame.apply(
        lambda row: _find_at_idx(mol, row[atoms_idx]), axis=1
    )
    # hxyz = len(col_names)
    # atidx = len(col_names) + 1
    # print(on_at_frame)
    # write gjfs
    on_at_frame.apply(
        lambda row: _write_frag_gjf(
            frag_mol=row[mol_idx],
            xyz_list=row[xyz_idx],
            symb_list=row[labels_idx],
            h_xyz=row[len(col_names)],
            at_idx=row[len(col_names) + 1],
            esm=esm,
            basis_set=basis_set,
            wfx=wfx,
            n_procs=n_procs,
            mem=mem,
            multiplicity=multiplicity,
        ),
        axis=1,
    )


def _clean_basis(basis_set):
    basis_set = basis_set.replace("(", "")
    basis_set = basis_set.replace(")", "")
    basis_set = basis_set.replace(",", "")
    basis_set = basis_set.replace("+", "p")
    return basis_set


def _write_frag_gjf(
    frag_mol,
    xyz_list,
    symb_list,
    h_xyz,
    at_idx,
    esm="wb97xd",
    basis_set="aug-cc-pvtz",
    wfx=True,
    n_procs=4,
    mem="3200MB",
    multiplicity=1,
):
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
    # pylint:disable=too-many-locals
    num_atoms = len(symb_list)
    smile = re.sub(r"\[[0-9]+\*\]", "*", Chem.MolToSmiles(frag_mol, canonical=False))
    charge = Chem.GetFormalCharge(frag_mol)
    molecule_name = _clean_molecule_name(smile)
    # print(at_idx)
    # print(xyz_list[at_idx])
    # print(h_xyz)
    # build xyz of molecule
    out_xyz = [xyz_list[at_idx], h_xyz]
    for i in range(num_atoms):
        if i != at_idx:
            out_xyz.append(xyz_list[i])
    geom_frame = pd.DataFrame(out_xyz, columns=["x", "y", "z"])
    symb_list.insert(1, "H")
    geom_frame["Atom"] = symb_list
    geom_frame = geom_frame[["Atom", "x", "y", "z"]]
    # create file name
    clean_basis = _clean_basis(basis_set)
    new_file_name = "SubH" + "_" + molecule_name + "_" + esm + "_" + clean_basis
    if os.path.exists(new_file_name + ".gjf"):
        # print('deleting')
        os.remove(new_file_name + ".gjf")
    # write file
    with open(new_file_name + ".gjf", "a", encoding="utf-8") as f:
        f.write(f"%chk={new_file_name}" + ".chk\n")
        if n_procs:
            f.write(f"%nprocs={n_procs}\n")
        if mem:
            f.write(f"%mem={mem}\n")
        if wfx:
            f.write(f"#p {esm}/{basis_set} opt freq output=wfx\n")
        else:
            f.write(f"#p {esm}/{basis_set} opt freq\n")
        f.write("\n")
        f.write(f"{smile}")
        f.write("\n\n")
        f.write(f"{charge} {multiplicity}\n")
        dfAsString = geom_frame.to_string(header=False, index=False)
        f.write(dfAsString)
        f.write("\n\n")
        if wfx:
            f.write(new_file_name + ".wfx\n\n\n")
        else:
            f.write("\n\n\n")


def _find_at_idx(mol, at_list):
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
    # find
    at_idx = _find_neigh_notin_frag(mol, at_list)[0]
    list_idx = at_list.index(at_idx)
    return list_idx


def _construct_unique_frame(
    uni_smi: list[str],
    uni_smi_count: list[int],
    xyz=[],
    atoms=[],
    parents=[],
    labels=[],
    at_types=[],
) -> pd.DataFrame:
    """given smiles, counts and (optional) xyz coordinates, create frame"""
    # pylint:disable=dangerous-default-value
    uniquefrag_frame = pd.DataFrame(uni_smi, columns=["Smiles"])
    if xyz:
        uniquefrag_frame["xyz"] = xyz
    if atoms:
        uniquefrag_frame["Atoms"] = atoms
    if parents:
        uniquefrag_frame["Parent"] = parents
    if labels:
        uniquefrag_frame["Labels"] = labels
    if at_types:
        uniquefrag_frame["atom_types"] = at_types
    PandasTools.AddMoleculeColumnToFrame(
        uniquefrag_frame, "Smiles", "Molecule", includeFingerprints=True
    )
    uniquefrag_frame["count"] = uni_smi_count
    # if we don't have xyz already add them, from MMFF94 opt
    if not xyz:
        uniquefrag_frame = _add_xyz_coords(uniquefrag_frame)
    # count number placeholders
    uniquefrag_frame = _add_number_attachements(uniquefrag_frame)
    return uniquefrag_frame


def _drop_smi_attach(smile: str):
    """completely remove placeholder if number of non-placeholder in smiles is > 1"""
    mol = Chem.MolFromSmiles(smile)
    non_zero_atoms = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            non_zero_atoms += 1
    if non_zero_atoms > 1:
        temp = re.sub(r"\[[0-9]+\*\]", "", smile)
        t_mol = Chem.MolFromSmiles(re.sub(r"\(\)", "", temp))
        if t_mol is None:
            t_mol = Chem.MolFromSmiles(re.sub(r"\[[0-9]+\*\]", "*", smile))
            out_smi = Chem.MolToSmiles(t_mol)
            # Warning('Could not construct {smile} without attachments'.format(smile=smile))
        else:
            out_smi = Chem.MolToSmiles(t_mol)
    else:
        out_smi = Chem.MolToSmiles(
            Chem.MolFromSmiles(re.sub(r"\[[0-9]+\*\]", "*", smile))
        )
    return out_smi


def merge_uniques(
    frame1: pd.DataFrame, frame2: pd.DataFrame, uni_smi_ty=True
) -> pd.DataFrame:
    """Given two frames of unique fragments, identify shared unique fragments,
    merge count and frames together.

    Args:
        frame1: a frame output from count_uniques
        frame2: a distinct frame also from count_uniques
        uni_smi_ty: If True, include atom types in determination of unique
            fragments. If false, only determine unique by SMILES

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
        >>> frame1
        Smiles  count
        C       2
        C1CCC1  1

        >>> frame2
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
        rows_to_drop = _find_rows_to_drop(frame1, frame2, uni_smi_ty)
        merge_frame = rows_to_drop["merge_frame"]
        drop_frame_1 = frame1.drop(rows_to_drop["drop_rows_1"])
        drop_frame_2 = frame2.drop(rows_to_drop["drop_rows_2"])
        merge_frame = pd.concat([drop_frame_1, drop_frame_2, merge_frame])
        merge_frame.reset_index(inplace=True, drop=True)
        # _add_frame(drop_frame_1,merge_frame)
        # merge_frame = _add_frame(drop_frame_2,merge_frame)
    return merge_frame


def _find_rows_to_drop(frame_a: pd.DataFrame, frame_b: pd.DataFrame, uni_smi_ty=True):
    """for two frames, find rows with same smile, store what rows to drop in each"""
    rows_to_drop_one = []
    rows_to_drop_two = []
    col_names = list(frame_a.columns)
    merge_frame = pd.DataFrame(columns=col_names)
    frame_a_idx = list(frame_a.index)
    frame_b_idx = list(frame_b.index)
    for i, smi in enumerate(frame_a["Smiles"]):
        if smi in list(frame_b["Smiles"]):
            # print(f'i is {i}')
            if uni_smi_ty:
                smi_idx = [
                    x
                    for x in range(0, frame_b.shape[0])
                    if smi == list(frame_b["Smiles"])[x]
                ]
                ctr = [list(map(Counter, frame_b["atom_types"]))[x] for x in smi_idx]
                if list(map(Counter, frame_a["atom_types"]))[i] not in ctr:
                    continue
                j = smi_idx[ctr.index(list(map(Counter, frame_a["atom_types"]))[i])]
            else:
                j = list(frame_b["Smiles"]).index(smi)
            # print(f'j is {j}')
            # print(frame_a.shape)
            cum_count = (
                frame_a.at[frame_a_idx[i], "count"]
                + frame_b.at[frame_b_idx[j], "count"]
            )
            # print('cum_count is {cum_count}')
            merge_frame = pd.concat(
                [
                    merge_frame,
                    pd.DataFrame(
                        [list(frame_a.iloc[frame_a_idx[i]])], columns=col_names
                    ),
                ]
            )
            merge_frame.reset_index(inplace=True, drop=True)
            # print(f'merge frame before updating cumcount {merge_frame}')
            # print(merge_frame.shape[0]-1)
            merge_frame.at[merge_frame.shape[0] - 1, "count"] = cum_count
            # print(f'merge_frame after updating cumcount {merge_frame}')
            # if made_frame == 0:
            #     merge_frame = pd.DataFrame.from_dict({'Smiles':[smi],'count':[cum_count]})
            #     made_frame=1
            # else:
            #     merge_frame.loc[len(merge_frame)] = [smi, cum_count]
            rows_to_drop_one.append(i)
            rows_to_drop_two.append(j)
    # print(merge_frame)
    return {
        "drop_rows_1": rows_to_drop_one,
        "drop_rows_2": rows_to_drop_two,
        "merge_frame": merge_frame,
    }


def count_groups_in_set(
    list_of_inputs: list[str],
    drop_attachments: bool = False,
    input_type: str = "smile",
    bb_patt: str = "[$([C;X4;!R]):1]-[$([R,!$([C;X4]);!#0;!#9;!#17;!#35;!#1]):2]",
    cml_list=None,
    uni_smi_ty: bool = True,
    aiida: bool = False,
) -> pd.DataFrame:
    """Identify unique fragments in molecules defined in the list_of_smiles,
    and count the number of occurences for duplicates.

    Args:
        list_of_smiles: A list, with each element being a SMILES string, e.g. ['CC','C1CCCC1']
        drop_attachments: Boolean for whether or not to drop attachment points from fragments
            if True, will remove all placeholder atoms indicating connectivity
            if False, placeholder atoms will remain
        input_type: smile, xyzfile, cmlfile or molfile, based on elements of lists_of_inputs
        cml_list: defaults empty, but can be a list of cml files corresponding to the molfile inputs
        bb_patt: SMARTS pattern for bonds to break in linkers and side chains. Defaults to breaking
            bonds between nonring carbons with four bonds single bonded to ring atoms or carbons that
            don't have four bonds, and are not H, halide, or placeholder
        uni_smi_type: if True, include atom types in determination of unique
            fragments. If false, only determine unique by SMILES

    Returns:
        an output pd.DataFrame, with columns 'Smiles' for fragment Smiles,
        'count' for number of times each fragment occurs in the list, and
        'Molecule' holding a rdkit.Chem.Molecule object

    Example usage:
        >>> count_groups_in_set(['c1ccc(c(c1)c2ccc(o2)C(=O)N3C[C@H](C4(C3)CC[NH2+]CC4)C(=O)NCCOCCO)F',
        'Cc1nc2ccc(cc2s1)NC(=O)c3cc(ccc3N4CCCC4)S(=O)(=O)N5CCOCC5'],drop_attachments=False)

    """

    out_frame = pd.DataFrame()
    for i, inp in enumerate(list_of_inputs):
        print(inp)
        if cml_list:
            frame = identify_connected_fragments(
                inp,
                bb_patt=bb_patt,
                input_type=input_type,
                cml_file=cml_list[i],
                include_parent=True,
            )
        else:
            frame = identify_connected_fragments(
                inp, bb_patt=bb_patt, input_type=input_type, include_parent=True
            )
        if frame is not None:
            unique_frame = count_uniques(frame, drop_attachments, uni_smi_type=True)
            if out_frame.empty:
                out_frame = unique_frame
            else:
                out_frame = merge_uniques(out_frame, unique_frame, uni_smi_ty)
    out_frame = out_frame.drop("Molecule", axis=1)
    if not aiida:
        PandasTools.AddMoleculeColumnToFrame(
            out_frame, "Smiles", "Molecule", includeFingerprints=True
        )
    # out_frame = _add_xyz_coords(out_frame)
    # out_frame = _add_number_attachements(out_frame)
    return out_frame
