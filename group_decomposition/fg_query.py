"""
fg_query module

Functions and SMARTs strings for querying specific functional groups from results table of
:attr:`group_decomposition.fragfunctions.identify_connected_fragments`
:attr:`group_decomposition.fragfunctions.count_uniques`, or :attr:`group_decomposition.fragfunctions.count_groups_in_set`
"""
import pandas as pd  # lots of work with data frames
from rdkit import Chem  # pylint:disable=import-error

fg_dict = {
    "vinylic carbon": "[$([CX3]=[CX3])]",
    "allenic carbon": "[$([CX2](=C)=C)]",
    "alkyne": "[$([CX2]#C)]",
    "acyl halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "amidinium": "[NX3][CX3]=[NX3+]",
    "carbamic ester": "[NX3][CX3](=[OX1])[OX2H0]",
    "carbamic acid": "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]",
    "carboxylate": "[CX3](=O)[O-]",
    "carbonic acid/acid-ester": "[CX3](=[OX1])([OX2])[OX2H,OX1H0-1]",
    "carbonic ester": "C[OX2][CX3](=[OX1])[OX2]C",
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "cyanamide": "[NX3][CX2]#[NX1]",
    "ester or anhydride": "[#6][CX3](=O)[OX2H0][#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "ether": "[OD2]([#6])[#6]",
    "primary amine": "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
    "enamine": "[NX3][CX3]=[CX3]",
    "secondary amine": "[NX3;H1;!$(NC=O)]",
    "azide": "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]",
    "hydrazine": "[NX3][NX3]",
    "hydrazone": "[NX3][NX2]=[*]",
    "substituted imine": "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
    "iminium": "[NX3+]=[CX3]",
    "unsubstituted dicarboximide": "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
    "substituted dicarboximide": "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
    # 'nitrate': '[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]',
    "nitrile": "[NX1]#[CX2]",
    "isonitrile": "[CX1-]#[NX2+]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "nitroso": "[NX2]=[OX1]",
    "alcohol hydroxyl": "[OX2H][CX4]",
    "enol": "[OX2H][#6X3]=[#6]",
    "phenol": "[OX2H][cX3]:[c]",
    "peroxide": "[OX2,OX1-][OX2,OX1-]",
    "thioester": "S([#6])[CX3](=O)[#6]",
    "thiol": "[#16X2H]",
    "thioamide": "[NX3][CX3]=[SX1]",
    "thioketone": "[#6][CX3](=S)[#6]",
    "thioaldehyde": "[CX3H1](=S)[#6]",
    "monosulfide": "[#16X2H0][!#16]",
    "disulfide": "[#16X2H0][#16X2H0]",
    "sulfinic acid": "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
    "sulfonic acid": "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
    "sulfonamide": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
    "sulfone": "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]",
    "sulfoxide": "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]",
    "sulfate": "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
    "sulfuric acid diester": "[$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6]),$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6])]",  # pylint:disable=line-too-long
    "chlorine": "Cl",
    "fluorine": "F",
    "bromine": "Br",
}


def query_pattern(frag_frame: pd.DataFrame, patt: str) -> int:
    """Determine number of functional groups matching pattern in fragment table

    Args:
        frag_frame: fragment frame from output of :attr:`group_decomposition.fragfunctions.identify_connected_fragments`
            :attr:`group_decomposition.fragfunctions.count_uniques`,
            or :attr:`group_decomposition.fragfunctions.count_groups_in_set`
        patt: SMARTs string to match functional group. See :attr:`group_decomposition.fg_query.fg_dict` for SMARTS
            of common groups

    Returns:
        Number of matches of the functional group queried in the table

    Example Usage:
        >>> frag_frame
        Smiles  Molecule        count
        *C=CC=C <mol object>    3
        *C=C    <mol object>    1
        >>> query_pattern(frag_frame,'[$([CX3]=[CX3])]')
        7

    Note:
        If the column has a count of the number of times a fragment occurs (count column), multiply the number of matches
        in a group by the count of times it occurs

    """
    patt_mol = Chem.MolFromSmarts(patt)
    mol_list = list(frag_frame["Molecule"])
    if "count" in list(frag_frame.columns):
        count_list = list(frag_frame["count"])
        patt_count = [
            count_list[i] * len(mol.GetSubstructMatches(patt_mol))
            for i, mol in enumerate(mol_list)
        ]
    else:
        patt_count = [len(mol.GetSubstructMatches(patt_mol)) for mol in mol_list]
    return sum(patt_count)


def count_fgs(frag_frame: pd.DataFrame, patt_dict: dict) -> pd.DataFrame:
    """Given functional group dictionary with patterns to match, return counts of each in fragment_frame

    Args:
        frag_frame: fragment frame from output of :attr:`group_decomposition.fragfunctions.identify_connected_fragments`
            :attr:`group_decomposition.fragfunctions.count_uniques`,
            or :attr:`group_decomposition.fragfunctions.count_groups_in_set`
        patt_dict: has keys of functional group names and values being a SMARTs string to match

    Returns:
        Frame with one column containing names of functional group and another with count

    Example Usage:
        >>> frag_frame
        Smiles  Molecule        count
        *C=CC=C <mol object>    3
        *C=O    <mol object>    1
        >>> patt_dict
        {
        'aldehyde': '[CX3H1](=O)[#6]',
        'vinylic carbon': '[$([CX3]=[CX3])]'
        }
        >>> count_fgs(frag_frame,patt_dict)
        group           count
        aldehye         1
        vinylic carbon  6

    Note:
        If the column has a count of the number of times a fragment occurs (count column), multiply the number of matches
        in a group by the count of times it occurs

    """
    return pd.DataFrame(
        {
            "group": [list(patt_dict.keys())],
            "count": [query_pattern(frag_frame, value) for value in patt_dict.values()],
        }
    )
