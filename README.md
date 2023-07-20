# Identifying fragments in molecule SMILES codes

Python functions to identify fragments in a molecule(or set of molecules) based on their SMILES codes, or .mol(and/or cml files). The fragments are chemically meaningful. Fragments identified include rings, linkers, side chains of molecules, and the functional groups(as defined by Ertl - heteroatoms and double bonds), and alkyl groups that compose them. There are two main functionalities currently, at the single molecule level, and at the batch level. At the single molecular level, the molecule is broken up into the fragments, and each fragment retains connectivity. At the batch level, for multiple molecules, connectivity is removed, and the unique fragments are identified, and their occurences counted. For example, this would identify that there are N methyl groups in the set of molecules.


The code is implemented primarily through RDKit, using a mix of the rdScaffoldNetwork module, and SMARTS pattern matching. rdScaffoldNetwork is used to identify the ring systems due to its flexibility in bond breaking rules, and that rdScaffoldNetwork will not break fused rings. The molecule is fragmented using the rdkit FragmentOnBonds functionality, which provides the option to label the dummy atoms produced with labels that can indicate connectivity. SMARTS matching is used to break the molecule further into functional groups and alkyl groups  by breaking single bonds between non-ring sp3 carbons and ring atoms or heteroatoms.


# Authors
Kevin Lefrancois-Gagnon
Robert C. Mawhinney

# Installation prior to distribution
```
pip install git+https://github.com/kmlefran/group_decomposition
```


# Usage Examples

## Identifying fragments in a single molecule

Passing any SMILES to identify_connected_fragments will return the identified fragments for that molecule in a pandas data frame. Fragments are included with connectivity information as dummy labels. That is, where the bonds were broken in the molecule to identify the fragment, there is a placeholder atom (*). This atom has a label in will appear in the smiles code as \[n\*\]  for integer n. The integer n will match with another fragment that will also have \[n\*\] in the smiles code. Each broken bond is assigned a different n, starting from 1, up to number broken.

```
identify_connected_fragments(smile='C1C(C)CCCC1')
```

The above output will include all fragments, even for example, multiple F atoms as \[1\*\]-F and \[2\*\]-F. One can remove connectivity information and count the number of unique fragments with the below code. fragFrame here is a frame returned by identify_connected_fragments. dropAttachments is a Boolean, defaulting to False. While False, placeholder atoms will remain in all fragments with more than one atom. This would, however make it so that similar fragments will not match if they have a difference in connectivity. (for example, ortho and para substituted aromatic rings would not match). If you would like such cases to match, set dropAttachments=True to do so.

The output of this below code is a similar data frame to identify_connected_fragments, but with a column 'count' for number of times each unique fragment occurs, and the SMILES lack connectivity information

```
count_uniques(fragFrame,dropAttachements)
```

## Identifying fragments in a set of molecules

If you have a set of molecules, and wish to identify unique fragments in the set, and total the number of times each fragment occurs, one can use the below code. dropAttachments is defined as above, and listOfSmiles is exactly as it sounds, a list with each element containing the SMILES of a molecule, e.g. ['CC', 'CCF']. The output is similar to count_uniques, but with rows for all fragments in a set of molecules, not just one.

```
count_groups_in_set(listOfSmiles,dropAttachments)
```