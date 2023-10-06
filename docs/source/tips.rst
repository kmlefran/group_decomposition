Tips
====

- Molecules are built from .cml files by building the molecule atom by atom from the list in atomArray sections, then adding in connectivity using information in bondArray. This only adds single bonds. Bond orders and formal atomic charges are then determined by mapping to the SMILEs in the .cml file

- If you simply want ring and not-ring systems broken apart, you can pass `bb_patt=''`

- If a molecule was unable to be constructed from a .cml file, the error is written to errorlog.txt
     - This could occur if the SMILEs in a .cml file does not match the xyz structure

- If you have a large set of molecules that you would like to fragment in batches in parallel, you can use :attr:`group_decomposition.fragfunctions.count_groups_in_set` on each batch, and later recombine with :attr:`group_decomposition.fragfunctions.merge_uniques`.

.. code-block:: python

    from group_decomposition import fragfunctions as ff
    list_list_inputs = [['input1','input2'...],['input3','input4'...]]
    # you could instead implement this next line with parallelization
    # the example here is just that. Adapt to suit your needs
    list_of_frames = [ff.count_groups_in_set(list_of_inputs = x) for x in list_list_files]
    flag = 0
    for frame in list_of_frames:
        if flag == 0:
            out_frame = frame
            flag = 1
        else:
            out_frame = ff.merge_uniques(out_frame,frame)
    # out_frame will have the combined results of the individual count_groups_in_set runs
    return out_frame
