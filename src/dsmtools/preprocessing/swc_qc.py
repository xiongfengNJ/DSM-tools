import pandas as pd
import numpy as np
from itertools import chain

from dsmtools.utils.swc import get_path_len, get_subtree_nodes, get_child_dict, sort_swc, get_segments


class SWCQualityControl:
    """Interface for executing SWC quality controls. See [the above explanation](#reasons-for-qc) for its principle.

    To use this class, initialize it with an SWC dataframe (used as reference, so all the changes are inplace), and
    call each member function in a specific order (the recommended order is implemented by that of
    `NeuronSequenceDataset.qc_proc`).
    """

    def __init__(self, swc: pd.DataFrame):
        """Set the referenced SWC dataframe to modify.

        :param swc: An SWC dataframe, all changes are in place.
        """

        self._swc = swc

    def retain_only_1st_root(self):
        """
        Find the first root(parent=-1) in the SWC dataframe and remove all the other nodes outside this component.
        """

        roots = self._swc[self._swc.parent == -1]
        if len(roots) == 1:
            return self
        if len(roots) == 0:
            raise RuntimeError("No root in this SWC.")
        visited = get_subtree_nodes(self._swc, roots.iloc[0, 0])
        self._swc.drop(index=self._swc.index.difference(visited), inplace=True)
        return self

    def degrade_to_bifurcation(self):
        """Turn all branch nodes with more than 2 children into multiple serial bifurcations except for the root node.

        It commits a node ID sorting at first to find the start number for new nodes (as new bifurcations can be
        generated). Then it iterates through the unqualified nodes and turn them into bifurcations.

        The new bifurcations from the original branch node are arranged in a way that subtrees of lesser total path
        length are nearer to the root.
        """

        sort_swc(self._swc)
        swc = self._swc.copy(deep=True)
        new_ind = max(swc.index)
        child_dict = get_child_dict(swc)
        for node, children in child_dict.items():
            if len(children) <= 2 or swc.at[node, 'parent'] == -1:
                continue
            # get path length of all subtrees and sort
            len_list = [get_path_len(swc, get_subtree_nodes(swc, c)) for c in children]
            # from short to long disintegrate child segments to make them bifurcation
            # short child seg would be brought up to a parent node first
            # the parent node share the coordinate with the original branch node
            for i in np.argsort(len_list)[:-1]:
                to_move = children[i]
                multi = self._swc.at[to_move, 'parent']
                parent = self._swc.at[multi, 'parent']
                new_ind += 1
                self._swc.loc[new_ind] = self._swc.loc[multi].copy(deep=True)       # new node
                self._swc.at[new_ind, 'parent'] = parent
                self._swc.at[multi, 'parent'] = new_ind
                self._swc.at[to_move, 'parent'] = new_ind
        sort_swc(self._swc)
        return self

    def prune_by_len(self, len_thr=10):
        """
        This function iteratively prune the tree. Every time it finds the short terminal branches and delete them,
        and new terminal branches emerge, until no short branches detected. This way, the tree is ensured to be with
        no terminal branches shorter than the threshold, but the main skeleton to the farthest reach is maintained.

        :param len_thr: The min length allowed for terminal branches, default as 10
        """

        while True:
            segments = get_segments(self._swc)
            drop_ind = []
            terminal_branch = np.unique([row['parentSeg'] for ind, row in segments.iterrows() if row['childSeg'] == []])
            for ind in terminal_branch:
                # retain non-terminal branches
                child_seg = [i for i in segments.at[ind, 'childSeg'] if segments.at[i, 'childSeg'] == []]
                # get length for all segments
                seg_len = [get_path_len(self._swc, segments.at[c, 'nodes']) for c in child_seg]
                # all are terminal branches, then the longest branch must be retained
                if child_seg == segments.at[ind, 'childSeg']:
                    k = np.argmax(seg_len)
                    child_seg.pop(k)
                    seg_len.pop(k)
                # drop short branches
                drop_ind.extend(chain.from_iterable(segments.at[c, 'nodes']
                                                    for c, s in zip(child_seg, seg_len) if s <= len_thr))
            if not drop_ind:
                break
            self._swc.drop(index=drop_ind, inplace=True)
        return self
