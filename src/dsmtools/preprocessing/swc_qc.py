from queue import SimpleQueue
import pandas as pd
import numpy as np
from itertools import chain

from dsmtools import utils


class SWCQualityControl:
    """A class used to do necessary quality control steps for our deep learning application.
    It ensures:

    - only one soma(parent=-1) in one SWC
    - no short tip branches(length < 10 by default)
    - no multifurcation node

    By providing a neuron tree in `Pandas.DataFrame` for class initialization,
    you can call each member function for quality control, and each time the functions will store their results.
    The tree used by this class is only a shallow reference, meaning after this control, the result is already in
    your I dataframe.

    Advised order of QC process:

    1. retain_only_1st_root
    2. adjust_multifurcation
    3. prune_by_len
    """

    def __init__(self, swc: pd.DataFrame):
        """
        Input an SWC that will only be referenced.

        :param swc: an SWC pandas dataframe.
        """

        self._swc = swc

    def retain_only_1st_root(self):
        """
        Our model assumes only one tree in an SWC. Here the tree of first root in SWC,
        if multiple roots exist, will be retained and any other nodes will be dropped.
        If your neuron structure is just broken for some reason, you can connect them
        using Vaa3D's sort_neuron_swc plugin or whatever.
        """

        roots = self._swc[self._swc.parent == -1]
        if len(roots) == 1:
            return self
        if len(roots) == 0:
            raise "No root in this SWC."
        sq = SimpleQueue()
        sq.put(roots.iloc[0, 0])
        visited = set()
        child_dict = utils.swc.get_child_dict(self._swc)
        while not sq.empty():
            head = sq.get()
            visited.add(head)
            for ch in child_dict[head]:
                if ch not in visited:
                    sq.put(ch)
        self._swc = self._swc.loc[list(visited)]
        self._swc.drop(index=list(visited), inplace=True)
        return self

    def prune_by_len(self, len_thr=10):
        """
        Too short terminal segments in a neuron tree is meaningless for our model and too many of
        them would heavily burden the computation.

        This function iteratively prune the tree. Every time it finds the short terminal branches and delete them,
        and new terminal branches emerge, until no short branches detected. This way, the tree is ensured to be with
        no terminal branches shorter than the threshold, and the main skeleton to the farthest reach is maintained.

        :param len_thr: the min length allowed for terminal branches, default as 10
        """

        while True:
            segments = utils.swc.get_segments(self._swc)
            drop_ind = []
            terminal_branch = np.unique([row['parentSeg'] for ind, row in segments.iterrows() if row['childSeg'] == []])
            for ind in terminal_branch:
                # retain non-terminal branches
                child_seg = [i for i in segments.at[ind, 'childSeg'] if segments.at[i, 'childSeg'] == []]
                # get length for all segments
                seg_len = [utils.swc.get_path_len(self._swc, segments.at[c, 'nodes']) for c in child_seg]
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

    def adjust_multifurcation(self):
        """Turn all multifurcations into multiple serial bifurcations except root nodes.

        Converting neuron tree to sequence requires all branches to be bifurcation as binary tree traversal requires.
        This function disintegrate multifurcations as bifurcations, by the order of the length of their child segments,
        i.e. the path distance from this multifurcation to the next branch point or terminal.
        """

        utils.swc.sort(self._swc)
        segments = utils.swc.get_segments(self._swc)
        new_ind = max(self._swc.index)
        for ind, row in segments.iterrows():
            # only for multifurcation and when it's not root
            # it attempts
            if len(row['childSeg']) > 2 and row['parentSeg'] != -1:
                # get path length of all child segments and sort
                len_list = [utils.swc.get_path_len(self._swc, segments.loc[i, 'nodes']) for i in row['childSeg']]
                # from short to long disintegrate child segments to make them bifurcation
                # short child seg would be brought up to a parent node first
                # the parent node share the coordinate with the multifurcation
                for i in np.argsort(len_list)[:-1]:
                    to_move = segments.loc[row['childSeg'][i], 'nodes'][-1]
                    multi = self._swc.loc[to_move, 'parent']
                    parent = self._swc.loc[multi, 'parent']
                    new_ind += 1
                    self._swc.loc[new_ind] = self._swc.loc[multi].copy(deep=True)       # new node
                    self._swc.loc[new_ind, 'parent'] = parent
                    self._swc.loc[multi, 'parent'] = new_ind
                    self._swc.loc[to_move, 'parent'] = new_ind
        utils.swc.sort(self._swc)
        return self
