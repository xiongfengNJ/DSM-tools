"""This module contains basic handlers for SWC as pandas dataframes.

## What is SWC

An SWC is a text file containing a space-delimited table that holds all the geometric data required for
reconstructing a neuron morphology. Usually with 7 columns, illustrated by:

```
# n type x y z   r parent
1   1    0 0 0   1 -1
2   1    2 0 0   1  1
3   2   -3 0 0 0.7  1
4   3   20 0 0   1  2
```

* The 1st column is node ID, must be a positive integer.
* The 2nd column is node type: 1 for soma, 2 for axon, 3 for basal dendrite, 4 for apical dendrite.
* The 3rd-5th columns are the 3D coordinates of the node in space.
* The 6th column is the node radius.
* The 7th column is the parent node ID, corresponding to some record in the first column. As all node IDs are positive,
-1 is used for no parent, and the node without a parent is a root.

SWC Files allow you to add comments started in each line with '#', including the header. The naming of the columns
in the file, as such, does not really matter. The dataframe column names used in this package
are consistent with the previous example.

## Pandas Dataframe

The [pandas](https://pandas.pydata.org/) package is perfect for handling table data like what SWC files typically hold.
The retrieval of nodes as
records is based on the hash table index of node IDs, and nodes can also be easily filtered with logical operations.

However, as objects in Python are usually passed by reference and pandas indexing can introduce shallow copy of
dataframes, you should make yourself familiar with the usage of the pandas package and avoid the problems for yourself.
Functions in this module have already been implemented in a way that is minimally bug-causing.
"""
import warnings
from typing import Iterable, Union, Sequence
from os import PathLike
from enum import IntEnum
from queue import LifoQueue, SimpleQueue
from .misc import safe_recursion_bootstrap

import numpy as np
import pandas as pd


__all__ = ['read_swc', 'sort_swc', 'get_child_dict', 'get_path_len', 'get_segments', 'get_subtree_nodes']


class Type(IntEnum):
    """
    An IntEnum class for clear type identification in the code. Numbering is consistent with the general SWC protocol.
    """
    SOMA = 1
    AXON = 2
    BASAL = 3
    APICAL = 4


def read_swc(path: Union[str, 'PathLike[str]'], more_features: Sequence[str] = tuple()) -> pd.DataFrame:
    """A wrapper reading function for SWC based on `pandas.read_csv`.

    This wrapper does the following job more than just load the 7-column table:

    * If your table is delimited by comma, it's still readable with this wrapper.
    * Specification of `more_features` with more column names to load more features on the nodes.

    :param path: A string or path like object to an SWC file.
    :param more_features: A sequence of column names with proper order, default as empty.
    :return: A `pandas.DataFrame`.
    """
    n_skip = 0
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("#"):
                n_skip += 1
            else:
                break
    names = ["n", "type", "x", "y", "z", "r", "parent"]
    used_cols = [0, 1, 2, 3, 4, 5, 6]
    if more_features:
        names.extend(more_features)
        used_cols = list(range(len(used_cols) + len(more_features)))
    try:
        df = pd.read_csv(path, index_col=0, skiprows=n_skip, sep=" ",
                         usecols=used_cols,
                         names=names
                         )
    except:
        df = pd.read_csv(path, index_col=0, skiprows=n_skip, sep=",",
                         usecols=used_cols,
                         names=names
                         )
    if df['parent'].iloc[0] != -1:
        print('In func readSWC: ', path.split('\\')[-1], ' not sorted')
    return df


def sort_swc(swc: pd.DataFrame) -> None:
    """Use recursion to give the SWC a new set of id starting from 1, and make sure the ID of parents are smaller than
    their children.

    The SWC has to contain only 1 root (parent as -1), and the modification is inplace (doesn't return a new dataframe).
    As you may ask, it is different from the Vaa3D *sort_neuron_swc* plugin, it doesn't use a LUT and node with the same
    coordinates and radius won't be merged, which is why only a single root is allowed.

    The recursion used by this piece of code is safe from overflow. You don't need to reset the recursion limit. Yet, it
    doesn't mean it wouldn't fall into an endless loop. Try not to input a ring-like structure.

    :param swc: An SWC dataframe to be sorted.
    """

    assert 'n' not in swc.columns
    root = swc.loc[swc['parent'] == -1].index
    assert len(root) == 1
    root = root[0]

    child_dict = get_child_dict(swc)
    ind = [0]
    swc['n'] = -1

    @safe_recursion_bootstrap
    def dfs(node: int, stack: LifoQueue):
        ind[0] += 1
        swc.loc[node, 'n'] = ind[0]
        for i in child_dict[node]:
            yield dfs(i, stack=stack)
        yield

    dfs(root, stack=LifoQueue())
    par_to_change = swc.loc[swc['parent'] != -1, 'parent']
    swc.loc[par_to_change.index, 'parent'] = swc.loc[par_to_change, 'n'].tolist()
    swc.set_index('n', inplace=True)
    swc.sort_index(inplace=True)


def get_child_dict(swc: pd.DataFrame) -> dict[int, list[int]]:
    """Get a Python dict object that maps from the SWC node IDs to their children.

    As for the SWC dataframe, you can find the parent for a node by index a hash table, but it's not so straightforward
    for locating their children. Therefore, a one-to-many mapping is needed. This function iterates through the SWC
    to document the children for each node as a list, and assemble them as a dict, so that you can find a node's
    children with constant time as well.

    Nodes without any children will be given an empty list.

    :param swc: An SWC dataframe, no modification takes place.
    :return:A Python dict mapping from node ID as int to lists of children node IDs.
    """
    child_dict = dict([(i, []) for i in swc.index])
    for ind, row in swc.iterrows():
        p_idx = int(row['parent'])
        if p_idx in child_dict:
            child_dict[p_idx].append(ind)
        else:
            if p_idx != -1:
                warnings.warn(f"Node index {p_idx} doesn't exist in SWC, but referenced as parent.")
    return child_dict


def get_path_len(swc: pd.DataFrame, nodes: Iterable[int] = None) -> float:
    """Calculate the total path length within an SWC, given a subset of node IDs.

    The path length is defined as the sum of the cartesian distance between pairs of connected nodes, which
    can be a good indicator for the size of a neuron structure.

    This function also offers an argument to select only part of the neuron tree to calculate the path length. The nodes
    specified are seen as children nodes.

    This SWC can also be one without a parent. The aggregation overlooks nodes without parent found.

    :param swc: An SWC dataframe, no modification takes place.
    :param nodes: A set of node indices as children among the connections.
    :return: The sum of path length.
    """
    if nodes is None:
        down_nodes = swc.index[swc['parent'].isin(swc.index)]
    else:
        par = swc.loc[pd.Index(nodes).intersection(swc.index), 'parent']        # input nodes must be in index
        down_nodes = par.index[par.isin(swc.index)]      # their parents must be in index
    up_nodes = swc.loc[down_nodes, 'parent']
    return np.linalg.norm(swc.loc[down_nodes, ['x', 'y', 'z']].values -
                          swc.loc[up_nodes, ['x', 'y', 'z']].values, axis=1).sum()


def get_segments(swc: pd.DataFrame) -> pd.DataFrame:
    """Convert SWC to a new dataframe of segments.

    A segment is defined as a series connected points between critical nodes (root, branches & terminals),
    usually start from the first node out of the branching point and ends at the next branching point or terminal.

    The conversion can manifest the topology of a neuron tree and convenience some operations at segment levels, such
    as pruning.

    Utilizing pandas powerful expressiveness, this function will set up a new dataframe object formatted like:

    | tail(index) |       nodes       | parentSeg |  childSeg   |
    |-------------|-------------------|-----------|-------------|
    |      1      |        [1]        |     -1    | [2, 20, 90] |
    |     ...     |        ...        |    ...    |     ...     |
    |      26     | [26, 25, ..., 18] |     17    |  [27, 50]   |
    |     ...     |        ...        |    ...    |     ...     |

    The index is set as the end point of each segment, which are unique. The *nodes* field contains the list of nodes
    for the segment from the end to the start. The *parentSeg* field is similar to the parent column of and SWC.
    Also, the *childSeg* field functions like what `get_child_dict` would do for an SWC.

    Root nodes are seen as a single segment, with *parentSeg* as -1.

    :param swc: An SWC dataframe, no modification takes place.
    :return: A new dataframe similar to SWC, seeing segments as nodes.
    """
    children_counts = swc['parent'].value_counts()
    branch = children_counts.index[children_counts.values > 1]
    terminal = swc.index.difference(swc.loc[filter(lambda x: x != -1, swc['parent'])].index)

    # assign each node with a segment id
    # seg: from soam_far 2 soam_near or from terminal2(branch-1) or branch2(branch2branch-1)
    # (down->up)
    segments = pd.DataFrame(columns=['tail', 'nodes', 'parentSeg', 'childSeg'])
    segments.set_index('tail', inplace=True)
    temp_index = set(swc.index)
    for ind in terminal:
        seg = []
        while True:
            seg.append(ind)
            temp_index.remove(ind)
            ind = swc.loc[ind, 'parent']
            if ind in branch or ind == -1:      # soma will be a single node segment if it's a branch point
                segments.loc[seg[0]] = [seg, ind, []]
                seg = []
            if ind not in temp_index:
                break

    for ind in segments.index.intersection(branch):
        child_seg_id = segments.index[segments['parentSeg'] == ind].tolist()
        segments.at[ind, 'childSeg'] = child_seg_id
        segments.loc[child_seg_id, 'parentSeg'] = ind

    return segments


def get_subtree_nodes(swc: pd.DataFrame, n_start: int) -> list[int]:
    """Retrieve IDs of all nodes under a specified node in an SWC by BFS.

    **Note**: there should be no ring in the SWC, or it will raise an error.

    :param swc: An SWC dataframe, no modification takes place.
    :param n_start: the node ID of the subtree root to start with.
    :return: A list of node IDs.
    """
    sq = SimpleQueue()
    sq.put(n_start)
    visited = set()
    child_dict = get_child_dict(swc)
    while not sq.empty():
        head = sq.get()
        visited.add(head)
        for ch in child_dict[head]:
            if ch in visited:
                raise RuntimeError("Detected a ring in the SWC, couldn't a get subtree.")
            else:
                sq.put(ch)
    return list(visited)
