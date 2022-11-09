import warnings
from typing import Iterable, Union
from os import PathLike
from enum import IntEnum
from queue import LifoQueue
from .misc import safe_recursion_bootstrap

import numpy as np
import pandas as pd


class Type(IntEnum):
    SOMA = 1
    AXON = 2
    BASAL = 3
    APICAL = 4


def read(path: Union[str, 'PathLike[str]'], mode='simple') -> pd.DataFrame:
    """
    function for reading SWC file (neuron tree)

    :param path: string or path like object to an SWC file
    :param mode: "simple" for classic SWC files with 7 columns; "with_features" for extra columns storing preprocessing for each
    node
    :return: a `Pandas.DataFrame` that contains everything.
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
    if mode == 'simple':
        pass
    elif mode == 'with_features':
        names = ["n", 'type', 'x', 'y', 'z', 'r', 'parent', 'seg_id',
                 'node2soma_x', 'node2soma_y', 'node2soma_z',
                 'node2parent_x', 'node2parent_y', 'node2parent_z',
                 'node2branch_x', 'node2branch_y', 'node2branch_z',
                 'node2sibling_x', 'node2sibling_y', 'node2sibling_z',
                 'angle_pn', 'dis_pn',
                 'angle_cur_seg', 'dis_cur_seg',
                 'angle_cur_gravity', 'dis_cur_gravity',
                 'angle_sibling_seg', 'dis_sibling_seg',
                 'angle_sibling_gravity', 'dis_sibling_gravity',
                 'angle_parent_seg', 'dis_parent_seg',
                 'angle_parent_seg_gravity', 'dis_parent_seg_gravity']
        used_cols = list(range(len(names)))
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
    if df['parent'].iloc[0] != (-1):
        print('In func readSWC: ', path.split('\\')[-1], ' not sorted')
    return df


def sort(swc: pd.DataFrame):
    """
    Give the SWC a new set of id from 1, using dfs. Allowing I SWC to have only 1 root. The modification is inplace.
    Different from the Vaa3D plugin if you may notice, it doesn't use a LUT and node of same coordinates and radius
    won't be merged.

    :param swc: SWC to be sorted.
    """

    assert 'n' not in swc.columns
    root = swc.loc[swc['parent'] == -1].index
    assert len(root) == 1
    root = root[0]

    child_dict = get_child_dict(swc)
    ind = [0]
    swc['n'] = 0

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


def get_child_dict(swc: pd.DataFrame) -> dict[int, list]:
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
    """
    Calculate total path distance given an SWC. This SWC can be incomplete, i.e.
    it ignores nodes with parents not found.
    :param swc: SWC preprocessing frame.
    :param nodes: specify a set of node indices instead of using all.
    :return: sum of path distance.
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
    """
    Convert SWC to a new dataframe of segments. A segment is defined as a series connected points between critical nodes
    (branches & terminals), usually include the far node and exclude the near node.

    :param swc: SWC dataframe.
    :return: dataframe indexed by the far node of each segment, fields include nodes, parentSeg, childSeg
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
