import pytest
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd

from dsmtools.utils import swc, misc
from dsmtools.preprocessing import SWCQualityControl


raw_swc_paths = [*Path('data/swc').rglob('*.swc')]


@pytest.mark.parametrize("swc_path", raw_swc_paths)
def test_swc_prune(swc_path, tmp_path):
    """Compare the index of pruning results by vaa3d and py prune.

    It verifies the function of get_path_len, SWCQualityControl.prune_by_len, utils.swc.read
    """

    v3d_out = tmp_path / 'out.swc'
    min_seg_len = 10
    os.system(f"vaa3d_msvc.exe /x pruning_swc_simple /f pruning_iterative /i {swc_path} /o {v3d_out} /p {min_seg_len}")
    raw = swc.read(swc_path)
    expected = swc.read(v3d_out)
    q = SWCQualityControl(raw)
    q.prune_by_len(min_seg_len)
    assert set(raw.index) == set(expected.index)


@pytest.mark.parametrize("swc_path", raw_swc_paths)
def test_swc_sort(swc_path, tmp_path, worker_id):
    """
    Make sort let parent id always smaller than children.

    It verifies utils.swc.sort, utils.swc.get_child_dict
    """

    raw = swc.read(swc_path)
    swc.sort(raw)
    assert np.all(raw.index > raw['parent'])


# @pytest.fixture(params=[1, 2, 3 ,4, 5])
# def multi_root_swc(request):
#     """Generate SWC dataframe with multiple root nodes"""
#     out = pd.DataFrame(columns=['n', 'type', 'x', 'y', 'z', 'r', 'parent'])
#     nid = 0
#     for i in range(request.param):
#         node_set = set()
#         component_size = random.randint(10, 100)
#         for j in range(component_size):
#             nid += 1
#             out
#     out.set_index('n', inplace=True)
#     return out
#
#
# def test_retain_only_1st_root(multi_root_swc):
