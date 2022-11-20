import numpy as np
from typing import Sequence
from ._image import Image
import pandas as pd
import json
from importlib_resources import files, as_file


# import ccf_atlas resources
with as_file(files('dsmtools.utils.ccf_atlas') / 'annotation_25.nrrd') as path:
    annotation = Image(path)
with as_file(files('dsmtools.utils.ccf_atlas') / 'bs_level.csv') as path:
    bs_level = pd.read_csv(path, sep=',', index_col=0)
with as_file(files('dsmtools.utils.ccf_atlas') / 'dict_to_selected.json') as path:
    with open(path) as f:
        dict_to_selected = dict((int(k), v) for k, v in json.loads(f.read()).items())


def id2abbr(region_id: int) -> str:
    """Convert a CCFv3 region ID to its region name.

    :param region_id: CCFv3 structure ID.
    :return: The abbreviation of the structure.
    """
    return bs_level.at[region_id, 'Abbreviation']


def coord2abbr(point: Sequence[float]) -> str:
    """Given the 3D coordinate of a point, retrieve the abbreviation of the region containing the point.
    It will locate the finest region for the location within the 25uμm template.

    :param point: A 3D coordinate by xyz.
    :return: The abbreviation of its brain region.
    """
    p = np.array(point, dtype=float)
    assert p.shape == (3,)
    p /= (annotation.space['x'], annotation.space['y'], annotation.space['z'])
    p = p.round().astype(int)
    if 0 <= p[0] < annotation.size['x'] and 0 <= p[1] < annotation.size['y'] and 0 <= p[2] < annotation.size['z']:
        region_id = annotation.array[p[0], p[1], p[2]]
        if region_id in dict_to_selected:
            return id2abbr(dict_to_selected[region_id])
    return 'unknown'


def annotate_swc(swc: pd.DataFrame) -> pd.Series:
    """Given an SWC dataframe (with columns indicating coordinates, named by x, y, z),
    return a series of regions mapped by node locations.

    :param swc: SWC as a pandas dataframe.
    :return: A column of region abbreviation as pandas series.
    """
    return swc.apply(lambda r: coord2abbr(r[['x', 'y', 'z']].values), axis=1)
