from .image import Image
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


def id_to_name(region_id):
    # region_name can be either Abbreviation (checked first) or description
    if region_id in bs_level.index:
        return bs_level.loc[region_id, 'Abbreviation']
    else:
        print("Cannot find any regions with ID %s." % region_id)


def get_node_region(point) -> str:
    p = point[['x', 'y', 'z']].copy()
    p['x'] = p['x'] / annotation.space['x']
    p['y'] = p['y'] / annotation.space['y']
    p['z'] = p['z'] / annotation.space['z']
    p = p.round(0).astype(int)
    if ((p.x.iloc[0] >= 0) & (p.x.iloc[0] < annotation.size['x']) &
            (p.y.iloc[0] >= 0) & (p.y.iloc[0] < annotation.size['y']) &
            (p.z.iloc[0] >= 0) & (p.z.iloc[0] < annotation.size['z'])
    ):
        region_id = annotation.array[p.x.iloc[0],
                                     p.y.iloc[0],
                                     p.z.iloc[0]]
        if region_id in dict_to_selected:
            return id_to_name(dict_to_selected[region_id])
    return 'unknown'


def get_region(swc: pd.DataFrame):
    return swc.apply(lambda r: get_node_region(pd.DataFrame({'x': [r.x], 'y': [r.y], 'z': [r.z]})), axis=1)
