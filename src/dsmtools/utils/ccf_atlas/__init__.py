"""
This module imported from [neuro-morpho-toolbox](https://github.com/pengxie-bioinfo/neuro_morpho_toolbox) (by Peng Xie)
and only includes the necessary functions for loading CCFv3 mouse brain atlas. It can be used to retrieve brain region
types of a certain location in the mouse brain template space.

It's used by importing this module, and only CCFv3 related functions are exposed, such as:

* `id_to_name`: Convert a CCFv3 region ID to its region name.
* `get_node_region`: Given a coordinate, return the region name.
* `annotate_swc`: Given an SWC (with a series of coordinates), add a new column with regions mapped by node locations.
"""

from ._annotation import id_to_name, get_node_region, annotate_swc
