"""
This module imported from [neuro-morpho-toolbox](https://github.com/pengxie-bioinfo/neuro_morpho_toolbox) (by Peng Xie)
and only includes the necessary functions for loading CCFv3 mouse brain atlas. It can be used to retrieve brain region
types of a certain location in the mouse brain template space.

It's used by importing this module, and only CCFv3 related functions are exposed. You can find them in this page.

For more information on CCFv3, you can:

* check this paper: [Wang Q, Ding SL, Li Y, et al. The Allen Mouse Brain Common Coordinate Framework:
A 3D Reference Atlas. Cell. 2020;181(4):936-953.e20. doi:10.1016/j.cell.2020.04.007](https://pubmed.ncbi.nlm.nih.gov/32386544/)
* download the [table of all annotations](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8152789/bin/NIHMS1585864-supplement-10.xlsx)
* explore the mouse brain atlas at http://connectivity.brain-map.org.
"""

from ._annotation import id2abbr, coord2abbr, annotate_swc


__all__ = ['id2abbr', 'coord2abbr', 'annotate_swc']