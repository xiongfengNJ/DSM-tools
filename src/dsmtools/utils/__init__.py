"""
This module contains generic utilities used by the data processing workflows served by its submodules. They are:

* `swc`: Functions related to basic SWC dataframes manipulations.
* `ccf_atlas`: CCFv3 mouse brain atlas support.
* `misc`: All other ungrouped utilities.

Proceed into each submodule for detailed descriptions of usage.
"""

from . import swc
from . import ccf_atlas
from . import misc

__all__ = ['swc', 'ccf_atlas', 'misc']
