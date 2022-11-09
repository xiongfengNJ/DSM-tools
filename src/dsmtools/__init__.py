from importlib.metadata import version, PackageNotFoundError
from . import modeling
from . import preprocessing
from . import utils

try:
    __version__ = version("DSM-tools")
except PackageNotFoundError:
    __version__ = "UNKNOWN"
