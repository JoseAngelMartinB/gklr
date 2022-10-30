"""GKLR package"""

from .logger import *
from .gklr import *

__author__ = """José Ángel Martín Baos"""

if sys.version_info[:2] >= (3, 8):
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

logger_debug("GKLR module initialized.")  

def display_info():
    """Display GKLR module information."""
    print("\n" + "-"*34 + " GKLR info " + "-"*35)
    print(__doc__)
    print("Version: " + __version__)
    print("Author: " + __author__)
    print("-"*80)