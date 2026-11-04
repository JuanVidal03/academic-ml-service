"""Academic dropout prediction service."""

from importlib.metadata import PackageNotFoundError, version

try:
    #: Read from the installed package metadata, so pyproject.toml is the only place the
    #: version is written. Keeping a second literal here is the same duplication this
    #: codebase removes everywhere else.
    __version__ = version("dropout-service")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
