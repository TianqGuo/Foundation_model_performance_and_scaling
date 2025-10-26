import importlib.metadata

__version__ = importlib.metadata.version("cs336-systems")

from . import benchmark

__all__ = ["benchmark"]