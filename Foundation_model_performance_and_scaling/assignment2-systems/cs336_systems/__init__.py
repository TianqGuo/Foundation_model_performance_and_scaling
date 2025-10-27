import importlib.metadata

__version__ = importlib.metadata.version("cs336-systems")

from . import profiling_benchmarking

__all__ = ["profiling_benchmarking"]