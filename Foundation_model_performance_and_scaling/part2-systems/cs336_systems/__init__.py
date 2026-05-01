import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336-systems")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

from . import profiling_benchmarking, attention_benchmarking, torch_compile_benchmarking

__all__ = ["profiling_benchmarking", "attention_benchmarking", "torch_compile_benchmarking"]