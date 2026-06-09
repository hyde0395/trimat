class TrimatError(Exception):
    """Base exception for all trimat errors."""


class PackError(TrimatError):
    """Raised when packing or quantization input is invalid."""


class KernelError(TrimatError):
    """Raised when GEMV/GEMM arguments are invalid."""


class LoaderError(TrimatError):
    """Raised when HuggingFace model loading fails (used in P3)."""
