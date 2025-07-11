import torch
from torch import Tensor

__all__ = ["hash_pair_rows", "hash_pair_rows_sorted", "hash_pair_rows_simple"]


def hash_pair_rows(matrix: Tensor) -> Tensor:
    """
    Hash-based row pairing for efficient pair detection (CPU implementation).

    Args:
        matrix: Input tensor of shape (M, N) where M is number of rows,
               N is number of columns (sequence length)
    Returns:
        Tensor of shape (M/2, 2) where each row contains indices of paired rows
    """
    return torch.ops.extension_cpp.hash_pair_rows.default(matrix)


def hash_pair_rows_sorted(matrix: Tensor) -> Tensor:
    """
    Hash-based row pairing using CUDA with sorting (O(N log N)).

    Args:
        matrix: Input tensor of shape (M, N) where M is number of rows,
               N is number of columns (sequence length)
    Returns:
        Tensor of shape (M/2, 2) where each row contains indices of paired rows
    """
    return torch.ops.extension_cpp.hash_pair_rows_sorted.default(matrix)


def hash_pair_rows_simple(matrix: Tensor) -> Tensor:
    """
    Hash-based row pairing using CUDA without sorting (O(N) average case).

    Args:
        matrix: Input tensor of shape (M, N) where M is number of rows,
               N is number of columns (sequence length)
    Returns:
        Tensor of shape (M/2, 2) where each row contains indices of paired rows
    """
    return torch.ops.extension_cpp.hash_pair_rows_simple.default(matrix)


@torch.library.register_fake("extension_cpp::hash_pair_rows")
def _(matrix):
    torch._check(matrix.ndim == 2, "Input must be a 2D matrix")
    M, N = matrix.shape
    torch._check(
        M % 2 == 0, "Number of rows must be even for complete pairing"
    )
    torch._check(matrix.dtype == torch.long, "Matrix must be of integer type")

    # Return tensor of shape (M/2, 2) containing pair indices
    pairs = torch.empty((M // 2, 2), dtype=torch.long, device=matrix.device)
    return pairs


@torch.library.register_fake("extension_cpp::hash_pair_rows_sorted")
def _(matrix):
    torch._check(matrix.ndim == 2, "Input must be a 2D matrix")
    M, N = matrix.shape
    torch._check(
        M % 2 == 0, "Number of rows must be even for complete pairing"
    )
    torch._check(matrix.dtype == torch.long, "Matrix must be of integer type")

    # Return tensor of shape (M/2, 2) containing pair indices
    pairs = torch.empty((M // 2, 2), dtype=torch.long, device=matrix.device)
    return pairs


@torch.library.register_fake("extension_cpp::hash_pair_rows_simple")
def _(matrix):
    torch._check(matrix.ndim == 2, "Input must be a 2D matrix")
    M, N = matrix.shape
    torch._check(
        M % 2 == 0, "Number of rows must be even for complete pairing"
    )
    torch._check(matrix.dtype == torch.long, "Matrix must be of integer type")

    # Return tensor of shape (M/2, 2) containing pair indices
    pairs = torch.empty((M // 2, 2), dtype=torch.long, device=matrix.device)
    return pairs