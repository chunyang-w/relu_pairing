import torch
from torch import Tensor

__all__ = ["hash_pair_rows", "mymuladd", "myadd_out"]


def hash_pair_rows(matrix: Tensor) -> Tensor:
    """
    Hash-based row pairing for efficient pair detection.

    Args:
        matrix: Input tensor of shape (M, N) where M is number of rows,
               N is number of columns (sequence length)
    Returns:
        Tensor of shape (M/2, 2) where each row contains indices of paired rows
    """
    return torch.ops.extension_cpp.hash_pair_rows.default(matrix)


@torch.library.register_fake("extension_cpp::hash_pair_rows")
def _(matrix):
    torch._check(matrix.ndim == 2, "Input must be a 2D matrix")
    M, N = matrix.shape
    torch._check(M % 2 == 0, "Number of rows must be even for complete pairing")
    torch._check(matrix.dtype == torch.long, "Matrix must be of integer type")

    # Return tensor of shape (M/2, 2) containing pair indices
    pairs = torch.empty((M // 2, 2), dtype=torch.long, device=matrix.device)
    return pairs


def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.extension_cpp.mymuladd.default(a, b, c)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("extension_cpp::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.extension_cpp.mymul.default(grad, b)
    if ctx.needs_input_grad[1]:
        grad_b = torch.ops.extension_cpp.mymul.default(grad, a)
    return grad_a, grad_b, None


def _setup_context(ctx, inputs, output):
    a, b, c = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "extension_cpp::mymuladd", _backward, setup_context=_setup_context
)


@torch.library.register_fake("extension_cpp::mymul")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Writes a + b into out"""
    torch.ops.extension_cpp.myadd_out.default(a, b, out)
