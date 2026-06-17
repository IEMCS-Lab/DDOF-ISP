"""Matrix helpers for complex-valued tensors."""

import torch


def split_complex(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a complex tensor into real and imaginary parts."""
    if not z.is_complex():
        raise ValueError("Input must be a complex tensor")
    return torch.real(z), torch.imag(z)


def make_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    """Combine real and imaginary tensors into one complex tensor."""
    if real.shape != imag.shape:
        raise ValueError("Real and imag parts must have the same shape")
    return torch.complex(real, imag)


# Alias used by older scripts
concatenate_to_complex = make_complex


def complex_matmul(
    a_real: torch.Tensor,
    a_imag: torch.Tensor,
    b_real: torch.Tensor,
    b_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex matrix multiplication using real/imaginary blocks."""
    real = torch.matmul(a_real, b_real) - torch.matmul(a_imag, b_imag)
    imag = torch.matmul(a_real, b_imag) + torch.matmul(a_imag, b_real)
    return real, imag


# Alias
complex_matrix_multiply = complex_matmul


def to_diag(a: torch.Tensor) -> torch.Tensor:
    """Convert a vector-like tensor to a diagonal matrix."""
    if a.dim() == 1 or a.shape[0] == 1 or a.shape[1] == 1:
        return torch.diag(a.squeeze())
    return a


def from_diag(diag: torch.Tensor) -> torch.Tensor:
    """Extract the main diagonal from a square matrix."""
    if diag.dim() != 2 or diag.shape[0] != diag.shape[1]:
        raise ValueError("Input must be a square matrix")
    return torch.diag(diag)


# Aliases
Convert_to_Diag = to_diag
convert_from_diag = from_diag


def column_first_reshape(
    tensor: torch.Tensor, shape: tuple[int, int]
) -> torch.Tensor:
    """Reshape using MATLAB-style column-major order."""
    rows, cols = shape
    if tensor.numel() != rows * cols:
        raise ValueError(
            f"Element count mismatch: {tensor.numel()} != {rows}*{cols}"
        )
    return tensor.reshape(cols, rows).t()


# Alias used in ProcOpt/
ColumnFirstReshape = column_first_reshape


def normalize_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize magnitude to [0, 1] while preserving phase."""
    if not torch.is_complex(tensor):
        raise ValueError(f"Expected complex tensor, got {tensor.dtype}")

    mag = torch.abs(tensor)
    min_mag = torch.min(mag)
    max_mag = torch.max(mag)

    if torch.isclose(max_mag, min_mag):
        return torch.zeros_like(tensor)

    norm_mag = (mag - min_mag) / (max_mag - min_mag)

    phase = torch.where(
        mag > 0,
        tensor / mag,
        torch.zeros_like(tensor),
    )
    return norm_mag * phase


Normalize = normalize_complex


def mask_by_magnitude(
    tensor: torch.Tensor, threshold: float = 0.5, expected_shape: tuple = (64, 64)
) -> torch.Tensor:
    """Create a binary mask from complex magnitude."""
    if tensor.shape != expected_shape:
        raise ValueError(
            f"Expected shape {expected_shape}, got {tensor.shape}"
        )
    if not torch.is_complex(tensor):
        raise TypeError(f"Expected complex tensor, got {tensor.dtype}")

    return (torch.abs(tensor) > threshold).float()


Mask = mask_by_magnitude
