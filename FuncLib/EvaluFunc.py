"""Evaluation metrics for complex-valued tensors."""

import torch


def compute_relative_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "none",
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute relative error with optional reduction."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        )

    abs_error = torch.abs(pred - target)
    denominator = torch.abs(target) + epsilon
    rel_error = abs_error / denominator

    if reduction == "mean":
        return torch.mean(rel_error)
    if reduction == "sum":
        return torch.sum(rel_error)
    if reduction == "none":
        return rel_error

    raise ValueError(
        f"Unsupported reduction '{reduction}'; choose from: none, mean, sum"
    )


# Canonical alias used throughout the project
Evaluate_Relative_Error = compute_relative_error
