"""Visualization helpers for susceptibility reconstructions."""

import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


def create_custom_turbo(
    low_range_ratio: float = 0.02,
    low_color_ratio: float = 0.05,
) -> mcolors.LinearSegmentedColormap:
    """Create a Turbo colormap with more contrast near low values."""
    turbo = cm.get_cmap("turbo", 256)
    palette = turbo(np.linspace(0, 1, 256))

    n_low = int(256 * low_color_ratio)
    n_high = int(256 * (1 - low_color_ratio))

    indices = np.concatenate([
        np.linspace(0, int(256 * low_range_ratio), n_low),
        np.linspace(int(256 * low_range_ratio), 255, n_high),
    ]).astype(int)

    return mcolors.LinearSegmentedColormap.from_list(
        "custom_turbo", palette[indices]
    )


create_cmap = create_custom_turbo


def _to_2d(tensor: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    """Convert a tensor to a 2D NumPy array using column-major order."""
    t = tensor.cpu().detach()
    rows, cols = shape
    if t.dim() > 1:
        t = t.reshape(-1)
    if t.numel() != rows * cols:
        raise ValueError(
            f"Element count mismatch: {t.numel()} != {rows}*{cols}"
        )
    t = t.reshape(cols, rows).t()
    return t.numpy()


def visualize_six_panel(
    gt_real: torch.Tensor,
    gt_imag: torch.Tensor,
    ddof_real: torch.Tensor,
    ddof_imag: torch.Tensor,
    err_real: torch.Tensor,
    err_imag: torch.Tensor,
    save_path: str,
    target_shape: tuple[int, int] = (64, 64),
    vmin_real: float = 0.0,
    vmax_real: float = 2.0,
    vmin_imag: float = -2.0,
    vmax_imag: float = 0.0,
    dpi: int = 600,
) -> None:
    """Draw ground truth, reconstruction, and error in a 2 x 3 panel."""
    # Data arranged by panel position
    data = {
        (0, 0): _to_2d(gt_real,   target_shape),
        (0, 1): _to_2d(ddof_real, target_shape),
        (0, 2): _to_2d(err_real,  target_shape),
        (1, 0): _to_2d(gt_imag,   target_shape),
        (1, 1): _to_2d(ddof_imag, target_shape),
        (1, 2): _to_2d(err_imag,  target_shape),
    }

    gt_real_2d = _to_2d(gt_real, target_shape)
    gt_imag_2d = _to_2d(gt_imag, target_shape)
    refs = [gt_real_2d, gt_imag_2d]

    col_titles = ["Ground Truth", "Reconstruction", "Error"]
    row_labels = ["Real Part", "Imaginary Part"]
    vmins  = [vmin_real, vmin_imag]
    vmaxes = [vmax_real, vmax_imag]

    cmap = create_cmap(low_range_ratio=0.02, low_color_ratio=0.05)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    plt.subplots_adjust(right=0.85, wspace=0.04, hspace=0.08)

    for row in range(2):
        for col in range(3):
            ax = axes[row][col]
            arr = data[(row, col)]
            ax.imshow(
                arr, cmap=cmap,
                vmin=vmins[row], vmax=vmaxes[row],
                origin="lower",
            )

            # Overlay a ground-truth contour for reference
            ref_arr = refs[row]
            ref_val = np.mean(ref_arr) * 0.2
            if abs(ref_val) > 1e-12:
                ax.contour(
                    ref_arr, levels=[ref_val],
                    colors="white", linewidths=1.2,
                    linestyles="--", origin="lower",
                )

            ax.axis("off")

            if row == 0:
                ax.set_title(col_titles[col], fontsize=14, fontweight="bold")

        # One colorbar per row
        last_ax = axes[row][-1]
        bbox = last_ax.get_position()
        cax = fig.add_axes([
            bbox.x1 + 0.02,
            bbox.y0,
            0.015,
            bbox.y1 - bbox.y0,
        ])
        sm = plt.cm.ScalarMappable(
            norm=plt.Normalize(vmins[row], vmaxes[row]), cmap=cmap,
        )
        cbar = fig.colorbar(sm, cax=cax)
        ticks = [vmins[row], (vmins[row] + vmaxes[row]) / 2, vmaxes[row]]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.1f}" for t in ticks], fontsize=12,
                             weight="bold")
        cbar.set_label(row_labels[row], fontsize=14, fontweight="bold",
                       rotation=270, labelpad=18)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"Image saved to {save_path}")
    plt.show()
