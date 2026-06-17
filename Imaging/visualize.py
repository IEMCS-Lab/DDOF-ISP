"""Visualize DDOF reconstruction against ground truth."""

import os
import sys

import scipy.io
import torch

# Project path setup
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, "..")
sys.path.insert(0, _project_root)

from FuncLib.ImageFunc import visualize_six_panel
from FuncLib.MatrixFunc import split_complex

# I/O paths
DATA_DIR = os.path.join(_script_dir, "..", "Data")
SAVE_DIR = os.path.join(_script_dir, "..", "ImageSave", "Case")

DDOF_FILE = "Chi_DDOF.mat"
TARGET_SHAPE = (64, 64)

# Colormap ranges
VMIN_REAL, VMAX_REAL = 0.0, 2.0
VMIN_IMAG, VMAX_IMAG = -2.0, 0.0

if __name__ == "__main__":
    # Plot font
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print("-" * 36)

    def _load_mat(filename):
        """Load one .mat variable as a complex64 tensor."""
        path = os.path.join(DATA_DIR, filename)
        key = filename.replace(".mat", "")
        return torch.tensor(
            scipy.io.loadmat(path)[key], device=device
        ).to(torch.complex64)

    # Load inputs
    chi_gt   = _load_mat("Chi.mat")
    chi_init = _load_mat("Chi0.mat")

    # Load DDOF reconstruction
    ddof_path = os.path.join(DATA_DIR, DDOF_FILE)
    if not os.path.exists(ddof_path):
        raise FileNotFoundError(f"DDOF result not found: {ddof_path}")

    chi_ddof  = _load_mat(DDOF_FILE)
    chi_error = chi_ddof - chi_gt

    print("Data loading completed.")

    # Split real / imaginary parts
    gt_real,   gt_imag   = split_complex(chi_gt)
    ddof_real, ddof_imag = split_complex(chi_ddof)
    err_real,  err_imag  = split_complex(chi_error)

    # Generate comparison figure
    visualize_six_panel(
        gt_real, gt_imag,
        ddof_real, ddof_imag,
        err_real, err_imag,
        save_path=os.path.join(SAVE_DIR, "ImagingAll_six_panel.png"),
        target_shape=TARGET_SHAPE,
        vmin_real=VMIN_REAL,
        vmax_real=VMAX_REAL,
        vmin_imag=VMIN_IMAG,
        vmax_imag=VMAX_IMAG,
    )

    print("All done.")
