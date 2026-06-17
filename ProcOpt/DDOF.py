"""A Differentiable Discrete Optimization Framework (DDOF) for 2D Electromagnetic Inverse Scattering"""

import os
import sys

import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim

# Project path setup
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, "..")
sys.path.insert(0, _project_root)

from FuncLib.MatrixFunc import ColumnFirstReshape, split_complex
from FuncLib.EvaluFunc import Evaluate_Relative_Error

# I/O paths
DATA_DIR = os.path.join(_script_dir, "..", "Data")
OUTPUT_FILE = "Chi_DDOF.mat"

# L-BFGS optimizer settings
LBFGS_MAX_ITER = 100               # Maximum inner iterations per optimizer step
LBFGS_HISTORY_SIZE = 500           # Limited-memory history size
LBFGS_TOL_GRAD = 1e-15             # Gradient tolerance
LBFGS_TOL_CHANGE = 1e-15           # Parameter-change tolerance

# PnP regularization settings
LAMBDA_TV = 1e-9                    # TV prior strength
LAMBDA_L1 = 5e-10                   # L1 prior strength
TV_EPS = 1e-12                      # TV stability term

# Training / display settings
PRINT_INTERVAL = 5                  # Log every N inner iterations
NUM_EPOCHS = 2                      # Number of L-BFGS epochs

# Early-stopping settings
ENABLE_EARLY_STOP = True
MRE_SF_CONVERGE_THRESHOLD = 2.0     # MRE-SF convergence threshold (%)
MRE_SF_CONVERGE_COUNT = 5           # Consecutive hits before stopping

# Problem dimensions
DIM_64 = 64                         # Spatial grid size
DIM_4096 = 4096                     # 64 x 64 grid points
NUM_SOURCES = 32                    # Number of sources

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
    data = scipy.io.loadmat(path)
    key = filename.replace(".mat", "")
    return torch.tensor(data[key], device=device).to(torch.complex64)


# Load data
chi_gt = _load_mat("Chi.mat")          # Ground-truth susceptibility
chi_init = _load_mat("Chi0.mat")        # Initial guess for Chi
e_inc = _load_mat("Einc.mat")          # Incident electric field
e_sca = _load_mat("Esca.mat")          # Scattered electric field (measurement)
gd_mat = _load_mat("GD.mat")           # Domain Green's function operator
gs_mat = _load_mat("GS.mat")           # Source Green's function operator

e_tot = _load_mat("Etot.mat")          # Total electric field (reference)

# Reshape Green's functions from MATLAB column-major storage
gd_mat = ColumnFirstReshape(gd_mat, (DIM_4096, DIM_4096))
gs_mat = ColumnFirstReshape(gs_mat, (NUM_SOURCES, DIM_4096))

print("Data loading & preprocessing completed.")

# Reused by the forward solver
identity = torch.eye(DIM_4096, dtype=torch.complex64, device=device)
chi_init_real, chi_init_imag = split_complex(chi_init)


class DDOFModel(nn.Module):
    """Differentiable forward model for predicting scattered fields."""

    def __init__(self, init_real, init_imag):
        super().__init__()
        self.param_real = nn.Parameter(init_real)
        self.param_imag = nn.Parameter(init_imag)

    def forward(self, e_inc, gd, gs):
        """Run the forward scattering model."""
        chi = torch.complex(self.param_real, self.param_imag)
        a_mat = identity - gd * chi
        x_field = torch.linalg.solve(a_mat, e_inc)
        return torch.matmul(gs * chi, x_field)


model = DDOFModel(chi_init_real, chi_init_imag)

mse_criterion = nn.MSELoss()


def compute_mse_loss(pred, target):
    """MSE on real and imaginary parts."""
    return mse_criterion(pred.real, target.real) + mse_criterion(pred.imag, target.imag)


def compute_l1_loss(real, imag):
    """PnP L1 regularization term."""
    return torch.sum(torch.abs(real)) + torch.sum(torch.abs(imag))


def compute_tv_loss(real, imag):
    """PnP TV regularization term on the 2D susceptibility map."""
    real_2d = ColumnFirstReshape(real, (DIM_64, DIM_64))
    imag_2d = ColumnFirstReshape(imag, (DIM_64, DIM_64))

    def _tv_2d(x):
        diff_row = x[1:, :] - x[:-1, :]
        diff_row = torch.cat([diff_row, torch.zeros_like(diff_row[-1:, :])], dim=0)

        diff_col = x[:, 1:] - x[:, :-1]
        diff_col = torch.cat([diff_col, torch.zeros_like(diff_col[:, -1:])], dim=1)

        return torch.sum(torch.sqrt(diff_row ** 2 + diff_col ** 2 + TV_EPS))

    return _tv_2d(real_2d) + _tv_2d(imag_2d)


# L-BFGS optimizer with strong Wolfe line search
optimizer = optim.LBFGS(
    model.parameters(),
    lr=1,
    max_iter=LBFGS_MAX_ITER,
    max_eval=None,
    tolerance_grad=LBFGS_TOL_GRAD,
    tolerance_change=LBFGS_TOL_CHANGE,
    history_size=LBFGS_HISTORY_SIZE,
    line_search_fn="strong_wolfe",
)

print("Start training and optimization")
print("-" * 36)

iteration_count = 0
_iter_state = {"prev_count": 0}

_conv_state = {"count": 0, "converged": False}


class EarlyStopSignal(Exception):
    """Raised when early stopping is triggered."""
    pass


def closure():
    """Evaluate loss, gradients, and convergence state for L-BFGS."""
    global iteration_count

    optimizer.zero_grad()

    output = model(e_inc, gd_mat, gs_mat)

    # Scattered-field mean relative error
    re_error = Evaluate_Relative_Error(output, e_sca, reduction="mean")
    mre_sf_pct = re_error.item() * 100.0

    r, c = model.param_real, model.param_imag
    loss_mse = compute_mse_loss(output, e_sca)
    loss_tv = compute_tv_loss(r, c)
    loss_l1 = compute_l1_loss(r, c)

    loss_total = loss_mse + LAMBDA_TV * loss_tv + LAMBDA_L1 * loss_l1

    loss_total.backward()

    iteration_count += 1

    # Early-stopping tracking
    if ENABLE_EARLY_STOP:
        if mre_sf_pct < MRE_SF_CONVERGE_THRESHOLD:
            _conv_state["count"] += 1
            converged_note = " ✓" if _conv_state["count"] >= MRE_SF_CONVERGE_COUNT else ""
        else:
            _conv_state["count"] = 0
            converged_note = ""
    else:
        converged_note = ""

    # Progress logging
    if iteration_count % PRINT_INTERVAL == 0:
        print(
            f"[{iteration_count:04d}]-> "
            f"Total: {loss_total.item():.18f} | "
            f"MRE-SF: {mre_sf_pct:.6f}%"
            f"{converged_note}"
        )

    # Stop once convergence is stable
    if ENABLE_EARLY_STOP and _conv_state["count"] >= MRE_SF_CONVERGE_COUNT:
        _conv_state["converged"] = True
        raise EarlyStopSignal(loss_total)

    return loss_total


# Training loop
for epoch in range(NUM_EPOCHS):
    _iter_state["prev_count"] = iteration_count
    try:
        final_loss = optimizer.step(closure)
    except EarlyStopSignal as e:
        final_loss = e.args[0]
    inner_steps = iteration_count - _iter_state["prev_count"]

    print("-" * 36)
    if ENABLE_EARLY_STOP and _conv_state["converged"]:
        print(f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}], Loss: {final_loss.item():.18f}, "
              f"Inner steps: {inner_steps}")
        print(f">>> Converged! MRE-SF < {MRE_SF_CONVERGE_THRESHOLD}% for "
              f"{MRE_SF_CONVERGE_COUNT} consecutive iterations. Early stopping. <<<")
        print("-" * 36)
        break

    print(f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}], Loss: {final_loss.item():.18f}, "
          f"Inner steps: {inner_steps}")
    print("-" * 36)

print(f"Training finished — total inner iterations: {iteration_count}")

# Save learned susceptibility
chi_learned = torch.complex(model.param_real.data, model.param_imag.data)

print("\nLearned Chi:")
print(chi_learned)
print("\nGround Truth Chi:")
print(chi_gt)

scipy.io.savemat(
    os.path.join(DATA_DIR, OUTPUT_FILE),
    {"Chi_DDOF": chi_learned.cpu().numpy()},
)
print(f"\nResult saved to {os.path.join(DATA_DIR, OUTPUT_FILE)}")
