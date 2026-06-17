# A Differentiable Discrete Optimization Framework (DDOF) for 2D Electromagnetic Inverse Scattering

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-required-ee4c2c)
![Status](https://img.shields.io/badge/status-research%20code-lightgrey)
![Target](https://img.shields.io/badge/target-IEEE%20AWPL-00629B)

This repository provides the implementation and data for our DDOF-based
two-dimensional electromagnetic inverse scattering reconstruction work.
The project is prepared as companion code for a manuscript planned for
submission to **IEEE Antennas and Wireless Propagation Letters (AWPL)**.

The code includes preprocessed `.mat` data, the DDOF optimization pipeline,
evaluation utilities, and visualization scripts for reproducing the
reconstruction result used in the paper.

## Preview

The reconstruction script generates a six-panel comparison figure containing
the ground truth, DDOF reconstruction, and reconstruction error for both the
real and imaginary parts.

![DDOF reconstruction result](ImageSave/Case/ImagingAll_six_panel.png)

## Repository Structure

```text
DDOF_Project/
├── Data/
│   ├── Chi.mat
│   ├── Chi0.mat
│   ├── Einc.mat
│   ├── Esca.mat
│   ├── GD.mat
│   ├── GS.mat
│   ├── Etot.mat
│   └── Chi_DDOF.mat
├── FuncLib/
│   ├── MatrixFunc.py
│   ├── EvaluFunc.py
│   └── ImageFunc.py
├── ProcOpt/
│   └── DDOF.py
├── Imaging/
│   └── visualize.py
└── ImageSave/Case/
    └── ImagingAll_six_panel.png
```

## Requirements

The project is tested with Python 3.10+. The main dependencies are:

- `torch`
- `scipy`
- `numpy`
- `matplotlib`

Install them with:

```bash
pip install torch scipy numpy matplotlib
```

If CUDA is available, `ProcOpt/DDOF.py` automatically uses GPU acceleration.
Otherwise, it runs on CPU.

## Quick Start

Clone or enter the project directory, then run:

```bash
python ProcOpt/DDOF.py
python Imaging/visualize.py
```

The DDOF reconstruction result will be saved to:

```text
Data/Chi_DDOF.mat
```

The visualization result will be saved to:

```text
ImageSave/Case/ImagingAll_six_panel.png
```

Note that `ProcOpt/DDOF.py` overwrites the existing `Data/Chi_DDOF.mat`.
The optimization solves a `4096 x 4096` complex linear system during each
iteration, so CPU execution may take some time.

## Data Description

The data follow the matrix notation used in the manuscript:

| Physical interpretation | Matrix representation | Matrix size |
| --- | --- | --- |
| Total field | $\mathbf{E}^{t}$ | $\mathbf{E}^{t} \in \mathbb{C}^{N_e \times N_t}$ |
| Incident field | $\mathbf{E}^{i}$ | $\mathbf{E}^{i} \in \mathbb{C}^{N_e \times N_t}$ |
| Scattered field | $\mathbf{E}^{s}$ | $\mathbf{E}^{s} \in \mathbb{C}^{N_r \times N_t}$ |
| Contrast | $\chi$ | $\chi \in \mathbb{C}^{1 \times N_e}$ |
| Green's function | $\mathbf{G}_{D}$ | $\mathbf{G}_{D} \in \mathbb{C}^{N_e \times N_e}$ |
| Green's function | $\mathbf{G}_{S}$ | $\mathbf{G}_{S} \in \mathbb{C}^{N_r \times N_e}$ |

$N_e$ denotes the number of elements in the contrast $\chi$.
For the current `64 x 64` grid, $N_e = 4096$. $N_t$ represents the number
of transmitters, and $N_r$ represents the number of receivers. In the
computation, the contrast $\chi$ is reshaped into a diagonal
matrix when used in the forward model.

## Method Overview

The forward scattering model used in `ProcOpt/DDOF.py` is:

```text
A     = I - G_D diag(Chi)
Solve A E_tot = E_inc
E_sca = G_S diag(Chi) E_tot
```

The optimization objective combines scattered-field data fidelity with PnP
regularization:

```text
Loss = MSE(E_sca_pred, E_sca_meas)
     + lambda_TV * TV(Chi)
     + lambda_L1 * L1(Chi)
```

The TV and L1 terms are used as PnP regularization priors to constrain the
spatial structure and sparsity of the reconstructed susceptibility.

## Usage

### Run DDOF Reconstruction

```bash
python ProcOpt/DDOF.py
```

This script:

1. Loads the `.mat` data from `Data/`;
2. Reshapes `GD` and `GS` from MATLAB column-major storage;
3. Builds the differentiable forward scattering model;
4. Optimizes the susceptibility using L-BFGS;
5. Applies PnP regularization;
6. Saves the result as `Data/Chi_DDOF.mat`.

### Generate Visualization

```bash
python Imaging/visualize.py
```

This script loads `Chi.mat` and `Chi_DDOF.mat`, computes the reconstruction
error, and generates the six-panel comparison figure.

The default color ranges are:

```text
Real part:      [0, 2]
Imaginary part: [-2, 0]
```

To change the display range, edit `VMIN_REAL`, `VMAX_REAL`, `VMIN_IMAG`, and
`VMAX_IMAG` in `Imaging/visualize.py`.

## Main Files

| Path | Purpose |
| --- | --- |
| `ProcOpt/DDOF.py` | Main DDOF optimization script |
| `FuncLib/MatrixFunc.py` | Complex tensor and matrix utilities |
| `FuncLib/EvaluFunc.py` | Error metric utilities |
| `FuncLib/ImageFunc.py` | Figure generation utilities |
| `Imaging/visualize.py` | Visualization entry point |

## Citation

This repository corresponds to our DDOF electromagnetic inverse scattering
imaging work, planned for submission to:

```text
IEEE Antennas and Wireless Propagation Letters (AWPL)
```

The full citation will be updated after the manuscript is accepted or made
public:

```bibtex
@article{ddof_awpl,
  title   = {Paper Title},
  author  = {Author Name and Author Name},
  journal = {IEEE Antennas and Wireless Propagation Letters},
  year    = {Year}
}
```

## Notes

- This repository is research code for paper reproduction.
- The current version uses fixed data dimensions: `64 x 64` grid and `32`
  transmitters/receivers.
- For new experiments, make sure the variable names inside the `.mat` files
  match their file names, such as `Chi.mat` containing variable `Chi`.
- The PnP parameters and other settings should be adjusted according to the
  actual use case.
