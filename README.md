# Code for Simulation Experiments

This repository contains the Python code and Jupyter notebooks used for the simulation experiments in the paper.

## Files

- `g_standard.py`: generates the standard PA network.
- `g_anomaly.py`: generates the PA network with an anomaly.
- `pa_loglikelihood.py`: calculates the log-likelihood function.
- `estimation_pa_normal.py`: estimates $\delta$ under the null hypothesis (standard PA network).
- `estimation_pa_anomaly_fixed.py`: estimates parameters separately.
- `estimation_pa_anomaly_joint.py`: estimates $\beta$ and $\delta$ simultaneously.
- `support_tools.py`, `pa_core.py`: helper functions used in the simulations.
- `detection_delta_known.ipynb`: notebook for running detection experiments with known $\delta$.
- `detection_delta_unknown_H1.ipynb`: notebook for running detection experiments with unknown $\delta$.
- `experiment_estimation.ipynb`: notebook for running estimation experiments.
- `numerical_analysis.ipynb`: notebook for analyzing additional results and generating figures.

## Requirements

The code was written in Python and uses standard scientific computing packages, including NumPy, pandas, matplotlib, Numba, and SciPy.

## Usage

Open the notebooks in Jupyter Notebook or JupyterLab and run the cells in order.
