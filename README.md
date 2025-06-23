# BPINN-Fracture-Conductivity-Correction

**Bayesian Physics-Informed Neural Network (BPINN) for correcting model misspecification in fracture conductivity estimation**

This repository implements a physics-informed, AI-driven framework to reconstruct **hydraulic aperture fields** from mechanical aperture maps, accounting for fracture roughness and quantifying the associated uncertainties.

---
## Overview
Traditional models such as the **Cubic Law** and **Darcy-based upscaling** often **overestimate fracture conductivity**, especially in rough or complex geometries. This project addresses those limitations by:
- Reconstructing latent **hydraulic aperture fields** $a_h(x,y)$
- Applying **Bayesian inference** to correct empirical models
- Combining **data-driven learning** with **physics-based constraints**
- Producing **local permeability maps** $K_{NN}^{ah}(x, y)$ reliable for **upscaling**
  
We employ **Bayesian Physics-Informed Neural Network (BPINN)** for uncertainty-aware inference enhanced by the **AW-HMC sampler** for adaptive task balancing across objectives

---
## Main Code Modules
### `Sampling.py` — Bayesian Inference with B-PINN
Performs **Bayesian inference** to reconstruct the **latent hydraulic aperture field** from noisy mechanical aperture data. Outputs posterior samples of the neural network parameters for hydraulic aperture and permeability fields inference.
### `Prediction.py` — Posterior Averaging and Visualization
Processes the output of `sampling.py` to compute the **Bayesian Model Average (BMA)**, local uncertainties, and final corrected permeability maps. Also generates diagnostic plots and model comparisons.

---
## Citation
This repository contains the supplementary code to the following manuscript: 
S. Perez et al. "When Cubic Law and Darcy Fail: Bayesian Correction of Model Misspecification in Fracture Conductivities".
If you use this work, please cite the associated paper, currently its ArXiv version https://arxiv.org/abs/2503.20788


