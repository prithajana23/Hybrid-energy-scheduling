# Constraint-Aware Hybrid Framework for Smart Home Energy Scheduling

This repository provides the implementation of a **deployment-aware hybrid framework** that integrates probabilistic load forecasting and reinforcement learning for smart home energy scheduling under uncertainty.

The framework is designed to bridge the gap between **prediction and decision-making** by explicitly incorporating uncertainty into the scheduling process while ensuring constraint-aware operation.

---

## Overview

Smart home energy management systems (SHEMS) aim to reduce electricity costs and alleviate peak demand stress on the grid. However, most existing approaches treat **load forecasting and scheduling as separate tasks**, leading to suboptimal performance under uncertainty.

This work proposes a **forecast-to-decision pipeline** that combines:

- Probabilistic forecasting (DeepAR-style LSTM)
- Reinforcement learning (DQN, PPO, Risk-aware DQN)
- Constraint-aware scheduling via feasibility-preserving heuristics
- Optimization baselines (MILP)

The framework achieves **near-zero peak violations** while maintaining competitive electricity cost and user comfort.

---

## Key Features

### Probabilistic Forecasting
- LSTM-based DeepAR-style model
- Outputs mean (μ) and uncertainty (σ)
- Gaussian likelihood training
- Early stopping for stability

### Hybrid Scheduling Framework
- Combines forecasting + RL + optimization principles
- Safety-aware scheduling using uncertainty bounds
- Feasibility-preserving block construction
- Multi-objective optimization:
  - Cost minimization
  - Peak violation reduction
  - User comfort
  - Switching minimization

### Reinforcement Learning
- DQN (baseline)
- PPO (policy gradient)
- Risk-aware DQN (enhanced violation penalty)

### Optimization Baselines
- Greedy scheduling
- Price-based block scheduling
- MILP (Mixed-Integer Linear Programming):
  - Deterministic (oracle)
  - Forecast-based
  - Robust (quantile-based)
  - Strict peak constraint variant

### Statistical Analysis
- Wilcoxon signed-rank test
- Rank-biserial effect size
- Multi-seed evaluation

---

## Datasets
- **UCI Individual Household Electric Power Consumption**: <https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption>
- **AEP Hourly Energy Consumption (Kaggle)**: <https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption>
- **Solar Plant Dataset 1 (Kaggle)**: <https://www.kaggle.com/datasets/anikannal/solar-power-generation-data?select=Plant_1_Generation_Data.csv>
- **Solar Plant Dataset 2 (Kaggle)**: <https://www.kaggle.com/datasets/anikannal/solar-power-generation-data?select=Plant_2_Generation_Data.csv>


> Note: Datasets are not included in this repository due to size/licensing restrictions.
> Please download them from the links above.
> Preprocessing is implemented inside the training scripts using the 'get_datasets()' function.

---

### Note

If the dataset is not found, the code automatically generates **synthetic data** for testing purposes.

---

## Installation

Install all required dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn torch gymnasium stable-baselines3 pulp seaborn scipy
