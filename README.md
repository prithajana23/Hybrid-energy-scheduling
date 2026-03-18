# Hybrid Forecast-to-Decision Framework for Smart Home Energy Scheduling

This repository provides the implementation of a **constraint-aware hybrid framework** that integrates probabilistic load forecasting and reinforcement learning for smart home energy scheduling under uncertainty.

The framework combines:
- Probabilistic LSTM (DeepAR-style) forecasting
- Reinforcement Learning (DQN, PPO, Risk-aware DQN)
- Feasibility-preserving hybrid scheduling
- MILP-based optimization baselines

---

## Key Features

- **Probabilistic Forecasting**
  - LSTM-based DeepAR-style model
  - Outputs mean and uncertainty (μ, σ)
  - Supports hourly, daily, and weekly forecasting

- **Hybrid Scheduling**
  - Combines forecasting + RL + constraint-aware heuristic
  - Near-zero peak violations via safety masking
  - Multi-objective optimization (cost, comfort, switching)

- **Baselines Included**
  - Greedy scheduling
  - Price-based block scheduling
  - MILP (deterministic, forecast-based, robust)
  - RL (DQN, PPO, Risk-sensitive DQN)

- **Statistical Evaluation**
  - Wilcoxon signed-rank test
  - Rank-biserial effect size

---

## Dataset

This code uses the **UCI Household Power Consumption dataset**:

https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

### Setup:
1. Download the dataset
2. Place it in the project root as:
