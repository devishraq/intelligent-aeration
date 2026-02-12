# Intelligent Aeration: Balancing Water Quality & Energy Savings Through Constraint-Adaptive Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Stable Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://github.com/DLR-RM/stable-baselines3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for the research paper **"Intelligent Aeration: Balancing Water Quality & Energy Savings Through Constraint-Adaptive Reinforcement Learning"**, submitted for the **Bangladesh Stockholm Junior Water Prize (SJWPBD) 2026** National Competition.

## 🌟 Overview

Wastewater treatment plants (WWTPs) are critical infrastructure that consume significant energy, primarily for **aeration**. Traditional control strategies often struggle to balance the trade-off between energy conservation and strict effluent quality standards (SNH4, NTot).

This project implements a Reinforcement Learning (RL) framework using the **BSM1 (Benchmark Simulation Model No. 1)** environment to optimize aeration control. The core innovation is the **ZAMOR (Zone-Adaptive Multi-Objective Reward)** function, which enables agents to learn complex control policies that respect physical constraints while minimizing energy footprint.

### Key Contributions:
- **ZAMOR Reward Function**: A physically-motivated reward including logarithmic energy costs, three-zone constraint penalties, and log-barrier soft safety walls.
- **Multi-Algorithm Benchmark**: Comparative analysis of state-of-the-art RL algorithms (SAC, TD3, PPO, DDPG, and CATD3).
- **Curriculum Learning**: Multi-phase training strategy (Dry, Rain, Storm) to ensure policy robustness across weather conditions.
- **Pareto Efficiency**: Analysis of the optimal trade-off between energy consumption and effluent quality.

---

## 🏗️ Project Structure

```text
├── data/               # BSM1 Inflow data (Dry, Rain, Storm 2006)
├── paper/              # LaTeX source for the research paper
├── results/            # Generated metrics, plots, and JSON results
├── scripts/            # Helper scripts for evaluation and metrics
├── src/
│   ├── algo/           # Custom RL algorithm implementations (CATD3, SAC, etc.)
│   ├── config/         # Hyperparameters and curriculum settings
│   ├── environment/    # BSM1 Gym environment and ZAMOR reward logic
│   ├── evaluation/     # Metrics logging and performance evaluation
│   ├── training.py     # Main training and benchmarking orchestration
│   └── cli.py          # Command-line interface entry point
└── requirements.txt    # Project dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- (Optional but recommended) A virtual environment

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/devishraq/intelligent-aeration.git
   cd intelligent-aeration
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Usage

The project uses a unified CLI for all operations.

### Run Full Benchmark
To train all algorithms and generate a comparative report:
```bash
python -m src --action benchmark --mode demo
```
*Use `--mode full` for high-fidelity training (takes longer).*

### Train a Specific Algorithm
```bash
python -m src --action train --method sac --mode full
```

### Generate Pareto Frontier
```bash
python -m src --action frontier --mode demo
```

### Options:
- `--action`: `train`, `benchmark`, `frontier`, `full`, `clean`
- `--method`: `sac`, `td3`, `ppo`, `ddpg`, `catd3`
- `--mode`: `demo` (fast) or `full` (research-grade)
- `--seed`: Set random seed for reproducibility

---

## 🧪 Experiments & Algorithms

We utilize **Stable-Baselines3** as the backend for robust RL implementations, with custom modifications in `src/algo/` for specialized architectures:

| Algorithm | Description |
| :--- | :--- |
| **SAC** | Soft Actor-Critic (Off-policy, maximum entropy) |
| **TD3** | Twin Delayed DDPG (Addresses overestimation bias) |
| **PPO** | Proximal Policy Optimization (On-policy, stable) |
| **CATD3** | Constraint-Aware TD3 (Custom implementation for safe RL) |

### Training Curriculum
Agents are trained through three distinct weather phases:
1. **Dry Phase**: Stable inflow for baseline stability.
2. **Rain Phase**: Increased volume to test flow adaptation.
3. **Storm Phase**: Extreme peak events to verify constraint adherence.

---

## 📊 Results

All results are stored in the `results/` directory as JSON files and visualization plots. You can find:
- `evaluation.json`: Detailed metrics for effluent violations and energy usage.
- `plots/`: Trajectory comparisons and reward curves.

---

## 👥 Authors

- **Ishraq Tanvir** - [devishraq@gmail.com](mailto:devishraq@gmail.com)
- **Kazi Wasi-Uddin Saad** - [shadkazisami@gmail.com](mailto:shadkazisami@gmail.com)

**Bangladesh Stockholm Junior Water Prize (SJWPBD) 2026**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The **BSM1** community for providing the simulation standards.
- **Stockholm Junior Water Prize Bangladesh** for the opportunity to showcase this research.
- **Stable-Baselines3** developers for their excellent RL library.
