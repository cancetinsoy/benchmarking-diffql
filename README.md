# Benchmarking Differential Q-Learning and RVI Q-Learning using Storm

This project benchmarks two average-reward RL algorithms—**Differential Q-Learning** (DiffQL) and **Relative Value Iteration Q-Learning** (RVIQL)—by comparing their learned results to exact values computed using the **Storm model checker**.

## Overview

- Custom MDPs created in Storm’s explicit format (`.tra`, `.lab`, `.rew`)
- Each algorithm is run **1000 times per environment**
- Storm is used to compute the ground-truth long-run average reward (LRA)
- Results include:
  - Statistical summaries
  - Histograms and running mean plots

## Contents

- `DiffQL.ipynb & RVIQL.ipynb`: Jupyter notebooks for DiffQL and RVIQL
- `mdp-explicit/`: All tested MDPs in explicit format
- `results/`: CSVs and plots for all runs
- `report.pdf`: Final research report

authored by Can Çetinsoy — 2025 master’s research project