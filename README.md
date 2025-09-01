# Benchmarking Differential Q-Learning and RVI Q-Learning using Storm

This project benchmarks two average-reward RL algorithms—**Differential Q-Learning** (DiffQL) and **Relative Value Iteration Q-Learning** (RVIQL)—by comparing their learned results to exact values computed using the **Storm model checker**.

## Overview

- Custom MDPs created in Storm’s explicit format (`.tra`, `.lab`, `.rew`)
- Each algorithm is run **1000 times per environment**
- Storm is used to compute the long-run average reward (LRA)
- Results include:
  - Statistical summaries
  - Histograms and running mean plots

## Contents

- `DiffQL.ipynb & RVIQL.ipynb`: Jupyter notebooks for DiffQL and RVIQL
- `explicit2py_converter.py`: A script that converts Storm's exlplicit format files to Python environments
- `access-control-generator.py`: A script that creates the explicit files for the Access Control Queuing Task environment
- `mdp-explicit/`: All tested MDPs in explicit format
- `results/`: CSVs and plots for all runs
- `Benchmarking Differential Q Learning and RVI Q Learning using Storm.pdf`: Final research report

authored by Can Çetinsoy — 2025 master’s research project