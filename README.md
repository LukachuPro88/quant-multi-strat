# Quantitative Finance with Multiple Trading Algorithms

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Official software implementation and research suite accompanying the paper ***Quantitative Finance with Multiple Trading Algorithms***. 

This repository provides a modular, type-safe, and reproducible Python backtesting and execution engine designed to evaluate classical technical indicators, tree-based machine learning models, and deep neural networks under unified risk and execution parameters.

---

## Architecture & Project Structure

The project follows a clean `src/` layout to separate Python source code, research documentation, data feeds, and generated results.

```text
quantitative-finance-with-multiple-trading-algorithms/
├── docs/                      # LaTeX research report source files
│   └── Report.tex             # Main paper manuscript
├── src/                       # Primary Python framework
│   ├── __init__.py
│   ├── config.py              # Centralized global parameters & signal constants
│   ├── backtest.py            # Backtesting engine & trade execution logic
│   ├── data.py                # Data loading, cleaning, and feature generation
│   ├── main.py                # Primary execution entry point
│   ├── analysis/              # Performance evaluation, PnL, & metrics
│   └── strategies/            # Isolated, type-hinted strategy implementations
│       ├── __init__.py
│       ├── buy_and_hold.py
│       ├── momentum.py
│       ├── moving_average.py
│       ├── mean_reversion.py
│       ├── knn_momentum.py
│       ├── xgboost.py
│       └── nn.py              # Deep Neural Network strategy (PyTorch)
├── data/                      # Historical price data (git-ignored)
├── results/                   # Generated backtest metrics & plots (git-ignored)
├── .gitignore
├── pyproject.toml             # Package setup & development configuration
├── README.md
└── LICENSE
```