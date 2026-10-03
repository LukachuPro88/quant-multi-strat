# Quantitative Finance with Multiple Trading Algorithms

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A quantitative finance research project investigating the performance of multiple algorithmic trading strategies through historical backtesting.

This repository contains the software implementation and research paper for ***Quantitative Finance with Multiple Trading Algorithms***.

The project provides a unified Python framework for implementing and comparing classical trading strategies, statistical methods, machine learning models, and a neural network under consistent backtesting conditions.

## Overview

The purpose of this project is to investigate whether increasingly complex trading algorithms can outperform a simple buy and hold strategy when applied to historical market data.

The implemented strategies range from traditional rule based approaches to machine learning and deep learning models. All strategies are evaluated using the same backtesting framework and portfolio assumptions.

The project includes:

* Classical technical trading strategies
* Statistical trading strategies
* Machine learning models
* A neural network based strategy
* A unified backtesting and trade execution engine
* Performance analysis and visualization
* Reproducible historical experiments
* A complete research paper documenting the methodology and results

## Trading Strategies

The project implements the following strategies:

| Strategy       | Category         | Description                                                                |
| -------------- | ---------------- | -------------------------------------------------------------------------- |
| Buy and Hold   | Baseline         | Buys the asset and holds it throughout the testing period                  |
| Momentum       | Technical        | Generates signals based on recent price movement                           |
| Moving Average | Technical        | Uses moving average relationships to generate trading signals              |
| Mean Reversion | Statistical      | Trades deviations from a moving average                                    |
| KNN Momentum   | Machine Learning | Uses nearest historical observations to identify similar market conditions |
| XGBoost        | Machine Learning | Uses gradient boosted decision trees to classify trading signals           |
| Neural Network | Deep Learning    | Uses a PyTorch neural network to predict future prices                     |

Buy and hold serves as the baseline benchmark throughout the research, allowing the other strategies to be evaluated against a simple passive approach.

## Research

The experiments use historical market data from three individual stocks:

* **AAPL** — Apple
* **CAT** — Caterpillar
* **KO** — Coca-Cola

Each strategy is evaluated using the same backtesting framework and portfolio assumptions.

The research investigates whether increasingly sophisticated trading algorithms provide an advantage over simpler approaches when tested on historical market data.

The complete methodology, mathematical formulations, feature engineering, model architectures, experimental results, and discussion are documented in the research paper.

## Project Structure

The repository follows a `src/` layout separating the Python implementation, research documentation, input data, and generated results.

```text
quantitative-finance-with-multiple-trading-algorithms/
├── docs/
│   ├── Report.pdf             # Complete research paper
│   └── Report.tex             # LaTeX source for the paper
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Global configuration and signal constants
│   ├── backtest.py            # Backtesting and trade execution engine
│   ├── data.py                # Data loading and feature preparation
│   ├── main.py                # Command line entry point
│   │
│   ├── analysis/              # Performance analysis and metrics
│   │
│   └── strategies/
│       ├── __init__.py
│       ├── buy_and_hold.py
│       ├── momentum.py
│       ├── moving_average.py
│       ├── mean_reversion.py
│       ├── knn_momentum.py
│       ├── xgboost.py
│       └── nn.py              # PyTorch neural network strategy
│
├── data/                      # Historical market data, not tracked
├── results/                   # Generated backtest plots
├── .gitignore
├── pyproject.toml
├── README.md
└── LICENSE
```

## Installation

Clone the repository and create a virtual environment:

```bash
git clone <repository-url>
cd quantitative-finance-with-multiple-trading-algorithms

python -m venv .venv
```

Activate the virtual environment.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
pip install -e .
```

## Running the Backtests

The main entry point is `src/main.py`.

By default, all configured strategies are executed for all configured tickers:

```bash
python src/main.py
```

### Select a Strategy

Use `--strategy` to run a specific strategy:

```bash
python src/main.py --strategy momentum
```

Available strategies are:

```text
buyandhold
momentum
movingaverage
meanreversion
knnmomentum
xgboost
nn
```

When a specific strategy is selected, buy and hold is automatically included as the baseline benchmark.

### Select Tickers

Use `--tickers` to specify one or more ticker symbols:

```bash
python src/main.py --tickers AAPL CAT KO
```

Multiple tickers can be supplied in the same command:

```bash
python src/main.py --tickers AAPL CAT KO TSLA
```

Ticker symbols are automatically converted to uppercase.

### Display Graphs

By default, generated plots are saved without being displayed interactively.

Use `--show-graph` to display the plots during execution:

```bash
python src/main.py --show-graph
```

Options can also be combined:

```bash
python src/main.py --strategy xgboost --tickers AAPL CAT --show-graph
```

## Results

Generated plots are stored in `results/`, organized by ticker:

```text
results/
├── AAPL/
│   ├── AAPL_momentum_vs_bah.png
│   ├── AAPL_movingaverage_vs_bah.png
│   └── ...
├── CAT/
│   └── ...
└── KO/
    └── ...
```

Each plot compares a trading strategy against the buy and hold baseline.

For machine learning strategies such as XGBoost and the neural network, the buy and hold comparison is aligned with the strategy's available testing period after model training.

The final generated plots are included in the repository so the experimental results can be inspected directly without running the backtests.

## Reproducibility

The project is designed so that different strategies can be evaluated under consistent conditions.

The strategies use the same:

* Historical price data
* Initial capital
* Portfolio representation
* Trade execution framework
* Performance calculations

Machine learning strategies additionally perform their own feature preparation and model training according to the methodology described in the research paper.

## Research Paper

The complete research paper is available as [`docs/Report.pdf`](docs/Report.pdf).

The LaTeX source is also included in [`docs/Report.tex`](docs/Report.tex) for reproducibility and reference.

The paper contains:

* Research methodology
* Strategy definitions
* Mathematical formulations
* Feature engineering
* Machine learning methods
* Neural network architecture
* Backtesting methodology
* Experimental results
* Analysis and discussion
* Conclusions

## Technologies

The project is implemented primarily in Python using:

* **Python** for the research and backtesting framework
* **Pandas** for data manipulation
* **NumPy** for numerical computation
* **Matplotlib** for visualization
* **scikit-learn** for machine learning components
* **XGBoost** for gradient boosted decision trees
* **PyTorch** for the neural network strategy
* **LaTeX** for the research paper

## Disclaimer

This project is an academic research and software engineering project.

The strategies and results presented in this repository are based on historical data and backtesting. Historical performance does not guarantee future results, and the implementation should not be interpreted as financial advice or a recommendation to trade any security.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for more information.
