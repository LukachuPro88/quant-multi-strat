import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import src.data as data
import src.config as config
from src.backtest import backtest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run quantitative backtests and generate comparison plots."
    )

    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="If set, graphs will be displayed interactively during runtime.",
    )
    parser.add_argument(
        "--strategy",
        choices=config.STRATEGIES,
        help="Specific strategy to run. If omitted, all strategies are executed.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=config.TICKERS,
        help="List of ticker symbols to backtest (e.g. AAPL CAT KO TSLA).",
    )
    args = parser.parse_args()

    for ticker in args.tickers:
        ticker = ticker.upper()
        print(f"\n{'=' * 40}")
        print(f"Running Backtests for: {ticker}")
        print(f"{'=' * 40}")

        data_path = config.DATA_DIR / f"{ticker}.csv"

        if not data_path.exists():
            print(
                f"Error: Data file for '{ticker}' not found at {data_path}. Skipping..."
            )
            continue

        df = data.get_data(data_path)

        if args.strategy:
            results = {
                args.strategy: backtest(df, args.strategy),
                "buyandhold": backtest(df, "buyandhold"),
            }
        else:
            results = {strat: backtest(df, strat) for strat in config.STRATEGIES}

        bah_result = results.get("buyandhold")

        if bah_result is None:
            raise KeyError(
                "The 'buyandhold' strategy is required as a baseline benchmark."
            )

        output_dir = config.RESULTS_DIR / ticker
        output_dir.mkdir(parents=True, exist_ok=True)

        for strat_name, strat_df in results.items():
            if strat_name == "buyandhold":
                continue

            # Create the plot
            fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)

            bah_plot = bah_result

            # Take training data into account for machine learning algorithms
            if strat_name in {"nn", "xgboost"}:
                bah_plot = bah_result.loc[strat_df.index.min() :]

            ax.plot(
                bah_plot.index,
                bah_plot["Portfolio Value"],
                label="Buy & Hold",
                color="tab:gray",
                linestyle="--",
                linewidth=1.5,
            )

            formatted_title = strat_name.replace("_", " ").title()

            ax.plot(
                strat_df.index,
                strat_df["Portfolio Value"],
                label=formatted_title,
                color="tab:blue",
                linewidth=2.0,
            )

            ax.set_title(
                f"{ticker} — {formatted_title} vs. Buy & Hold",
                fontweight="bold",
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value ($)")

            ax.legend(
                loc="upper left",
                frameon=True,
                fontsize=12,
            )

            ax.grid(True, linestyle=":", alpha=0.6)

            fig.tight_layout()

            # OS independent path
            output_path = output_dir / f"{ticker}_{strat_name}_vs_bah.png"
            fig.savefig(output_path, dpi=config.PLOT_DPI)

            # Show the process blocking graph if wanted
            # Otherwise skip over it
            if args.show_graph:
                plt.show()

            plt.close(fig)


if __name__ == "__main__":
    main()
