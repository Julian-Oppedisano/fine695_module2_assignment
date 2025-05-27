# FINE 695 - Machine Learning in Finance: Portfolio Strategy

This repository contains the implementation of various machine learning models for stock return prediction and portfolio management, developed as part of the FINE 695 course at McGill University.

## Repository Structure

```
.
├── Data/                      # Data directory (not included in repo)
│   ├── homework_sample_big.csv
│   ├── factor_char_list.csv
│   └── SPY returns.xlsx
├── outputs/                   # Generated outputs
│   ├── perf_summary.csv      # Aggregated performance metrics for all models
│   ├── metrics.csv           # Aggregated out-of-sample R² for all models
│   ├── best_alg.txt          # Stores the name of the best performing algorithm
│   ├── cum_returns.png       # Cumulative returns plot for the best algorithm vs SPY
│   ├── Module3_Report_{best_model_name}.pptx # Generated presentation for the best model
│   └── *_overall_port_ret.csv # Overall portfolio returns for each model
├── slides/                    # Contains the presentation template
│   └── Module2_template.pptx
├── model_lasso.py             # Lasso Regression model script
├── model_ridge.py             # Ridge Regression model script
├── model_elasticnet.py        # Elastic Net model script
├── model_tree.py              # Decision Tree model script
├── model_nn2.py               # Neural Network (2-layer) model script
├── model_ipca.py              # Instrumented PCA model script
├── model_autoencoder.py       # Autoencoder model script
├── utils.py                   # Utility functions (e.g., window generation)
├── portfolio_utils.py         # Utility functions for portfolio calculations
├── load_data.py               # Script for loading initial data (if needed separately)
├── preprocess.py              # Script for preprocessing steps (if used directly)
├── create_slide_deck.py       # Slide generation script, uses data from outputs/
├── select_best_model.py       # Model selection script, writes to outputs/best_alg.txt
├── plot_cumulative_returns.py # Script to generate cumulative returns plot
├── requirements.txt           # Python dependencies
└── .gitignore                 # Specifies intentionally untracked files

```

## Key Features

- Implementation of 7 machine learning models for stock return prediction:
  - Lasso Regression
  - Ridge Regression
  - Elastic Net
  - Decision Trees
  - Neural Networks (NN2)
  - Instrumented Principal Component Analysis (IPCA) - Module 3
  - Autoencoder - Module 3
- Expanding time window approach for robust backtesting:
  - Initial 10-year training period (e.g., 2005-2014)
  - Initial 2-year validation period (e.g., 2015-2016) for hyperparameter tuning
  - Out-of-sample (OOS) testing from 2017 through 2023, with models retrained/recalibrated annually on an expanding basis.
- Portfolio construction:
  - Selection of top 50 stocks based on model predictions.
  - Equal-weighted allocation.
  - Monthly rebalancing.
- Performance metrics tracked:
  - Alpha (monthly)
  - Sharpe ratio (annualized)
  - Average monthly returns
  - Monthly volatility
  - Maximum drawdown
  - Maximum one-month loss
  - Out-of-Sample R²

## Results

The project evaluates multiple models, with `select_best_model.py` determining the best performing one based on criteria like the Sharpe Ratio. Currently, the Neural Network (NN2) model is often selected. Instrumented PCA (IPCA) and Autoencoder models were implemented as part of Module 3 requirements to explore advanced techniques.

Detailed performance comparisons, including the best model against SPY and individual R² metrics for all models (including IPCA and Autoencoder), can be found in the dynamically generated PowerPoint presentation: `outputs/Module3_Report_{best_model_name}.pptx`.

## Setup and Reproduction

1.  Clone the repository:
    ```bash
    git clone https://github.com/Julian-Oppedisano/fine695_module2_assignment.git
    cd fine695_module2_assignment
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Ensure data files are placed in the `Data/` directory (these files are not included in the repository):
    *   `homework_sample_big.csv` (Main dataset of stock characteristics and returns)
    *   `factor_char_list.csv` (List of predictor variables)
    *   `SPY returns.xlsx` (S&P 500 benchmark returns)

5.  Run all the individual model scripts to generate their predictions and performance files:
    ```bash
    python model_lasso.py
    python model_ridge.py
    python model_elasticnet.py
    python model_tree.py
    python model_nn2.py
    python model_ipca.py
    python model_autoencoder.py
    ```

6.  Run utility scripts to select the best model, plot returns, and generate the final presentation:
    ```bash
    python select_best_model.py
    python plot_cumulative_returns.py
    python create_slide_deck.py
    ```
    The final presentation will be saved in `outputs/Module3_Report_{best_model_name}.pptx`.

## Presentation

The 5-slide presentation, dynamically generated by `create_slide_deck.py` and saved as `outputs/Module3_Report_{best_model_name}.pptx` (e.g., `Module3_Report_nn2.pptx`), includes:
1.  Detailed Methodology description for the implemented strategies.
2.  Out-of-sample R² analysis for NN2, IPCA, and Autoencoder, along with diagnostic interpretations.
3.  A comprehensive Performance Statistics Table comparing the best model, IPCA, Autoencoder, and SPY across key metrics (Alpha, Sharpe, Returns, Volatility, Max Drawdown, Max Loss).
4.  Cumulative out-of-sample returns visualization for the best performing strategy versus the S&P 500.
5.  A concise summary of key insights and potential future improvements.

## Author

Julian BTC

## License

This project is part of the FINE 695 course at McGill University. 