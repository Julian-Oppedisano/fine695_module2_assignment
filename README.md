# FINE 695 - Machine Learning in Finance: Portfolio Strategy

This repository contains the implementation of various machine learning models for stock return prediction and portfolio management, developed as part of the FINE 695 course at McGill University.

## Repository Structure

```
fine695_module2_assignment/
├── Data/                      # Data directory (not included in repo)
│   ├── homework_sample_big.csv
│   ├── factor_char_list.csv
│   └── SPY returns.xlsx
├── outputs/                   # Generated outputs
│   ├── perf_summary.csv      # Performance metrics
│   ├── metrics.csv           # Out-of-sample R²
│   ├── cum_returns.png       # Cumulative returns plot
│   └── *_port_ret.csv        # Portfolio returns
├── slides/                    # Presentation slides
│   └── Module2_template.pptx  # 5-slide presentation
├── model_*.py                # Model implementation scripts
├── utils.py                  # Utility functions
├── create_slide_deck.py      # Slide generation script
├── select_best_model.py      # Model selection script
└── requirements.txt          # Python dependencies
```

## Key Features

- Implementation of 5 machine learning models:
  - Lasso Regression
  - Ridge Regression
  - Elastic Net
  - Decision Trees
  - Neural Networks (NN2)
- Time window approach:
  - 10-year initial training (2005-2014)
  - 2-year validation (2015-2016)
  - 1-year out-of-sample testing (2017)
- Portfolio construction:
  - Top 50 stocks selection
  - Equal-weighted allocation
  - Monthly rebalancing
- Performance metrics:
  - Alpha
  - Sharpe ratio
  - Average returns
  - Volatility
  - Maximum drawdown
  - Maximum one-month loss

## Results

The ElasticNet model was selected as the best performing strategy, achieving:
- Monthly alpha: 0.35%
- Annualized Sharpe ratio: 3.05
- Monthly volatility: 7.55%
- Maximum drawdown: 3.06%

Full performance comparison with SPY and detailed methodology can be found in the presentation slides.

## Setup and Reproduction

1. Clone the repository:
```bash
git clone https://github.com/BTCJULIAN/fine695_module2_assignment.git
cd fine695_module2_assignment
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place data files in the `Data/` directory:
   - `homework_sample_big.csv`
   - `factor_char_list.csv`
   - `SPY returns.xlsx`

5. Run the models:
```bash
python model_lasso.py
python model_ridge.py
python model_elasticnet.py
python model_tree.py
python model_nn2.py
```

6. Generate the presentation:
```bash
python create_slide_deck.py
```

## Presentation

The 5-slide presentation (`slides/Module2_template.pptx`) includes:
1. Methodology description
2. Out-of-sample R² analysis
3. Performance statistics comparison
4. Cumulative returns visualization
5. Summary and future improvements

## Author

Julian BTC

## License

This project is part of the FINE 695 course at McGill University. 