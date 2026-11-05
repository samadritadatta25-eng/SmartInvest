# SmartInvest

**Python-Based Stock Screening, Portfolio Optimization and Personalized Investment Allocation**

IPP (Introduction to Python Programming) project, BBA (Honours), TAPMI Bengaluru.

## What it does
SmartInvest is a Streamlit web app for retail investors. The investor uploads daily
stock prices and a market benchmark (for example NIFTY 50), and the app:

1. Cleans and aligns the data by trading date.
2. Calculates annual return, volatility, beta (CAPM regression) and Jensen's alpha.
3. Ranks the stocks by Jensen's alpha and selects the top N.
4. Builds the maximum Sharpe ratio portfolio (SciPy SLSQP, long-only).
5. Shows the Security Market Line, Efficient Frontier and Capital Allocation Line.
6. Splits the budget between the risky portfolio and the risk-free asset using
   y* = (E(Rp) - Rf) / (A x Variance), capped between 0% and 100%.
7. Gives the rupee amount for each stock and a downloadable investment report.

## Investor inputs
- Stock price file (CSV or Excel; Yahoo Finance multi-sheet workbooks supported)
- Benchmark file (CSV or Excel)
- Annual risk-free rate (default 4%)
- Investment budget (default Rs 1,00,000)
- Number of stocks to select (default 3)
- Risk aversion: Aggressive (A = 1), Moderate (A = 2), Conservative (A = 4)

## Project files
| File | Purpose |
|---|---|
| `app.py` | Website layout, tabs and text |
| `data_processing.py` | Reading, cleaning and aligning price files |
| `financial_engine.py` | CAPM, optimization, frontier, allocation, report |
| `charts.py` | All Matplotlib charts |
| `requirements.txt` | Python libraries needed |
| `.streamlit/config.toml` | Website colour theme |
| `sample_data/` | Demo files (Dabur, Wipro, Suzlon, Welspun Living, NIFTY 50) |

## Run locally or in Colab
```
pip install -r requirements.txt
streamlit run app.py
```

## Assumptions and limitations
- 252 trading days per year; daily risk-free rate = annual rate / 252.
- Historical estimates do not guarantee future performance.
- NIFTY 50 is a price index (no dividends), which can slightly overstate alpha.
- Educational tool only, not investment advice.
