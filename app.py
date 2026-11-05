# ==================================================
# SMARTINVEST - WEBSITE (app.py)
# Python-Based Stock Screening, Portfolio Optimization
# and Personalized Investment Allocation
#
# Files used:
#   data_processing.py  -> reading, cleaning and aligning data
#   financial_engine.py -> CAPM, optimization, allocation, report
#   charts.py           -> all Matplotlib charts
# ==================================================

import io
import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_processing import (
    read_price_file, clean_price_data, align_stock_and_market,
    calculate_daily_returns,
)
from financial_engine import (
    TRADING_DAYS, calculate_stock_statistics, make_display_table,
    portfolio_performance, optimize_max_sharpe,
    calculate_efficient_frontier, generate_random_portfolios,
    calculate_utility_allocation, make_rupee_allocation,
    generate_key_insights, build_report_text,
)
from charts import (
    plot_return_risk_chart, plot_alpha_chart, plot_sml_chart,
    plot_risk_return_scatter, plot_weights_chart, plot_efficient_frontier,
    plot_capital_allocation_line, plot_risky_vs_riskfree, plot_rupee_allocation,
)

# --------------------------------
# 1. SETTINGS
# --------------------------------

MAX_STOCKS = 50

# Sample files shipped with the app (for a quick demo)
SAMPLE_STOCK_FILE = os.path.join("sample_data", "Stock_Historical_Data.xlsx")
SAMPLE_MARKET_FILE = os.path.join("sample_data", "NIFTY50_5_Year_Data.xlsx")

# Risk-aversion choices and their finalized coefficients (A)
RISK_AVERSION_OPTIONS = {
    "Low risk aversion - Aggressive (A = 1)": 1,
    "Medium risk aversion - Moderate (A = 2)": 2,
    "High risk aversion - Conservative (A = 4)": 4,
}

st.set_page_config(page_title="SmartInvest", page_icon="📈", layout="wide")


# --------------------------------
# 2. WEBSITE STYLE (CSS)
# --------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@600&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {
    font-family: 'IBM Plex Sans', 'Segoe UI', Arial, sans-serif;
}

.stApp { background: #F6F8FB; }
.block-container { padding-top: 1.2rem; max-width: 1250px; }

/* Header band */
.si-hero {
    background: #0F2440;
    color: #FFFFFF;
    padding: 2.2rem 2.4rem 1.8rem 2.4rem;
    border-radius: 6px;
    margin-bottom: 1.2rem;
}
.si-hero h1 {
    font-family: 'IBM Plex Serif', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: #FFFFFF;
    margin: 0 0 0.4rem 0;
    padding: 0;
    letter-spacing: -0.01em;
}
.si-hero p {
    color: #C9D6EA;
    font-size: 1.05rem;
    max-width: 62ch;
    margin: 0 0 1.4rem 0;
}
.si-steps { display: flex; gap: 0; flex-wrap: wrap; border-top: 1px solid #2B4467; }
.si-step {
    flex: 1 1 180px;
    padding: 0.9rem 1rem 0.2rem 0;
    color: #E6EDF7;
    font-size: 0.92rem;
}
.si-step b { color: #7FA7E8; font-weight: 600; margin-right: 0.35rem; }

/* Headings */
h2, h3 { color: #0F2440; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.2rem; border-bottom: 1px solid #D5DEEA; }
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    color: #3B4D66;
    padding: 0.6rem 0.9rem;
}
.stTabs [aria-selected="true"] { color: #0F2440; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #2A5DB0; }

/* Metrics */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border-left: 3px solid #2A5DB0;
    padding: 0.7rem 0.9rem;
    border-radius: 2px;
}
[data-testid="stMetricValue"] {
    color: #0F2440;
    font-variant-numeric: tabular-nums;
}
[data-testid="stMetricLabel"] { color: #52667F; }

/* Main button */
.stButton > button[kind="primary"] {
    background: #2A5DB0;
    border: none;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover { background: #0F2440; }
</style>
""", unsafe_allow_html=True)


# --------------------------------
# 3. ANALYSIS FUNCTION
#    Runs every step and returns all results in one dictionary.
# --------------------------------

def run_analysis(stock_file, market_file, risk_free_rate, investment_budget,
                 number_of_stocks, risk_preference):

    # ---- Step 1: read, clean, validate, align ----
    raw_stocks, stock_notes = read_price_file(stock_file)
    raw_market, market_notes = read_price_file(market_file)

    stock_prices, stock_columns, stock_report = clean_price_data(raw_stocks, "Stock file")
    market_data, market_columns, market_report = clean_price_data(raw_market, "Benchmark file")

    if len(stock_columns) > MAX_STOCKS:
        raise ValueError(f"Please upload a maximum of {MAX_STOCKS} stocks.")

    if len(market_columns) != 1:
        raise ValueError(
            "Benchmark file must contain Date and exactly one index price column. "
            f"Columns detected: {market_columns}"
        )

    market_column = market_columns[0]

    if market_column in stock_columns:
        raise ValueError(
            f"The benchmark column '{market_column}' has the same name as a stock. "
            "Please rename one of them."
        )

    aligned_data = align_stock_and_market(stock_prices, market_data, market_column)

    if number_of_stocks > len(stock_columns):
        raise ValueError(
            f"You asked for {number_of_stocks} stocks, but only "
            f"{len(stock_columns)} were uploaded."
        )

    # ---- Step 2: returns, beta, Jensen's alpha ----
    annual_rf = risk_free_rate / 100
    daily_returns = calculate_daily_returns(aligned_data)

    stock_table, market_summary = calculate_stock_statistics(
        daily_returns, stock_columns, market_column, annual_rf
    )

    alpha_check = np.allclose(
        stock_table["Jensen's Alpha"], stock_table["Regression Alpha x 252"]
    )

    selected_stocks = stock_table.head(int(number_of_stocks)).index.tolist()

    # ---- Step 4: maximum Sharpe portfolio ----
    selected_returns = daily_returns[selected_stocks]
    expected_returns = selected_returns.mean() * TRADING_DAYS
    covariance_matrix = selected_returns.cov() * TRADING_DAYS

    optimal_weights, optimizer_success, optimizer_message = optimize_max_sharpe(
        expected_returns.values, covariance_matrix.values, annual_rf
    )

    if not optimizer_success:
        raise ValueError(optimizer_message)

    optimal_return, optimal_risk, optimal_sharpe = portfolio_performance(
        optimal_weights, expected_returns.values, covariance_matrix.values, annual_rf
    )

    if np.isnan(optimal_sharpe):
        raise ValueError("Portfolio risk is zero, so the Sharpe ratio cannot be calculated.")

    equal_weights = np.repeat(1 / len(selected_stocks), len(selected_stocks))
    equal_return, equal_risk, equal_sharpe = portfolio_performance(
        equal_weights, expected_returns.values, covariance_matrix.values, annual_rf
    )

    weights_series = pd.Series(optimal_weights, index=selected_stocks)

    # ---- Step 5: efficient frontier ----
    frontier = calculate_efficient_frontier(
        expected_returns.values, covariance_matrix.values, annual_rf
    )
    random_portfolios = generate_random_portfolios(
        expected_returns.values, covariance_matrix.values, annual_rf
    )

    # ---- Step 6: utility-based allocation ----
    risk_aversion_A = RISK_AVERSION_OPTIONS[risk_preference]

    allocation = calculate_utility_allocation(
        optimal_return, optimal_risk, annual_rf, risk_aversion_A
    )

    allocation_table, risky_investment, risk_free_investment = make_rupee_allocation(
        weights_series, allocation["Practical y"], investment_budget
    )

    # ---- Step 7: report ----
    market_sharpe = (
        (market_summary["Market Annual Return"] - annual_rf)
        / market_summary["Market Annual Volatility"]
    )

    insights = generate_key_insights(
        stock_table, selected_stocks, weights_series, optimal_sharpe,
        market_sharpe, market_column,
        allocation["Theoretical y"], allocation["Practical y"]
    )

    # Everything the website needs, stored in one dictionary
    return {
        "stock_notes": stock_notes, "market_notes": market_notes,
        "stock_columns": stock_columns, "market_column": market_column,
        "stock_report": stock_report, "market_report": market_report,
        "stock_prices": stock_prices, "market_data": market_data,
        "aligned_data": aligned_data, "daily_returns": daily_returns,
        "annual_rf": annual_rf, "risk_free_rate": risk_free_rate,
        "investment_budget": investment_budget,
        "number_of_stocks": int(number_of_stocks),
        "risk_preference": risk_preference, "risk_aversion_A": risk_aversion_A,
        "stock_table": stock_table, "market_summary": market_summary,
        "alpha_check": alpha_check, "selected_stocks": selected_stocks,
        "selected_returns": selected_returns,
        "expected_returns": expected_returns, "covariance_matrix": covariance_matrix,
        "weights_series": weights_series, "optimizer_message": optimizer_message,
        "optimal_return": optimal_return, "optimal_risk": optimal_risk,
        "optimal_sharpe": optimal_sharpe,
        "equal_return": equal_return, "equal_risk": equal_risk,
        "equal_sharpe": equal_sharpe,
        "frontier": frontier, "random_portfolios": random_portfolios,
        "allocation": allocation, "allocation_table": allocation_table,
        "risky_investment": risky_investment,
        "risk_free_investment": risk_free_investment,
        "market_sharpe": market_sharpe, "insights": insights,
    }


class SampleFile(io.BytesIO):
    """
    Opens a sample Excel file so it behaves exactly like an uploaded file
    (it has a .name and a .size). Inherits from io.BytesIO.
    """

    def __init__(self, path):
        with open(path, "rb") as file:
            super().__init__(file.read())
        self.name = os.path.basename(path)
        self.size = os.path.getsize(path)


def show_chart(fig, caption):
    """Display a Matplotlib chart with a short explanation underneath."""
    st.pyplot(fig)
    plt.close(fig)
    st.caption(caption)


def results_or_message(section_name):
    """Return the saved results, or show a message if no analysis has run yet."""

    results = st.session_state.get("results")

    if results is None:
        st.info(
            f"No results yet. Go to **Investor Dashboard**, upload your two files "
            f"and click **Analyse My Portfolio** to see {section_name}."
        )
        return None

    if st.session_state.get("inputs_changed"):
        st.warning(
            "Your inputs have changed since this analysis. Click "
            "**Analyse My Portfolio** again to update the results."
        )

    return results


# --------------------------------
# 4. HEADER
# --------------------------------

st.markdown("""
<div class="si-hero">
  <h1>SmartInvest</h1>
  <p>Upload historical prices for the stocks you are considering. SmartInvest
  ranks them on risk-adjusted performance, builds the most efficient portfolio
  and tells you how many rupees to put in each.</p>
  <div class="si-steps">
    <div class="si-step"><b>1</b>Screen stocks with CAPM and Jensen's alpha</div>
    <div class="si-step"><b>2</b>Optimize the mix for the best Sharpe ratio</div>
    <div class="si-step"><b>3</b>Split your budget by your risk aversion</div>
  </div>
</div>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state["results"] = None

tab_home, tab_dashboard, tab_screening, tab_risk, tab_optimize, tab_allocate, tab_report = st.tabs([
    "Home",
    "Investor Dashboard",
    "Stock Screening",
    "Risk-Return Analysis",
    "Portfolio Optimization",
    "Investment Allocation",
    "Investment Report",
])


# --------------------------------
# 5. HOME
# --------------------------------

with tab_home:

    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("The problem SmartInvest solves")
        st.write(
            "Retail investors often find it hard to compare stocks, understand "
            "market risk, build a diversified portfolio and decide how much of "
            "their money to keep in safe, risk-free investments."
        )
        st.write(
            "SmartInvest turns your historical price data into a clear, "
            "personalised investment plan using three established finance models: "
            "the Capital Asset Pricing Model (CAPM), Modern Portfolio Theory and "
            "utility-based asset allocation."
        )

        st.subheader("How to use it")
        st.markdown(
            "1. Open **Investor Dashboard**.\n"
            "2. Upload daily stock prices (CSV or Excel) and a market benchmark such as NIFTY 50.\n"
            "3. Enter your risk-free rate, budget, number of stocks and risk aversion.\n"
            "4. Click **Analyse My Portfolio**, then read each tab from left to right.\n\n"
            "No files yet? Switch on **Use sample data** in Investor Dashboard."
        )

    with right:
        st.subheader("Accepted file formats")
        st.markdown(
            "- **Yahoo Finance Excel workbook**: one sheet per stock; "
            "Adj Close prices are used.\n"
            "- **Simple table**: `Date | Stock 1 | Stock 2 | ...`\n"
            "- **Benchmark**: `Date | Index price`\n"
            "- Daily prices, at least one year of matching dates."
        )

        st.subheader("Important")
        st.write(
            "SmartInvest is an educational decision-support tool. All results are "
            "based on historical data and do not guarantee future returns."
        )


# --------------------------------
# 6. INVESTOR DASHBOARD (inputs + data checks)
# --------------------------------

with tab_dashboard:

    st.subheader("Upload your data")

    samples_available = os.path.exists(SAMPLE_STOCK_FILE) and os.path.exists(SAMPLE_MARKET_FILE)

    use_sample_data = False
    if samples_available:
        use_sample_data = st.toggle(
            "Use sample data (Dabur, Wipro, Suzlon, Welspun Living and NIFTY 50, 2021-2026)"
        )

    if use_sample_data:
        stock_file = SampleFile(SAMPLE_STOCK_FILE)
        market_file = SampleFile(SAMPLE_MARKET_FILE)
        st.info("Sample data selected. Turn the switch off to upload your own files.")
    else:
        upload1, upload2 = st.columns(2)

        with upload1:
            stock_file = st.file_uploader("Stock price data", type=["csv", "xlsx"])

        with upload2:
            market_file = st.file_uploader("Market benchmark data (e.g. NIFTY 50)",
                                           type=["csv", "xlsx"])

    st.subheader("Your investment settings")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        risk_free_rate = st.number_input("Annual risk-free rate (%)",
                                         min_value=0.0, max_value=30.0, value=4.0)
    with col2:
        investment_budget = st.number_input("Investment budget (₹)",
                                            min_value=1000, value=100000, step=1000)
    with col3:
        number_of_stocks = st.number_input("Number of stocks to select",
                                           min_value=1, max_value=MAX_STOCKS, value=3)
    with col4:
        risk_preference = st.selectbox("Risk aversion",
                                       list(RISK_AVERSION_OPTIONS.keys()), index=1)

    st.caption(
        "Risk aversion is how much you dislike risk. Higher risk aversion "
        "means a larger share of your budget goes to the risk-free asset."
    )

    # A "signature" of the current inputs, to detect changes after an analysis
    current_inputs = (
        stock_file.name if stock_file else None,
        stock_file.size if stock_file else None,
        market_file.name if market_file else None,
        market_file.size if market_file else None,
        risk_free_rate, investment_budget, int(number_of_stocks), risk_preference,
    )

    if st.button("Analyse My Portfolio", type="primary"):

        if stock_file is None or market_file is None:
            st.error("Please upload both files.")
        else:
            try:
                with st.spinner("Running the analysis..."):
                    st.session_state["results"] = run_analysis(
                        stock_file, market_file, risk_free_rate,
                        investment_budget, number_of_stocks, risk_preference
                    )
                st.session_state["analysed_inputs"] = current_inputs
            except Exception as error:
                st.session_state["results"] = None
                st.error("There was a problem with your files.")
                st.write(str(error))

    st.session_state["inputs_changed"] = (
        st.session_state.get("results") is not None
        and st.session_state.get("analysed_inputs") != current_inputs
    )

    results = st.session_state.get("results")

    if results is not None:

        if st.session_state["inputs_changed"]:
            st.warning("Your inputs have changed. Click **Analyse My Portfolio** again "
                       "to update the results.")
        else:
            st.success("Analysis complete. Open the next tabs to see your results.")

        for note in results["stock_notes"]:
            st.info("Stock file: " + note)
        for note in results["market_notes"]:
            st.info("Benchmark file: " + note)

        st.subheader("Data overview")

        aligned_data = results["aligned_data"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stocks uploaded", len(results["stock_columns"]))
        m2.metric("Benchmark", results["market_column"])
        m3.metric("Matching trading days", len(aligned_data))
        m4.metric("Period", f"{aligned_data['Date'].min():%b %Y} - "
                            f"{aligned_data['Date'].max():%b %Y}")

        st.write("**Stocks detected:** " + ", ".join(results["stock_columns"]))

        with st.expander("Data cleaning report"):
            st.dataframe(pd.DataFrame([results["stock_report"], results["market_report"]]),
                         width="stretch")
            days_removed = (max(len(results["stock_prices"]), len(results["market_data"]))
                            - len(aligned_data))
            st.caption(f"{days_removed} date(s) were dropped because they were not present "
                       "in both files. No prices were filled in or invented.")

        with st.expander("Aligned price data (first 10 rows)"):
            st.dataframe(aligned_data.head(10), width="stretch")


# --------------------------------
# 7. STOCK SCREENING
# --------------------------------

with tab_screening:

    results = results_or_message("the stock ranking")

    if results is not None:

        stock_table = results["stock_table"]
        market_summary = results["market_summary"]
        market_column = results["market_column"]
        selected_stocks = results["selected_stocks"]

        st.subheader("Stock ranking by Jensen's alpha")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Daily returns used", len(results["daily_returns"]))
        k2.metric(f"{market_column} annual return",
                  f"{market_summary['Market Annual Return'] * 100:.2f}%")
        k3.metric(f"{market_column} annual volatility",
                  f"{market_summary['Market Annual Volatility'] * 100:.2f}%")
        k4.metric("Risk-free rate", f"{results['risk_free_rate']:.2f}%")

        st.dataframe(make_display_table(stock_table), width="stretch")

        if results["alpha_check"]:
            st.caption("Check passed: for every stock, the CAPM regression intercept x 252 "
                       "equals Annual Return - CAPM Required Return.")
        else:
            st.warning("Alpha consistency check did not pass.")

        st.write(f"**Selected stocks (top {results['number_of_stocks']}):** "
                 + ", ".join(selected_stocks))

        # ---- Negative alpha safeguard ----
        negative_count = int((stock_table["Jensen's Alpha"] < 0).sum())
        total_count = len(stock_table)
        selected_negative = [stock for stock in selected_stocks
                             if stock_table.loc[stock, "Jensen's Alpha"] < 0]

        if negative_count == total_count:
            st.warning(
                "All uploaded stocks have negative historical Jensen's alpha. They earned "
                "less than their CAPM-required return during this period. The ranking shows "
                "the least negative stocks first. You may continue with them, or upload "
                "different stocks."
            )
        elif negative_count > total_count / 2:
            st.warning(f"{negative_count} of {total_count} stocks have negative historical "
                       "Jensen's alpha (they underperformed their CAPM-required return).")

        if len(selected_negative) > 0 and negative_count != total_count:
            st.warning("These selected stocks have negative historical alpha: "
                       + ", ".join(selected_negative)
                       + ". Consider selecting fewer stocks or uploading others.")

        chart1, chart2 = st.columns(2)
        with chart1:
            show_chart(plot_return_risk_chart(stock_table),
                       "Dark bars show historical annual return; light bars show "
                       "volatility (how much the price swung, i.e. risk).")
        with chart2:
            show_chart(plot_alpha_chart(stock_table, selected_stocks),
                       "Positive alpha: the stock beat the return CAPM required for its "
                       "beta. Negative alpha: it fell short.")

        best_stock = stock_table.index[0]
        best_alpha = stock_table.loc[best_stock, "Jensen's Alpha"] * 100
        best_beta = stock_table.loc[best_stock, "Beta"]

        st.subheader("What this means")
        st.write(
            f"**{best_stock}** ranks first with a historical Jensen's alpha of "
            f"**{best_alpha:.2f}% per year** and a beta of **{best_beta:.2f}**. "
            "A beta above 1 means the stock moved more than the market; below 1 means "
            "it moved less."
        )
        st.info("These figures are historical estimates for the uploaded period. "
                "Past alpha does not guarantee future performance.")


# --------------------------------
# 8. RISK-RETURN ANALYSIS (SML)
# --------------------------------

with tab_risk:

    results = results_or_message("the Security Market Line")

    if results is not None:

        stock_table = results["stock_table"]

        st.subheader("Security Market Line and risk-return")

        chart3, chart4 = st.columns(2)
        with chart3:
            show_chart(plot_sml_chart(stock_table, results["market_summary"],
                                      results["annual_rf"], results["market_column"]),
                       "The line shows the return CAPM requires for each level of beta. "
                       "The dotted gap between a stock and the line is its Jensen's alpha.")
        with chart4:
            show_chart(plot_risk_return_scatter(stock_table, results["market_summary"],
                                                results["market_column"],
                                                results["selected_stocks"]),
                       "Stocks further up earned more; stocks further right were more "
                       "volatile. The market point is shown for comparison.")

        above_sml = stock_table[stock_table["Jensen's Alpha"] >= 0].index.tolist()
        below_sml = stock_table[stock_table["Jensen's Alpha"] < 0].index.tolist()

        st.subheader("What this means")
        st.write("**Above the SML (positive historical alpha):** "
                 + (", ".join(above_sml) if above_sml else "none"))
        st.write("**Below the SML (negative historical alpha):** "
                 + (", ".join(below_sml) if below_sml else "none"))
        st.write("Stocks above the line earned more than CAPM required for their beta "
                 "during this period; stocks below earned less. The SML uses beta (market "
                 "risk only), while the risk-return chart uses total volatility.")
        st.info("The Security Market Line (SML) is not the Capital Allocation Line (CAL). "
                "The CAL is in the Portfolio Optimization tab.")


# --------------------------------
# 9. PORTFOLIO OPTIMIZATION (Max Sharpe, Frontier, CAL)
# --------------------------------

with tab_optimize:

    results = results_or_message("the optimized portfolio")

    if results is not None:

        weights_series = results["weights_series"]
        selected_returns = results["selected_returns"]
        optimal_sharpe = results["optimal_sharpe"]

        st.subheader("Maximum Sharpe ratio portfolio")
        st.success(results["optimizer_message"])

        p1, p2, p3 = st.columns(3)
        p1.metric("Expected annual return", f"{results['optimal_return'] * 100:.2f}%")
        p2.metric("Annual risk (volatility)", f"{results['optimal_risk'] * 100:.2f}%")
        p3.metric("Maximum Sharpe ratio", f"{optimal_sharpe:.3f}")

        weights_table = pd.DataFrame({
            "Optimal Weight (%)": (weights_series * 100).round(2),
            "Annual Return (%)": (results["expected_returns"] * 100).round(2),
            "Annual Volatility (%)": (selected_returns.std() * np.sqrt(TRADING_DAYS) * 100).round(2),
        }).sort_values("Optimal Weight (%)", ascending=False)

        chart5, table5 = st.columns(2)
        with chart5:
            show_chart(plot_weights_chart(weights_series),
                       "Share of the risky portfolio given to each selected stock. "
                       "Weights add up to 100% and none is negative (no short selling).")
        with table5:
            st.write("**Optimal weights**")
            st.dataframe(weights_table, width="stretch")

            st.write("**Compared with an equal-weight portfolio**")
            st.dataframe(pd.DataFrame({
                "Portfolio": ["Equal weights", "Maximum Sharpe"],
                "Return (%)": [round(results["equal_return"] * 100, 2),
                               round(results["optimal_return"] * 100, 2)],
                "Risk (%)": [round(results["equal_risk"] * 100, 2),
                             round(results["optimal_risk"] * 100, 2)],
                "Sharpe Ratio": [round(results["equal_sharpe"], 3), round(optimal_sharpe, 3)],
            }).set_index("Portfolio"), width="stretch")

        with st.expander("Annualised covariance matrix"):
            st.dataframe(results["covariance_matrix"].round(4), width="stretch")
            st.caption("Diagonal = each stock's variance. Off-diagonal = how two stocks "
                       "move together. Portfolio risk = square root of (w' x Covariance x w).")

        zero_weight_stocks = [stock for stock in results["selected_stocks"]
                              if weights_series[stock] < 0.0001]

        st.write(f"The optimizer reaches a Sharpe ratio of **{optimal_sharpe:.3f}**, "
                 f"compared with **{results['equal_sharpe']:.3f}** for equal weights.")
        if zero_weight_stocks:
            st.write("**Weight 0%:** " + ", ".join(zero_weight_stocks)
                     + ". Adding these stocks did not improve risk-adjusted return.")
        if optimal_sharpe < 0:
            st.warning("Even the best portfolio earned less than the risk-free rate "
                       "(negative Sharpe ratio).")

        st.divider()
        st.subheader("Efficient Frontier and Capital Allocation Line")

        stock_points = pd.DataFrame({
            "Return": results["expected_returns"],
            "Risk": selected_returns.std() * np.sqrt(TRADING_DAYS),
        })

        chart6, chart7 = st.columns(2)
        with chart6:
            show_chart(plot_efficient_frontier(results["frontier"],
                                               results["random_portfolios"], stock_points,
                                               results["optimal_risk"],
                                               results["optimal_return"]),
                       "Grey dots are possible mixes of the selected stocks. The blue line "
                       "(Efficient Frontier) gives the lowest risk for each level of return. "
                       "The star is the maximum Sharpe portfolio.")
        with chart7:
            show_chart(plot_capital_allocation_line(results["frontier"], results["annual_rf"],
                                                    results["optimal_risk"],
                                                    results["optimal_return"],
                                                    optimal_sharpe),
                       "The CAL joins the risk-free asset to the star. Points on the solid "
                       "line are mixes of the two. The dashed part would need borrowing, "
                       "which SmartInvest does not allow.")

        frontier = results["frontier"]
        if len(frontier) > 1:
            f1, f2 = st.columns(2)
            f1.metric("Lowest-risk portfolio: risk", f"{frontier.iloc[0]['Risk'] * 100:.2f}%")
            f2.metric("Lowest-risk portfolio: return", f"{frontier.iloc[0]['Return'] * 100:.2f}%")

        st.write(
            "Portfolios on the Efficient Frontier give the best return for their level of "
            "risk. The Capital Allocation Line touches the frontier at the maximum Sharpe "
            f"portfolio; its slope ({optimal_sharpe:.3f}) is the extra return earned for "
            "each 1% of extra risk. Every investor holds this same risky portfolio; risk "
            "aversion decides **how much** of the budget goes into it."
        )
        st.info("Weights, frontier and CAL use historical estimates. Future returns and "
                "risks may differ.")


# --------------------------------
# 10. INVESTMENT ALLOCATION
# --------------------------------

with tab_allocate:

    results = results_or_message("your rupee allocation")

    if results is not None:

        allocation = results["allocation"]
        allocation_table = results["allocation_table"]
        theoretical_y = allocation["Theoretical y"]
        practical_y = allocation["Practical y"]
        risky_investment = results["risky_investment"]
        risk_free_investment = results["risk_free_investment"]
        investment_budget = results["investment_budget"]

        st.subheader("Your investment allocation")
        st.write(f"**Risk aversion:** {results['risk_preference']}")

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Theoretical risky share (y*)", f"{theoretical_y * 100:.1f}%")
        a2.metric("Practical risky share (y)", f"{practical_y * 100:.1f}%")
        a3.metric("Risky investment", f"₹{risky_investment:,.0f}")
        a4.metric("Risk-free investment", f"₹{risk_free_investment:,.0f}")

        st.caption(
            "y* = (E(Rp) - Rf) / (A x Variance of risky portfolio)  =  "
            f"({results['optimal_return'] * 100:.2f}% - {results['annual_rf'] * 100:.2f}%) / "
            f"({results['risk_aversion_A']} x {results['optimal_risk'] ** 2:.4f})"
        )

        if theoretical_y > 1 or theoretical_y < 0:
            st.warning(allocation["Adjustment Note"])
        else:
            st.info(allocation["Adjustment Note"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Your expected annual return", f"{allocation['Complete Return'] * 100:.2f}%")
        c2.metric("Your annual risk", f"{allocation['Complete Risk'] * 100:.2f}%")
        c3.metric("Utility score (U)", f"{allocation['Utility']:.4f}")

        chart9, chart10 = st.columns([1, 1.4])
        with chart9:
            show_chart(plot_risky_vs_riskfree(risky_investment, risk_free_investment),
                       "How your budget is split between the optimized risky portfolio "
                       "and the risk-free asset.")
        with chart10:
            show_chart(plot_rupee_allocation(allocation_table),
                       "Exact rupee amount for each stock and the risk-free asset.")

        st.subheader("Final allocation table")
        st.dataframe(allocation_table, width="stretch")
        st.write(f"**Total allocated:** ₹{allocation_table['Amount (Rs)'].sum():,.2f} "
                 f"(budget: ₹{investment_budget:,.2f})")

        with st.expander("Compare all three risk-aversion levels"):
            comparison_rows = []
            for label, A_value in RISK_AVERSION_OPTIONS.items():
                result = calculate_utility_allocation(
                    results["optimal_return"], results["optimal_risk"],
                    results["annual_rf"], A_value
                )
                comparison_rows.append({
                    "Risk Aversion": label,
                    "Theoretical y* (%)": round(result["Theoretical y"] * 100, 1),
                    "Practical y (%)": round(result["Practical y"] * 100, 1),
                    "Risky Amount (Rs)": round(result["Practical y"] * investment_budget, 0),
                    "Risk-Free Amount (Rs)": round((1 - result["Practical y"]) * investment_budget, 0),
                })
            st.dataframe(pd.DataFrame(comparison_rows).set_index("Risk Aversion"),
                         width="stretch")
            st.caption("Higher A (more risk aversion) means a smaller share in the "
                       "risky portfolio.")

        invested_stocks = [stock for stock in results["selected_stocks"]
                           if allocation_table.loc[stock, "Amount (Rs)"] > 0]

        st.subheader("What this means")
        st.write(
            f"With risk aversion A = {results['risk_aversion_A']}, SmartInvest puts "
            f"**₹{risky_investment:,.0f} ({practical_y * 100:.1f}%)** in the optimized "
            f"risky portfolio and **₹{risk_free_investment:,.0f}** in the risk-free asset."
        )
        if invested_stocks:
            st.write("**Stocks receiving money:** " + ", ".join(invested_stocks))
        st.info("This allocation is an educational estimate based on historical data. "
                "It is not investment advice and does not guarantee future returns.")


# --------------------------------
# 11. INVESTMENT REPORT
# --------------------------------

with tab_report:

    results = results_or_message("the investment report")

    if results is not None:

        allocation = results["allocation"]
        aligned_data = results["aligned_data"]
        market_column = results["market_column"]

        st.subheader("Investment report")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Portfolio Sharpe", f"{results['optimal_sharpe']:.3f}")
        r2.metric(f"{market_column} Sharpe", f"{results['market_sharpe']:.3f}")
        r3.metric("Risky share of budget", f"{allocation['Practical y'] * 100:.1f}%")
        r4.metric("Expected annual return", f"{allocation['Complete Return'] * 100:.2f}%")

        st.subheader("Executive summary: findings and actions")
        for number, insight in enumerate(results["insights"], start=1):
            st.markdown(f"**{number}.** {insight}")

        st.subheader("Limitations")
        st.markdown(
            "- Results use historical data; past performance does not guarantee future returns.\n"
            "- The benchmark is a price index without dividends, which can slightly overstate alpha.\n"
            "- Mean returns and covariances are estimates and can change over time.\n"
            "- SmartInvest is an educational tool, not investment advice."
        )

        # ---- Downloads (created in memory, nothing is saved on the server) ----
        st.subheader("Download your results")

        ranking_download = make_display_table(results["stock_table"])

        report_details = {
            "Analysis period": (f"{aligned_data['Date'].min():%d %b %Y} to "
                                f"{aligned_data['Date'].max():%d %b %Y}"),
            "Benchmark": market_column,
            "Risk-free rate": f"{results['risk_free_rate']:.2f}%",
            "Investment budget": f"Rs {results['investment_budget']:,.2f}",
            "Risk aversion": results["risk_preference"],
            "Stocks uploaded": len(results["stock_columns"]),
            "Stocks selected": ", ".join(results["selected_stocks"]),
            "Portfolio expected return": f"{results['optimal_return'] * 100:.2f}%",
            "Portfolio risk": f"{results['optimal_risk'] * 100:.2f}%",
            "Portfolio Sharpe ratio": f"{results['optimal_sharpe']:.3f}",
            "Theoretical risky share (y*)": f"{allocation['Theoretical y'] * 100:.1f}%",
            "Practical risky share (y)": f"{allocation['Practical y'] * 100:.1f}%",
        }

        report_text = build_report_text(report_details, ranking_download,
                                        results["allocation_table"], results["insights"])

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("Download stock ranking (CSV)", data=ranking_download.to_csv(),
                               file_name="smartinvest_stock_ranking.csv", mime="text/csv",
                               on_click="ignore")
        with d2:
            st.download_button("Download allocation (CSV)",
                               data=results["allocation_table"].to_csv(),
                               file_name="smartinvest_allocation.csv", mime="text/csv",
                               on_click="ignore")
        with d3:
            st.download_button("Download full report (TXT)", data=report_text,
                               file_name="smartinvest_report.txt", mime="text/plain",
                               on_click="ignore")

        with st.expander("Preview full report"):
            st.text(report_text)
