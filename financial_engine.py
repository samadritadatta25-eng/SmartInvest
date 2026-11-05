# ==================================================
# SMARTINVEST - FINANCIAL ENGINE
# CAPM screening, portfolio optimization, efficient frontier,
# utility-based allocation and the investment report.
# ==================================================

import numpy as np
import pandas as pd

from scipy.optimize import minimize

TRADING_DAYS = 252       # Trading days per year (annualisation)


# --------------------------------
# STEP 2: BETA, CAPM AND JENSEN'S ALPHA
# --------------------------------

def calculate_stock_statistics(daily_returns, stock_columns,
                               market_column, annual_rf):
    """
    For every stock calculate:
    annual return, volatility, beta, CAPM-required return
    and Jensen's alpha. Returns a ranked table.
    """

    # Annual risk-free rate -> daily rate (simple conversion)
    daily_rf = annual_rf / TRADING_DAYS

    # Market figures
    market_returns = daily_returns[market_column]
    market_excess = market_returns - daily_rf
    market_annual_return = market_returns.mean() * TRADING_DAYS
    market_volatility = market_returns.std() * np.sqrt(TRADING_DAYS)
    if market_excess.var() == 0:
        raise ValueError("Benchmark prices do not change, so beta cannot be calculated.")

    results = []   # list of dictionaries, one per stock

    for stock in stock_columns:

        stock_returns = daily_returns[stock]
        stock_excess = stock_returns - daily_rf

        # CAPM regression: (Ri - Rf) = alpha + beta (Rm - Rf)
        # Slope (beta) = Cov(stock excess, market excess) / Var(market excess)
        beta = stock_excess.cov(market_excess) / market_excess.var()

        # Intercept (daily alpha) = mean(stock excess) - beta x mean(market excess)
        daily_alpha = stock_excess.mean() - beta * market_excess.mean()

        # Annualised return and volatility
        annual_return = stock_returns.mean() * TRADING_DAYS
        annual_volatility = stock_returns.std() * np.sqrt(TRADING_DAYS)

        # CAPM-required return and Jensen's alpha
        capm_return = annual_rf + beta * (market_annual_return - annual_rf)
        jensen_alpha = annual_return - capm_return

        results.append({
            "Stock": stock,
            "Annual Return": annual_return,
            "Annual Volatility": annual_volatility,
            "Beta": beta,
            "CAPM Required Return": capm_return,
            "Jensen's Alpha": jensen_alpha,
            "Regression Alpha x 252": daily_alpha * TRADING_DAYS,
        })

    stock_table = pd.DataFrame(results).set_index("Stock")

    # Rank stocks: highest Jensen's alpha first
    stock_table = stock_table.sort_values("Jensen's Alpha", ascending=False)
    stock_table.insert(0, "Rank", range(1, len(stock_table) + 1))

    market_summary = {
        "Market Annual Return": market_annual_return,
        "Market Annual Volatility": market_volatility,
    }

    return stock_table, market_summary


def make_display_table(stock_table):
    """Convert decimals to percentages for an easy-to-read table."""

    display_table = pd.DataFrame(index=stock_table.index)
    display_table["Rank"] = stock_table["Rank"]
    display_table["Annual Return (%)"] = (stock_table["Annual Return"] * 100).round(2)
    display_table["Annual Volatility (%)"] = (stock_table["Annual Volatility"] * 100).round(2)
    display_table["Beta"] = stock_table["Beta"].round(3)
    display_table["CAPM Required Return (%)"] = (stock_table["CAPM Required Return"] * 100).round(2)
    display_table["Jensen's Alpha (%)"] = (stock_table["Jensen's Alpha"] * 100).round(2)

    return display_table


# --------------------------------
# STEP 4: MAXIMUM SHARPE RATIO PORTFOLIO
# --------------------------------

def portfolio_performance(weights, expected_returns, covariance_matrix, annual_rf):
    """
    Portfolio return   E(Rp) = sum of (weight x expected return)
    Portfolio risk     Sigma_p = square root of (w' x Covariance x w)
    Sharpe ratio       = (E(Rp) - Rf) / Sigma_p
    """

    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    portfolio_risk = np.sqrt(portfolio_variance)

    if portfolio_risk == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (portfolio_return - annual_rf) / portfolio_risk

    return portfolio_return, portfolio_risk, sharpe_ratio


def optimize_max_sharpe(expected_returns, covariance_matrix, annual_rf):
    """
    Find long-only weights that give the highest Sharpe ratio.
    Constraints: weights add up to 1, every weight between 0 and 1.
    Returns: weights, success (True/False), message
    """

    number_of_assets = len(expected_returns)

    # Only one stock: nothing to optimize
    if number_of_assets == 1:
        return np.array([1.0]), True, "Only one stock selected, so it receives 100%."

    def negative_sharpe(weights):
        portfolio_return, portfolio_risk, sharpe = portfolio_performance(
            weights, expected_returns, covariance_matrix, annual_rf
        )
        if portfolio_risk <= 0:
            return 1e6
        # minimize() finds the lowest value, so we minimize -Sharpe
        return -sharpe

    initial_weights = np.repeat(1 / number_of_assets, number_of_assets)

    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
    bounds = [(0, 1)] * number_of_assets    # long-only, no short selling

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    if not result.success:
        return None, False, "Optimizer did not converge: " + str(result.message)

    # Remove tiny rounding errors (e.g. -0.0000001) and make weights sum to 1
    weights = np.clip(result.x, 0, 1)
    weights = weights / weights.sum()

    return weights, True, "Optimization successful."


# --------------------------------
# STEP 5: EFFICIENT FRONTIER AND CAL
# --------------------------------

def minimum_risk_portfolio(expected_returns, covariance_matrix, target_return=None):
    """
    Find the long-only portfolio with the LOWEST risk.
    If target_return is given, the portfolio must also earn that return.
    Returns: weights (or None if not possible)
    """

    number_of_assets = len(expected_returns)

    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(covariance_matrix, weights))

    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]

    if target_return is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda weights: np.dot(weights, expected_returns) - target_return
        })

    initial_weights = np.repeat(1 / number_of_assets, number_of_assets)
    bounds = [(0, 1)] * number_of_assets

    result = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    if not result.success:
        return None

    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


def calculate_efficient_frontier(expected_returns, covariance_matrix,
                                 annual_rf, number_of_points=40):
    """
    Efficient Frontier: for each target return, find the lowest-risk portfolio.
    Targets go from the minimum-risk portfolio's return up to the
    highest single-stock return.
    """

    frontier_points = []    # list of dictionaries

    # Only one stock: the "frontier" is just that one stock
    if len(expected_returns) == 1:
        point_return, point_risk, point_sharpe = portfolio_performance(
            np.array([1.0]), expected_returns, covariance_matrix, annual_rf
        )
        return pd.DataFrame([{"Return": point_return, "Risk": point_risk,
                              "Sharpe Ratio": point_sharpe}])

    # Global minimum-risk portfolio (no return target)
    min_risk_weights = minimum_risk_portfolio(expected_returns, covariance_matrix)

    if min_risk_weights is None:
        return pd.DataFrame(columns=["Return", "Risk", "Sharpe Ratio"])

    lowest_target = np.dot(min_risk_weights, expected_returns)
    highest_target = np.max(expected_returns)

    target_returns = np.linspace(lowest_target, highest_target, number_of_points)

    for target in target_returns:

        weights = minimum_risk_portfolio(expected_returns, covariance_matrix, target)

        if weights is None:
            continue    # skip targets the optimizer could not reach

        point_return, point_risk, point_sharpe = portfolio_performance(
            weights, expected_returns, covariance_matrix, annual_rf
        )

        frontier_points.append({
            "Return": point_return,
            "Risk": point_risk,
            "Sharpe Ratio": point_sharpe,
        })

    return pd.DataFrame(frontier_points)


def generate_random_portfolios(expected_returns, covariance_matrix,
                               annual_rf, number_of_portfolios=3000):
    """Random long-only portfolios, used only to show the feasible region."""

    number_of_assets = len(expected_returns)
    random_generator = np.random.default_rng(42)    # fixed seed = same chart every time

    results = []

    for i in range(number_of_portfolios):
        weights = random_generator.random(number_of_assets)
        weights = weights / weights.sum()

        point_return, point_risk, point_sharpe = portfolio_performance(
            weights, expected_returns, covariance_matrix, annual_rf
        )
        results.append([point_return, point_risk, point_sharpe])

    return pd.DataFrame(results, columns=["Return", "Risk", "Sharpe Ratio"])


# --------------------------------
# STEP 6: UTILITY-BASED ALLOCATION
# --------------------------------

def calculate_utility_allocation(optimal_return, optimal_risk, annual_rf, risk_aversion_A):
    """
    Utility:  U = E(Rc) - 0.5 x A x Sigma_c^2

    Best share in the risky portfolio:
        y* = (E(Rp) - Rf) / (A x Sigma_p^2)

    SmartInvest allows no borrowing and no short selling, so:
        Practical y = min(1, max(0, y*))
    """

    risky_variance = optimal_risk ** 2

    theoretical_y = (optimal_return - annual_rf) / (risk_aversion_A * risky_variance)

    # Keep y between 0% and 100%
    practical_y = min(1, max(0, theoretical_y))

    if theoretical_y > 1:
        adjustment_note = (
            f"The theoretical allocation is {theoretical_y * 100:.1f}%. Investing more "
            "than 100% would need borrowing, which SmartInvest does not allow, "
            "so it is capped at 100%."
        )
    elif theoretical_y < 0:
        adjustment_note = (
            f"The theoretical allocation is {theoretical_y * 100:.1f}%. A negative "
            "value would mean short selling the portfolio, which SmartInvest does "
            "not allow, so it is set to 0%."
        )
    else:
        adjustment_note = "No adjustment was needed: the theoretical allocation is between 0% and 100%."

    # Complete portfolio (risk-free asset + risky portfolio)
    complete_return = annual_rf + practical_y * (optimal_return - annual_rf)
    complete_risk = practical_y * optimal_risk
    utility = complete_return - 0.5 * risk_aversion_A * complete_risk ** 2

    return {
        "Theoretical y": theoretical_y,
        "Practical y": practical_y,
        "Adjustment Note": adjustment_note,
        "Complete Return": complete_return,
        "Complete Risk": complete_risk,
        "Utility": utility,
    }


def make_rupee_allocation(weights_series, practical_y, investment_budget):
    """
    Risky Investment     = Practical y x Budget
    Risk-Free Investment = (1 - Practical y) x Budget
    Amount in Stock i    = Risky Investment x Optimal Weight i
    """

    risky_investment = practical_y * investment_budget

    rows = []    # list of dictionaries

    for stock, weight in weights_series.items():
        rows.append({
            "Investment": stock,
            "Type": "Stock",
            "Share of Budget (%)": round(practical_y * weight * 100, 2),
            "Amount (Rs)": round(risky_investment * weight, 2),
        })

    # Risk-free amount = whatever is left, so the total equals the budget exactly
    stock_total = sum(row["Amount (Rs)"] for row in rows)
    risk_free_investment = round(investment_budget - stock_total, 2)

    rows.append({
        "Investment": "Risk-Free Asset",
        "Type": "Risk-free",
        "Share of Budget (%)": round((1 - practical_y) * 100, 2),
        "Amount (Rs)": risk_free_investment,
    })

    allocation_table = pd.DataFrame(rows).set_index("Investment")

    return allocation_table, risky_investment, risk_free_investment


# --------------------------------
# STEP 7: INVESTMENT REPORT
# --------------------------------

def generate_key_insights(stock_table, selected_stocks, weights_series,
                          optimal_sharpe, market_sharpe, market_column,
                          theoretical_y, practical_y):
    """Create plain-language findings and actions for management."""

    insights = []    # list of sentences

    # 1. Best stock from screening
    best_stock = stock_table.index[0]
    best_alpha = stock_table.loc[best_stock, "Jensen's Alpha"] * 100
    insights.append(
        f"{best_stock} had the highest historical Jensen's alpha ({best_alpha:.2f}% per year), "
        "meaning it beat its CAPM-required return by the widest margin."
    )

    # 2. Stocks with negative alpha
    negative_stocks = stock_table[stock_table["Jensen's Alpha"] < 0].index.tolist()
    if len(negative_stocks) > 0:
        insights.append(
            "Underperformed CAPM (negative alpha): " + ", ".join(negative_stocks)
            + ". These are weak candidates unless there is a separate reason to hold them."
        )

    # 3. Optimized portfolio vs the market
    if optimal_sharpe > market_sharpe:
        insights.append(
            f"The optimized portfolio's Sharpe ratio ({optimal_sharpe:.3f}) is higher than "
            f"the market's ({market_column}: {market_sharpe:.3f}), so it gave more return "
            "per unit of risk during this period."
        )
    else:
        insights.append(
            f"The optimized portfolio's Sharpe ratio ({optimal_sharpe:.3f}) is not higher than "
            f"the market's ({market_column}: {market_sharpe:.3f}). A simple index investment "
            "would have been at least as efficient."
        )

    # 4. Concentration risk
    largest_stock = weights_series.idxmax()
    largest_weight = weights_series.max()
    if largest_weight > 0.6 and len(weights_series) > 1:
        insights.append(
            f"The portfolio is concentrated: {largest_stock} makes up {largest_weight * 100:.1f}% "
            "of the risky portfolio. Results depend heavily on one company, so the investor "
            "should consider adding more stocks or setting a maximum weight per stock."
        )

    # 5. Stocks the optimizer left out
    zero_stocks = [stock for stock in selected_stocks if weights_series[stock] < 0.0001]
    if len(zero_stocks) > 0:
        insights.append(
            "Selected but given 0% weight: " + ", ".join(zero_stocks)
            + ". They did not improve risk-adjusted return."
        )

    # 6. Allocation adjustment
    if theoretical_y > 1:
        insights.append(
            f"The investor's ideal risky share was {theoretical_y * 100:.1f}%, above 100%. "
            "Without borrowing, the whole budget goes into the risky portfolio."
        )
    elif theoretical_y < 0:
        insights.append(
            "The risky portfolio did not beat the risk-free rate, so the whole budget "
            "stays in the risk-free asset."
        )
    else:
        insights.append(
            f"{practical_y * 100:.1f}% of the budget goes to the risky portfolio and the rest "
            "to the risk-free asset, matching the investor's risk aversion."
        )

    return insights


def build_report_text(report_details, ranking_table, allocation_table, insights):
    """Write the full investment report as plain text (for download)."""

    lines = []    # each item is one line of the report

    lines.append("SMARTINVEST - INVESTMENT REPORT")
    lines.append("=" * 50)
    lines.append("")

    for label, value in report_details.items():
        lines.append(f"{label}: {value}")

    lines.append("")
    lines.append("BUSINESS PROBLEM")
    lines.append("-" * 50)
    lines.append("Retail investors find it hard to compare stocks, judge market risk,")
    lines.append("build a diversified portfolio and decide how much to keep risk-free.")
    lines.append("SmartInvest screens stocks with CAPM, optimizes a portfolio with")
    lines.append("Modern Portfolio Theory and sizes the investment using risk aversion.")

    lines.append("")
    lines.append("1. STOCK RANKING BY JENSEN'S ALPHA")
    lines.append("-" * 50)
    lines.append(ranking_table.to_string())

    lines.append("")
    lines.append("2. FINAL ALLOCATION")
    lines.append("-" * 50)
    lines.append(allocation_table.to_string())

    lines.append("")
    lines.append("3. KEY FINDINGS AND ACTIONS")
    lines.append("-" * 50)
    for number, insight in enumerate(insights, start=1):
        lines.append(f"{number}. {insight}")

    lines.append("")
    lines.append("4. LIMITATIONS")
    lines.append("-" * 50)
    lines.append("- Results use historical data; past performance does not guarantee future returns.")
    lines.append("- The benchmark is a price index without dividends, which can slightly overstate alpha.")
    lines.append("- Mean returns and covariances are estimates and can change over time.")
    lines.append("- This is an educational tool, not investment advice.")

    return "\n".join(lines)
