# ==================================================
# SMARTINVEST - CHARTS
# All Matplotlib charts used on the website (10 charts).
# ==================================================

import numpy as np
import matplotlib.pyplot as plt

# ---- SmartInvest colour palette ----
NAVY = "#0F2440"        # headings, lines
BLUE = "#2A5DB0"        # main accent, selected items
LIGHT_BLUE = "#A9BEDC"  # secondary bars
STEEL = "#6E87A8"       # neutral markers
GREY = "#C3CBD6"        # items not selected
CLOUD = "#D5DEEA"       # background portfolio cloud
GAIN = "#1F7A4D"        # positive alpha
LOSS = "#B23A3A"        # negative alpha
GOLD = "#C9A227"        # maximum Sharpe portfolio

# ---- One consistent chart style ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": NAVY,
    "axes.labelcolor": NAVY,
    "axes.edgecolor": "#9AA8BA",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#3B4D66",
    "ytick.color": "#3B4D66",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def plot_return_risk_chart(stock_table):
    """Chart 1: annual return and annual volatility of each stock."""

    stocks = list(stock_table.index)
    returns = stock_table["Annual Return"] * 100
    volatility = stock_table["Annual Volatility"] * 100

    positions = np.arange(len(stocks))
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(positions - bar_width / 2, returns, bar_width,
           label="Annual Return (%)", color=BLUE)
    ax.bar(positions + bar_width / 2, volatility, bar_width,
           label="Annual Volatility (%)", color=LIGHT_BLUE)

    ax.axhline(0, color=NAVY, linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(stocks)
    ax.set_ylabel("Percent per year")
    ax.set_title("Historical Annual Return vs Volatility")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return fig


def plot_alpha_chart(stock_table, selected_stocks):
    """Chart 2: Jensen's alpha ranking; selected stocks highlighted."""

    stocks = list(stock_table.index)
    alpha_values = stock_table["Jensen's Alpha"] * 100

    colors = []
    for stock in stocks:
        if stock in selected_stocks:
            colors.append(BLUE)    # selected: dark blue
        else:
            colors.append(GREY)    # not selected: grey

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(stocks, alpha_values, color=colors)

    # Value labels on the bars
    for bar, value in zip(bars, alpha_values):
        if value >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2, value,
                    f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, value,
                    f"{value:.2f}%", ha="center", va="top", fontsize=9)

    ax.axhline(0, color=NAVY, linewidth=0.8, linestyle="--")
    ax.margins(y=0.15)    # extra space so value labels are not cut off
    ax.set_ylabel("Jensen's Alpha (% per year)")
    ax.set_title("Jensen's Alpha Ranking (dark blue = selected stocks)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return fig


def plot_sml_chart(stock_table, market_summary, annual_rf, market_column):
    """
    Chart 3: Security Market Line.
    SML: Required Return = Rf + Beta x (Market Return - Rf)
    Stocks above the line have positive alpha; below have negative alpha.
    """

    market_return = market_summary["Market Annual Return"]

    # Beta values for drawing the line (from 0 up to a bit above the largest beta)
    largest_beta = max(stock_table["Beta"].max(), 1.0)
    beta_range = np.linspace(0, largest_beta * 1.2, 100)
    sml_returns = annual_rf + beta_range * (market_return - annual_rf)

    fig, ax = plt.subplots(figsize=(9, 5))
    # The SML
    ax.plot(beta_range, sml_returns * 100, color=NAVY,
            linewidth=2, label="Security Market Line")

    # Each stock: actual return vs required return on the SML
    for stock in stock_table.index:

        beta = stock_table.loc[stock, "Beta"]
        actual = stock_table.loc[stock, "Annual Return"] * 100
        required = stock_table.loc[stock, "CAPM Required Return"] * 100

        if actual >= required:
            color = GAIN     # above SML: positive alpha (green)
        else:
            color = LOSS     # below SML: negative alpha (red)

        # Dotted line = the stock's Jensen's alpha
        ax.vlines(beta, required, actual, colors=color,
                  linestyles="dotted", linewidth=1.5)
        ax.scatter(beta, actual, s=90, color=color, zorder=3)
        ax.annotate(stock, (beta, actual), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)

    # Market portfolio (beta = 1) and risk-free asset (beta = 0)
    ax.scatter(1, market_return * 100, marker="D", s=80,
               color=NAVY, zorder=3, label=f"Market ({market_column})")
    ax.scatter(0, annual_rf * 100, marker="s", s=60,
               color=STEEL, zorder=3, label="Risk-free rate")

    ax.axhline(0, color=NAVY, linewidth=0.6)
    ax.set_xlabel("Beta (systematic risk)")
    ax.set_ylabel("Annual Return (%)")
    ax.set_title("Security Market Line (green = above, red = below)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig


def plot_risk_return_scatter(stock_table, market_summary,
                             market_column, selected_stocks):
    """Chart 4: each stock's annual volatility vs annual return."""

    fig, ax = plt.subplots(figsize=(9, 5))

    for stock in stock_table.index:

        volatility = stock_table.loc[stock, "Annual Volatility"] * 100
        annual_return = stock_table.loc[stock, "Annual Return"] * 100

        if stock in selected_stocks:
            color = BLUE     # selected: dark blue
        else:
            color = GREY     # not selected: grey

        ax.scatter(volatility, annual_return, s=90, color=color, zorder=3)
        ax.annotate(stock, (volatility, annual_return), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)

    # Market for comparison
    ax.scatter(market_summary["Market Annual Volatility"] * 100,
               market_summary["Market Annual Return"] * 100,
               marker="D", s=80, color=NAVY, zorder=3,
               label=f"Market ({market_column})")

    ax.axhline(0, color=NAVY, linewidth=0.6)
    ax.set_xlabel("Annual Volatility (%)  ->  more risk")
    ax.set_ylabel("Annual Return (%)")
    ax.set_title("Risk vs Return of Each Stock (dark blue = selected)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig




def plot_weights_chart(weights_series):
    """Chart 8: optimized risky portfolio weights."""

    weights_percent = weights_series.sort_values() * 100

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(weights_percent.index, weights_percent.values, color=BLUE)

    for bar, value in zip(bars, weights_percent.values):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", va="center", fontsize=10)

    ax.set_xlim(0, 110)
    ax.set_xlabel("Weight in risky portfolio (%)")
    ax.set_title("Optimized Risky Portfolio Weights (Maximum Sharpe Ratio)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    return fig
def plot_efficient_frontier(frontier, random_portfolios, stock_points,
                            optimal_risk, optimal_return):
    """Charts 5 and 6: Efficient Frontier with the maximum Sharpe portfolio."""

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Feasible portfolios (light grey cloud)
    ax.scatter(random_portfolios["Risk"] * 100, random_portfolios["Return"] * 100,
               s=6, color=CLOUD, alpha=0.6, label="Possible portfolios")

    # Efficient Frontier (optimized line)
    if len(frontier) > 1:
        ax.plot(frontier["Risk"] * 100, frontier["Return"] * 100,
                color=BLUE, linewidth=2.5, label="Efficient Frontier")

    # Individual selected stocks
    for stock, row in stock_points.iterrows():
        ax.scatter(row["Risk"] * 100, row["Return"] * 100, s=60,
                   color=STEEL, zorder=3)
        ax.annotate(stock, (row["Risk"] * 100, row["Return"] * 100),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)

    # Maximum Sharpe (tangency) portfolio
    ax.scatter(optimal_risk * 100, optimal_return * 100, marker="*", s=350,
               color=GOLD, edgecolor=NAVY, zorder=4,
               label="Maximum Sharpe portfolio")

    ax.set_xlabel("Annual Risk (Volatility, %)")
    ax.set_ylabel("Expected Annual Return (%)")
    ax.set_title("Efficient Frontier of Selected Stocks")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig


def plot_capital_allocation_line(frontier, annual_rf, optimal_risk,
                                 optimal_return, optimal_sharpe):
    """
    Chart 7: Capital Allocation Line (CAL).
    E(Rc) = Rf + Sharpe x Sigma_c
    Solid part: mixes of the risk-free asset and the risky portfolio.
    Dashed part: would need borrowing (leverage), not allowed in SmartInvest.
    """

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Lending part of the CAL: 0% to 100% in the risky portfolio
    lending_risk = np.linspace(0, optimal_risk, 50)
    lending_return = annual_rf + optimal_sharpe * lending_risk

    # Borrowing part of the CAL: above 100% (shown for theory only)
    borrowing_risk = np.linspace(optimal_risk, optimal_risk * 1.4, 20)
    borrowing_return = annual_rf + optimal_sharpe * borrowing_risk

    ax.plot(lending_risk * 100, lending_return * 100, color=NAVY,
            linewidth=2.5, label="CAL (risk-free + risky portfolio)")
    ax.plot(borrowing_risk * 100, borrowing_return * 100, color=NAVY,
            linewidth=1.5, linestyle="--", label="CAL with borrowing (not used)")

    if len(frontier) > 1:
        ax.plot(frontier["Risk"] * 100, frontier["Return"] * 100,
                color=BLUE, linewidth=2, alpha=0.7, label="Efficient Frontier")

    ax.scatter(0, annual_rf * 100, marker="s", s=70, color=STEEL,
               zorder=4, label="Risk-free asset")
    ax.scatter(optimal_risk * 100, optimal_return * 100, marker="*", s=350,
               color=GOLD, edgecolor=NAVY, zorder=4,
               label="Maximum Sharpe portfolio")

    ax.set_xlabel("Annual Risk (Volatility, %)")
    ax.set_ylabel("Expected Annual Return (%)")
    ax.set_title(f"Capital Allocation Line (slope = Sharpe ratio = {optimal_sharpe:.3f})")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig
def plot_risky_vs_riskfree(risky_investment, risk_free_investment):
    """Chart 9: split of the budget between risky and risk-free."""

    labels = []
    values = []
    colors = []

    if risky_investment > 0:
        labels.append("Risky portfolio")
        values.append(risky_investment)
        colors.append(BLUE)

    if risk_free_investment > 0:
        labels.append("Risk-free asset")
        values.append(risk_free_investment)
        colors.append(LIGHT_BLUE)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_title("Budget Split: Risky vs Risk-Free")
    plt.tight_layout()

    return fig


def plot_rupee_allocation(allocation_table):
    """Chart 10: final rupee amount in each investment."""

    amounts = allocation_table["Amount (Rs)"]

    colors = []
    for investment_type in allocation_table["Type"]:
        if investment_type == "Risk-free":
            colors.append(LIGHT_BLUE)
        else:
            colors.append(BLUE)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(amounts.index, amounts.values, color=colors)

    for bar, value in zip(bars, amounts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                f"Rs {value:,.0f}", ha="center", va="bottom", fontsize=9)

    ax.margins(y=0.15)
    ax.set_ylabel("Amount (Rs)")
    ax.set_title("Final Investment Amount by Security")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return fig
